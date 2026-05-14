from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque


try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as exc:
    missing = exc.name or "a required package"
    print()
    print(f"Missing dependency: {missing}")
    print("Install the camera tracking packages once with:")
    print("  py -m pip install -r requirements.txt")
    print()
    print("Then run:")
    print("  py open_camera.py")
    print()
    sys.exit(1)


WINDOW_NAME = "HandCam Studio"
CAMERA_INDEX = 0
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


class Gesture(str, Enum):
    NO_HAND = "No hand"
    OPEN_PALM = "Open palm"
    FIST = "Fist"
    POINT = "Index point"
    PINCH = "Pinch"
    TAP = "Tap"
    HOVER = "Hover"
    SWIPE_LEFT = "Swipe left"
    SWIPE_RIGHT = "Swipe right"
    SWIPE_UP = "Swipe up"
    SWIPE_DOWN = "Swipe down"


@dataclass(slots=True)
class TrackerConfig:
    camera_width: int = 1280
    camera_height: int = 720
    target_fps: int = 60
    model_complexity: int = 1
    min_detection_confidence: float = 0.65
    min_tracking_confidence: float = 0.65
    smoothing_alpha: float = 0.42
    movement_deadzone_px: float = 5.5
    swipe_speed_px_s: float = 850.0
    tap_speed_px_s: float = 720.0
    tap_depth_delta: float = 0.045
    hover_speed_px_s: float = 42.0
    hover_radius_px: float = 20.0
    hover_time_s: float = 0.65
    pinch_distance_ratio: float = 0.36
    history_seconds: float = 1.3


@dataclass(slots=True)
class Point2D:
    x: float
    y: float

    def as_int(self) -> tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    def distance_to(self, other: "Point2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(slots=True)
class FingerSample:
    at: float
    point: Point2D
    depth: float


@dataclass(slots=True)
class TrackingState:
    hand_label: str = "Hand"
    gesture: Gesture = Gesture.NO_HAND
    direction: str = "Still"
    index_point: Point2D | None = None
    thumb_point: Point2D | None = None
    thumb_index_distance: float = 0.0
    velocity_px_s: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    speed_px_s: float = 0.0
    confidence: float = 0.0
    raised_fingers: tuple[str, ...] = ()
    debug: dict[str, str] = field(default_factory=dict)


class FingerHistory:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.samples: Deque[FingerSample] = deque()

    def add(self, sample: FingerSample) -> None:
        self.samples.append(sample)
        cutoff = sample.at - self.seconds
        while self.samples and self.samples[0].at < cutoff:
            self.samples.popleft()

    def clear(self) -> None:
        self.samples.clear()

    def latest(self) -> FingerSample | None:
        return self.samples[-1] if self.samples else None

    def previous(self, min_age: float = 0.06) -> FingerSample | None:
        if not self.samples:
            return None
        latest = self.samples[-1]
        for sample in reversed(self.samples):
            if latest.at - sample.at >= min_age:
                return sample
        return None

    def stable_inside(self, radius: float, duration: float) -> bool:
        if len(self.samples) < 4:
            return False
        latest = self.samples[-1]
        window = [s for s in self.samples if latest.at - s.at <= duration]
        if len(window) < 4:
            return False
        return all(s.point.distance_to(latest.point) <= radius for s in window)


class IndexFingerSmoother:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.point: Point2D | None = None

    def update(self, point: Point2D) -> Point2D:
        if self.point is None:
            self.point = point
            return point
        self.point = Point2D(
            self.alpha * point.x + (1.0 - self.alpha) * self.point.x,
            self.alpha * point.y + (1.0 - self.alpha) * self.point.y,
        )
        return self.point

    def reset(self) -> None:
        self.point = None


class GestureEngine:
    FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
    TIP_IDS = (4, 8, 12, 16, 20)
    PIP_IDS = (3, 6, 10, 14, 18)

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.history = FingerHistory(config.history_seconds)
        self.smoother = IndexFingerSmoother(config.smoothing_alpha)
        self.thumb_smoother = IndexFingerSmoother(config.smoothing_alpha)
        self.last_tap_at = 0.0

    def reset(self, hand_label: str = "Hand") -> TrackingState:
        self.history.clear()
        self.smoother.reset()
        self.thumb_smoother.reset()
        return TrackingState(hand_label=hand_label)

    def analyze(self, landmarks, width: int, height: int, handedness_score: float, hand_label: str) -> TrackingState:
        now = time.perf_counter()
        points = [Point2D(lm.x * width, lm.y * height) for lm in landmarks]
        index_tip = self.smoother.update(points[8])
        thumb_tip = self.thumb_smoother.update(points[4])
        thumb_index_distance = thumb_tip.distance_to(index_tip)
        index_depth = landmarks[8].z
        self.history.add(FingerSample(now, index_tip, index_depth))

        velocity, speed = self._velocity()
        direction = self._direction(velocity)
        raised = self._raised_fingers(points)
        gesture = self._gesture(points, raised, velocity, speed)

        return TrackingState(
            hand_label=hand_label,
            gesture=gesture,
            direction=direction,
            index_point=index_tip,
            thumb_point=thumb_tip,
            thumb_index_distance=thumb_index_distance,
            velocity_px_s=velocity,
            speed_px_s=speed,
            confidence=handedness_score,
            raised_fingers=raised,
            debug={
                "speed": f"{speed:0.0f}px/s",
                "vx": f"{velocity.x:0.0f}",
                "vy": f"{velocity.y:0.0f}",
                "circle": f"{thumb_index_distance:0.0f}px",
                "raised": ", ".join(raised) if raised else "none",
            },
        )

    def _velocity(self) -> tuple[Point2D, float]:
        latest = self.history.latest()
        previous = self.history.previous()
        if latest is None or previous is None:
            return Point2D(0.0, 0.0), 0.0
        dt = max(latest.at - previous.at, 1e-5)
        vx = (latest.point.x - previous.point.x) / dt
        vy = (latest.point.y - previous.point.y) / dt
        if math.hypot(vx, vy) < self.config.movement_deadzone_px / dt:
            vx, vy = 0.0, 0.0
        velocity = Point2D(vx, vy)
        return velocity, math.hypot(vx, vy)

    def _direction(self, velocity: Point2D) -> str:
        if math.hypot(velocity.x, velocity.y) < 80:
            return "Still"
        if abs(velocity.x) > abs(velocity.y):
            return "Right" if velocity.x > 0 else "Left"
        return "Down" if velocity.y > 0 else "Up"

    def _raised_fingers(self, points: list[Point2D]) -> tuple[str, ...]:
        raised: list[str] = []
        wrist = points[0]

        thumb_tip = points[4]
        thumb_ip = points[3]
        index_mcp = points[5]
        pinky_mcp = points[17]
        palm_width = max(index_mcp.distance_to(pinky_mcp), 1.0)
        if thumb_tip.distance_to(wrist) > thumb_ip.distance_to(wrist) + palm_width * 0.18:
            raised.append("thumb")

        for name, tip_id, pip_id in zip(self.FINGER_NAMES[1:], self.TIP_IDS[1:], self.PIP_IDS[1:]):
            tip = points[tip_id]
            pip = points[pip_id]
            mcp = points[tip_id - 3]
            if tip.y < pip.y and tip.distance_to(wrist) > mcp.distance_to(wrist) * 1.08:
                raised.append(name)

        return tuple(raised)

    def _gesture(
        self,
        points: list[Point2D],
        raised: tuple[str, ...],
        velocity: Point2D,
        speed: float,
    ) -> Gesture:
        palm_size = max(points[0].distance_to(points[9]), 1.0)
        pinch_distance = points[4].distance_to(points[8])
        latest = self.history.latest()
        previous = self.history.previous(min_age=0.10)

        if speed > self.config.swipe_speed_px_s:
            if abs(velocity.x) > abs(velocity.y) * 1.35:
                return Gesture.SWIPE_RIGHT if velocity.x > 0 else Gesture.SWIPE_LEFT
            if abs(velocity.y) > abs(velocity.x) * 1.35:
                return Gesture.SWIPE_DOWN if velocity.y > 0 else Gesture.SWIPE_UP

        if (
            latest is not None
            and previous is not None
            and time.perf_counter() - self.last_tap_at > 0.35
            and speed > self.config.tap_speed_px_s
            and previous.depth - latest.depth > self.config.tap_depth_delta
        ):
            self.last_tap_at = time.perf_counter()
            return Gesture.TAP

        if pinch_distance < palm_size * self.config.pinch_distance_ratio:
            return Gesture.PINCH

        if (
            speed < self.config.hover_speed_px_s
            and self.history.stable_inside(self.config.hover_radius_px, self.config.hover_time_s)
        ):
            return Gesture.HOVER

        if len(raised) >= 4:
            return Gesture.OPEN_PALM

        if len(raised) == 0:
            return Gesture.FIST

        if raised == ("index",):
            return Gesture.POINT

        return Gesture.POINT if "index" in raised else Gesture.OPEN_PALM


class CameraProcessor:
    def __init__(self) -> None:
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def improve_lighting(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        improved_l = self.clahe.apply(l_channel)
        improved = cv2.merge((improved_l, a_channel, b_channel))
        return cv2.cvtColor(improved, cv2.COLOR_LAB2BGR)


@dataclass(slots=True)
class CircleEffectState:
    center: Point2D | None = None
    radius: float = 0.0
    opacity: float = 0.0
    last_update: float = 0.0


class FingerCircleEffect:
    def __init__(self, color: tuple[int, int, int]) -> None:
        self.state = CircleEffectState()
        self.color = color

    def draw(self, frame, state: TrackingState) -> None:
        self._update(state)
        if self.state.center is None or self.state.opacity < 0.02 or self.state.radius < 8:
            return

        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 5.2)
        center = self.state.center.as_int()
        radius = int(round(self.state.radius + pulse * 2.0))
        opacity = self.state.opacity

        glow = frame.copy()
        for scale, alpha, thickness in (
            (1.42, 0.10, 18),
            (1.20, 0.16, 12),
            (1.02, 0.34, 4),
        ):
            cv2.circle(
                glow,
                center,
                max(1, int(radius * scale)),
                self.color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.addWeighted(glow, alpha * opacity, frame, 1.0 - alpha * opacity, 0, frame)
            glow = frame.copy()

        cv2.circle(frame, center, radius, (252, 248, 232), 2, cv2.LINE_AA)
        cv2.circle(frame, center, max(1, int(radius * 0.72)), self.color, 1, cv2.LINE_AA)
        self._draw_orbit_ticks(frame, center, radius, opacity)

    def active_center_radius_opacity(self) -> tuple[Point2D, float, float] | None:
        if self.state.center is None or self.state.opacity < 0.08 or self.state.radius < 8:
            return None
        return self.state.center, self.state.radius, self.state.opacity

    def _update(self, tracking: TrackingState) -> None:
        now = time.perf_counter()
        dt = min(max(now - self.state.last_update, 0.0), 0.08) if self.state.last_update else 0.016
        self.state.last_update = now

        active = (
            tracking.thumb_point is not None
            and tracking.index_point is not None
            and tracking.thumb_index_distance >= 58
        )
        target_opacity = 1.0 if active else 0.0
        self.state.opacity = self._smooth(self.state.opacity, target_opacity, dt, 15.0)

        if not active:
            return

        assert tracking.thumb_point is not None
        assert tracking.index_point is not None
        target_center = Point2D(
            (tracking.thumb_point.x + tracking.index_point.x) * 0.5,
            (tracking.thumb_point.y + tracking.index_point.y) * 0.5,
        )
        target_radius = max(24.0, min(tracking.thumb_index_distance * 0.5, 220.0))

        if self.state.center is None:
            self.state.center = target_center
            self.state.radius = target_radius
            return

        alpha = 1.0 - math.exp(-18.0 * dt)
        self.state.center = Point2D(
            self.state.center.x + (target_center.x - self.state.center.x) * alpha,
            self.state.center.y + (target_center.y - self.state.center.y) * alpha,
        )
        self.state.radius = self.state.radius + (target_radius - self.state.radius) * alpha

    def _draw_orbit_ticks(self, frame, center: tuple[int, int], radius: int, opacity: float) -> None:
        angle_offset = time.perf_counter() * 1.8
        tick_overlay = frame.copy()
        for i in range(16):
            angle = angle_offset + i * (math.tau / 16)
            if i % 2:
                length = 9
                color = (75, 220, 255)
            else:
                length = 14
                color = self.color
            inner = radius + 8
            outer = radius + 8 + length
            p1 = (
                int(center[0] + math.cos(angle) * inner),
                int(center[1] + math.sin(angle) * inner),
            )
            p2 = (
                int(center[0] + math.cos(angle) * outer),
                int(center[1] + math.sin(angle) * outer),
            )
            cv2.line(tick_overlay, p1, p2, color, 2, cv2.LINE_AA)
        cv2.addWeighted(tick_overlay, 0.72 * opacity, frame, 1.0 - 0.72 * opacity, 0, frame)

    def _smooth(self, current: float, target: float, dt: float, speed: float) -> float:
        return current + (target - current) * (1.0 - math.exp(-speed * dt))


class Overlay:
    def __init__(self) -> None:
        self.connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        self.circle_effects = {
            "Left": FingerCircleEffect((255, 195, 70)),
            "Right": FingerCircleEffect((75, 220, 255)),
        }
        self.dual_opacity = 0.0
        self.dual_last_update = 0.0

    def draw(self, frame, landmarks_list, states: list[TrackingState], fps: float) -> None:
        height, width = frame.shape[:2]

        for landmarks in landmarks_list:
            self._draw_landmarks(frame, landmarks, width, height)

        for state in states:
            self._effect_for(state.hand_label).draw(frame, state)

        self._draw_dual_hand_effect(frame)
        primary_state = states[0] if states else TrackingState()
        self._draw_top_panel(frame, states, fps)
        for state in states:
            self._draw_index_marker(frame, state)
        self._draw_debug_panel(frame, primary_state, states, width, height)

    def _draw_top_panel(self, frame, states: list[TrackingState], fps: float) -> None:
        panel_x, panel_y = 18, 18
        panel_w, panel_h = 430, 132
        self._rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, (18, 22, 28), 0.76)
        cv2.putText(frame, "HandCam Studio", (panel_x + 20, panel_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 248, 255), 2)
        gesture_text = " | ".join(f"{s.hand_label}: {s.gesture.value}" for s in states) if states else "No hands"
        direction_text = " | ".join(f"{s.hand_label}: {s.direction}" for s in states) if states else "Still"
        cv2.putText(frame, f"Gesture: {gesture_text[:34]}", (panel_x + 20, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (118, 224, 168), 2)
        cv2.putText(frame, f"Direction: {direction_text[:34]}", (panel_x + 20, panel_y + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 215, 230), 1)
        cv2.putText(frame, f"FPS {fps:04.1f}", (panel_x + 318, panel_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 206, 255), 2)

    def _draw_index_marker(self, frame, state: TrackingState) -> None:
        if state.index_point is None:
            return
        x, y = state.index_point.as_int()
        cv2.circle(frame, (x, y), 18, (255, 255, 255), 2)
        cv2.circle(frame, (x, y), 7, (75, 220, 255), -1)
        vx = int(max(min(state.velocity_px_s.x * 0.065, 90), -90))
        vy = int(max(min(state.velocity_px_s.y * 0.065, 90), -90))
        if abs(vx) + abs(vy) > 8:
            cv2.arrowedLine(frame, (x, y), (x + vx, y + vy), (75, 220, 255), 3, tipLength=0.35)

    def _draw_landmarks(self, frame, landmarks, width: int, height: int) -> None:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for connection in self.connections:
            start = points[connection.start]
            end = points[connection.end]
            cv2.line(frame, start, end, (72, 190, 255), 2, cv2.LINE_AA)

        for index, point in enumerate(points):
            radius = 5 if index in (4, 8, 12, 16, 20) else 4
            cv2.circle(frame, point, radius + 2, (18, 22, 28), -1, cv2.LINE_AA)
            cv2.circle(frame, point, radius, (238, 246, 255), -1, cv2.LINE_AA)

    def _draw_debug_panel(self, frame, state: TrackingState, states: list[TrackingState], width: int, height: int) -> None:
        panel_w, panel_h = 355, 158
        panel_x, panel_y = width - panel_w - 18, height - panel_h - 18
        self._rounded_rect(frame, panel_x, panel_y, panel_w, panel_h, (18, 22, 28), 0.62)
        lines = [
            f"Hands: {len(states)}",
            f"{state.hand_label} confidence: {state.confidence:0.2f}",
            f"Index: {self._point_text(state.index_point)}",
            f"Circle: {state.debug.get('circle', '0px')}",
            f"Velocity: {state.debug.get('vx', '0')}, {state.debug.get('vy', '0')}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (panel_x + 18, panel_y + 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (224, 231, 242),
                1,
            )

    def _point_text(self, point: Point2D | None) -> str:
        if point is None:
            return "--"
        x, y = point.as_int()
        return f"{x}, {y}"

    def _rounded_rect(self, frame, x: int, y: int, width: int, height: int, color: tuple[int, int, int], alpha: float) -> None:
        overlay = frame.copy()
        radius = 12
        x2, y2 = x + width, y + height
        cv2.rectangle(overlay, (x + radius, y), (x2 - radius, y2), color, -1)
        cv2.rectangle(overlay, (x, y + radius), (x2, y2 - radius), color, -1)
        cv2.circle(overlay, (x + radius, y + radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y + radius), radius, color, -1)
        cv2.circle(overlay, (x + radius, y2 - radius), radius, color, -1)
        cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _effect_for(self, hand_label: str) -> FingerCircleEffect:
        return self.circle_effects.get(hand_label, self.circle_effects["Left"])

    def _draw_dual_hand_effect(self, frame) -> None:
        now = time.perf_counter()
        dt = min(max(now - self.dual_last_update, 0.0), 0.08) if self.dual_last_update else 0.016
        self.dual_last_update = now
        left = self.circle_effects["Left"].active_center_radius_opacity()
        right = self.circle_effects["Right"].active_center_radius_opacity()
        target = min(left[2], right[2]) if left and right else 0.0
        self.dual_opacity += (target - self.dual_opacity) * (1.0 - math.exp(-12.0 * dt))
        if self.dual_opacity < 0.03 or not left or not right:
            return

        left_center, left_radius, _ = left
        right_center, right_radius, _ = right
        p1 = left_center.as_int()
        p2 = right_center.as_int()
        mid = Point2D((left_center.x + right_center.x) * 0.5, (left_center.y + right_center.y) * 0.5).as_int()
        bridge = frame.copy()
        pulse = 0.5 + 0.5 * math.sin(now * 6.0)
        cv2.line(bridge, p1, p2, (255, 245, 185), 8, cv2.LINE_AA)
        cv2.line(bridge, p1, p2, (75, 220, 255), 3, cv2.LINE_AA)
        cv2.addWeighted(bridge, 0.35 * self.dual_opacity, frame, 1.0 - 0.35 * self.dual_opacity, 0, frame)

        dual_radius = int(max(34, min(left_center.distance_to(right_center) * 0.28 + (left_radius + right_radius) * 0.18, 260)))
        cv2.circle(frame, mid, int(dual_radius + pulse * 6), (255, 245, 185), 3, cv2.LINE_AA)
        cv2.circle(frame, mid, int(dual_radius * 0.72), (75, 220, 255), 1, cv2.LINE_AA)


class HandCameraApp:
    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.processor = CameraProcessor()
        self.gestures = {
            "Left": GestureEngine(config),
            "Right": GestureEngine(config),
        }
        self.overlay = Overlay()
        self.frame_times: Deque[float] = deque(maxlen=40)

    def run(self) -> int:
        if not MODEL_PATH.exists():
            print(f"Missing model file: {MODEL_PATH.name}")
            print("Download hand_landmarker.task into this folder, then run the app again.")
            print("Recommended URL:")
            print("  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            return 1

        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            print("Could not open the webcam. Check camera permissions or close another app using it.")
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.config.camera_width, self.config.camera_height)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_tracking_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        hands = mp.tasks.vision.HandLandmarker.create_from_options(options)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame was not available.")
                    return 1

                frame = cv2.flip(frame, 1)
                frame = self.processor.improve_lighting(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp_ms = int(time.perf_counter() * 1000)
                result = hands.detect_for_video(image, timestamp_ms)

                landmarks_list = result.hand_landmarks or []
                states: list[TrackingState] = []
                seen_labels: set[str] = set()
                for hand_index, landmarks in enumerate(landmarks_list):
                    label, score = self._handedness(result, hand_index)
                    seen_labels.add(label)
                    engine = self.gestures[label]
                    states.append(engine.analyze(landmarks, frame.shape[1], frame.shape[0], score, label))

                for label, engine in self.gestures.items():
                    if label not in seen_labels:
                        engine.reset(label)

                fps = self._fps()
                self.overlay.draw(frame, landmarks_list, states, fps)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    return 0
        finally:
            hands.close()
            cap.release()
            cv2.destroyAllWindows()

    def _handedness(self, result, hand_index: int) -> tuple[str, float]:
        if not result.handedness or hand_index >= len(result.handedness) or not result.handedness[hand_index]:
            return ("Left" if hand_index == 0 else "Right"), 0.0
        category = result.handedness[hand_index][0]
        label = str(category.category_name)
        if label not in self.gestures:
            label = "Left" if hand_index == 0 else "Right"
        return label, float(category.score)

    def _fps(self) -> float:
        now = time.perf_counter()
        self.frame_times.append(now)
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / max(elapsed, 1e-5)


def main() -> int:
    return HandCameraApp(TrackerConfig()).run()


if __name__ == "__main__":
    raise SystemExit(main())
