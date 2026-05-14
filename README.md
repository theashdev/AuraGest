# AuraGest
A futuristic real-time hand tracking system that uses webcam-based gesture recognition to create dynamic animated circles and dual-hand effects using thumb and index finger movements with smooth tracking, low latency, and immersive visual interactions.

`open_camera.bat` starts a standalone webcam window with real-time two-hand
tracking, index fingertip recognition, movement direction, tap/hover detection,
simple gestures, FPS, debug overlay, and animated thumb-index circle effects.

## First-time setup

Install the tracking libraries once:

```powershell
py -m pip install -r requirements.txt
```

The app also needs the MediaPipe hand landmark model file named
`hand_landmarker.task` in this folder.

## Run

Double-click `open_camera.bat`, or run:

```powershell
py open_camera.py
```

Press `q` or `Esc` to close the app.

## Gestures

- Open palm
- Fist
- Index point
- Pinch
- Tap
- Hover
- Swipe left/right/up/down

Spread your thumb and index fingertip apart on either hand to show a glowing
circular frame between them. Each hand gets its own smoothed ring. If both hands
activate the gesture together, the app adds a synchronized dual-hand bridge and
larger center pulse between both circles.

The code is organized around `GestureEngine`, `CameraProcessor`, and `Overlay`
so more gestures or controls can be added without rewriting the camera loop.
