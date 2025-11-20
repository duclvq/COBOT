# RealSense Hand Tracking

## Setup
```bash
pip install pyrealsense2 opencv-python mediapipe numpy
```

## Quick Start

### 1. Calibrate Camera
```bash
python realsense_calibrate.py
```
- Press `s` to save images (10+ images from different angles)
- Press `q` to finish capture
- Press `c` to set chessboard as coordinate origin

### 2. Run Hand Tracking
```bash
python realsense_reader.py
```
- `q` - Quit
- `s` - Save images
- `c` - Detect chessboard origin

## Output
- Wrist position in chessboard coordinates (X, Y, Z in mm)
- Distance from origin
- RGB + Depth visualization

## Files
- `realsense_calibrate.py` - Camera calibration
- `realsense_reader.py` - Hand tracking
- `realsense_hand_transform.py` - Coordinate transformation
- `matrix/` - Calibration data
- `capture/` - Saved images