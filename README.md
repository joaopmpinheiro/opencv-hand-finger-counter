# Hand Detection and Finger Counting

## Project Overview

This project includes two computer vision prototypes:
1. **Skin Tone Detection**: Detects and outlines skin tones ranging from light to dark brown
2. **Finger Counting**: Counts the number of extended fingers on a hand using convexity defects

## How to Use

### Setup

**Select your video device**: In `VideoCapture()`, specify the device number:
- `0` - Primary device (typically built-in webcam)
- `1` - Secondary device
- `2` - Third device, etc.

```python
cap = cv2.VideoCapture(1)  # Adjust based on your setup
```

### Calibration Phase

Both versions begin with a calibration phase that captures background frames to train the background subtractor. Keep your hand out of the frame during this period.

## Setup Notes & Known Issues

### My Configuration

- **Camera**: Camo Studio (iPhone as webcam)
  - All camera settings locked (exposure, white balance, etc.)
  - MacBook's native webcam has dynamic exposure causing significant brightness fluctuations that interfere with background subtraction
  
- **Background**: Black poster board with a custom cardboard stand
  - Available upon request if needed for replication

### Known Limitations

**Version 2 (Finger Counting)**:
- Removed `skin_mask` due to lighting instability
- Consequently, detection works with any color, not exclusively skin tones
- Unable to detect closed fist (zero fingers extended)

**Lighting Considerations**:
- Automatic camera adjustments (auto-exposure, auto-white balance) significantly degrade background subtraction performance
- Static camera settings are essential for reliable detection

## Requirements

```bash
pip install opencv-python numpy
```

## Running the Program

```bash
python number_count.py
```

Press `q` to exit.

---

**Recommendation**: For optimal results, use a consistent background and fixed camera settings. A controlled lighting environment substantially improves detection accuracy.
