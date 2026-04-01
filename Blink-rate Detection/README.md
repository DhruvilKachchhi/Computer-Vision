# 👁 Eye Blink Rate Detector — Movie vs. Document Reading

Measure and compare your **eye blink rate** while watching a movie clip versus reading a document. Uses real-time webcam-based detection with MediaPipe Face Mesh and EAR (Eye Aspect Ratio) analysis.

---

## Features

| Feature | Details |
|---------|---------|
| 🎬 Movie Mode | Plays any MP4/AVI/MOV clip in-app while tracking blinks |
| 📄 Document Mode | Renders page 1 of PDF or DOCX while tracking blinks |
| 🔬 EAR Detection | MediaPipe Face Mesh → Eye Aspect Ratio → blink counting |
| ⚙️ Auto-calibration | 5-second baseline calibrates EAR threshold to your eyes |
| 📊 Comparison Chart | Matplotlib bar chart comparing blinks/min and blinks/sec |
| ⚠️ Edge case handling | Poor lighting warning, face-not-detected alerts, webcam error handling |

---

## Requirements

### System
- **Python 3.10+**
- **Webcam** (built-in or USB)
- **LibreOffice** (for DOCX support — optional)
  - macOS: `brew install libreoffice`
  - Ubuntu/Debian: `sudo apt install libreoffice`
  - Windows: Download from https://www.libreoffice.org/

### Python Packages
```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone or download this folder
cd blink_detector/

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run the app
python main.py
```

---

## Usage

### Step-by-step
1. **Launch** the app — you'll see two session buttons
2. **Click "Start Movie Session"** → pick a video file (MP4/AVI/MOV)
3. A **5-second calibration phase** begins (keep eyes open, look at camera)
4. The video plays with **audio**; the webcam tracks your blinks for **60 seconds**
5. Session ends → **results popup** shows total blinks, blinks/sec, blinks/min
6. **Click "Start Document Session"** → pick a PDF or DOCX file
7. Repeat calibration + 60-second session (document is now scrollable with mouse wheel)
8. Click **"View Comparison Chart"** to see the side-by-side analysis

### Tips for best results
- Sit in **good lighting** (avoid backlighting)
- Position your face **straight-on** to the camera
- Keep your head **relatively still** during calibration
- Normal adult blink rate is **12–20 blinks/minute** at rest

---

## Project Structure

```
blink_detector/
├── main.py              # App entry point and main window
├── blink_detector.py    # EAR calculation, MediaPipe, blink counting
├── session_manager.py   # Session lifecycle, file picker, countdown
├── document_viewer.py   # PDF/DOCX rendering via PyMuPDF + LibreOffice
├── video_player.py      # OpenCV-based in-app video playback
├── results.py           # Stats display and matplotlib comparison chart
├── requirements.txt
└── README.md
```

---

## How Blink Detection Works

### Eye Aspect Ratio (EAR)
```
EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)
```
Where p1–p6 are the 6 key landmarks around each eye (corners + top/bottom lids).

- **Open eye** → EAR ≈ 0.25–0.35
- **Closed eye / blink** → EAR < 0.21 (default threshold)

### Auto-calibration
During the 5-second baseline phase, the app computes your mean EAR with eyes open, then sets:
```
threshold = mean_EAR − 0.10
```
This adapts to your eye shape, glasses, lighting, and camera distance.

### Blink counting
A blink is registered when EAR drops below the threshold for **2+ consecutive frames**, preventing false positives from single-frame noise.

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| "Could not open webcam" | Check another app isn't using the camera; try `cv2.VideoCapture(1)` |
| Face not detected | Improve lighting; move closer to camera; remove glasses |
| DOCX not rendering | Install LibreOffice and ensure `libreoffice` is in your PATH |
| Video plays slowly | Reduce video resolution or ensure sufficient CPU headroom |
| `mediapipe` install fails | Try `pip install mediapipe --pre` or use Python 3.10–3.11 |

---

## Research Context

Studies show that **blink rate decreases significantly** during screen-based cognitive tasks:
- Average resting blink rate: **~17 blinks/minute**
- During computer reading: **~7 blinks/minute** (60% reduction)
- During video viewing: **~10–14 blinks/minute**

This sustained reduction contributes to **Computer Vision Syndrome** (dry eyes, fatigue). The 20-20-20 rule is recommended: every 20 minutes, look at something 20 feet away for 20 seconds.

---

