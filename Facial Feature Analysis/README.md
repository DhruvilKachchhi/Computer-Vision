# Facial Analysis Pipeline

Automated batch video analysis that extracts **blink rate** and **facial dimension metrics** using MediaPipe Face Mesh (468 landmarks) and OpenCV. Produces a timestamped Markdown report and CSV data file after processing an entire folder of videos.

---

## Features

| Feature | Detail |
|---------|--------|
| **Blink Detection** | Eye Aspect Ratio (EAR) threshold method (Soukupová & Čech, 2016) |
| **Facial Dimensions** | Face, Eyes (L/R + avg), Nose, Mouth — width & height in pixels |
| **Batch Processing** | Scans a folder recursively for `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv` |
| **Reports** | Timestamped Markdown report + raw CSV file |
| **Error Handling** | Skips videos with no detectable face; logs all warnings |
| **Progress Display** | Per-frame progress bar via `tqdm` |

---

## Project Structure

```
Mini Project/
├── facial_analysis.py      ← Main script (all logic is here)
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── venv/                   ← Virtual environment (created during setup)
└── Videos/                 ← Default input folder
    ├── GX010968.MP4
    ├── GX010969.MP4
    └── ...
```

---

## Prerequisites

- **Python 3.9 or newer** (tested on 3.11.9)
- **Windows 10/11, macOS, or Linux**
- Microsoft Visual C++ Redistributable (Windows only — usually already installed)

---

## Setup Instructions

### Step 1 — Clone / Download the project

Make sure all files are in a single folder, e.g. `C:\Repositories\Mini Project`.

### Step 2 — Create a virtual environment

Open a **PowerShell** or **Command Prompt** window in the project folder:

```powershell
# PowerShell
python -m venv venv
```

```cmd
# Command Prompt (cmd.exe)
python -m venv venv
```

### Step 3 — Activate the virtual environment

```powershell
# PowerShell
.\venv\Scripts\Activate.ps1
```

```cmd
# Command Prompt
venv\Scripts\activate.bat
```

You should see `(venv)` appear at the start of your prompt.

> **PowerShell Execution Policy:** If you get a "cannot be loaded" error, run:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `opencv-python`, `mediapipe`, `numpy`, `scipy`, `tqdm` and their transitive dependencies.

### Step 5 — Run the pipeline

```bash
# Process the default ./Videos folder
python facial_analysis.py

# Or specify a custom folder
python facial_analysis.py --video_dir "C:\path\to\your\videos"

# Specify both input folder and output report destination
python facial_analysis.py --video_dir ./Videos --report_dir ./Reports
```

---

## Output

After the run completes, two timestamped files are written to the **report directory** (defaults to the same folder as the videos):

| File | Description |
|------|-------------|
| `facial_analysis_report_YYYYMMDD_HHMMSS.md` | Full Markdown report with per-video tables and aggregate summary |
| `facial_analysis_data_YYYYMMDD_HHMMSS.csv`  | Raw numeric data for every processed video (import into Excel, pandas, etc.) |

### Sample Markdown Report Structure

```
# Facial Analysis Pipeline — Summary Report

## Measurement Legend
## Per-Video Results
  ### 1. GX010968.MP4
      Video Info table
      Blink Analysis table
      Facial Dimensions table
  ### 2. GX010969.MP4
      ...
## Aggregate Summary (Averages Across All Videos)
## All-Video Data Table  ← compact comparison table with AVG row
```

---

## Configuration

All tunable parameters are at the **top of `facial_analysis.py`** (Section 1):

```python
# ── Blink Detection ──────────────────────────────────────────
EAR_THRESHOLD      = 0.20   # EAR < this → eye "closed"   (range: 0.18–0.25)
EAR_CONSEC_FRAMES  = 2      # consecutive closed frames → one blink

# ── Processing Speed ─────────────────────────────────────────
FRAME_SKIP         = 1      # process every Nth frame (1=all, 2=every other, …)
FACE_DETECTION_GRACE_FRAMES = 90  # frames to search before skipping a video

# ── MediaPipe Sensitivity ────────────────────────────────────
MP_DETECTION_CONFIDENCE = 0.5
MP_TRACKING_CONFIDENCE  = 0.5
```

### Tuning Guide

| Parameter | Lower value | Higher value |
|-----------|-------------|--------------|
| `EAR_THRESHOLD` | Detects only deep/slow blinks | May cause false positives from squints |
| `EAR_CONSEC_FRAMES` | Catches fast blinks | Ignores brief eye movements |
| `FRAME_SKIP` | More accurate blink timing | Faster processing |
| `MP_DETECTION_CONFIDENCE` | Detects more faces (inc. partial) | Fewer false faces |

---

## How It Works

### Task A — Blink Rate (EAR Method)

1. MediaPipe extracts 6 landmark points around each eye per frame.
2. The **Eye Aspect Ratio** is computed:

   ```
   EAR = (||P2–P6|| + ||P3–P5||) / (2 × ||P1–P4||)
   ```

   Open eye ≈ 0.25–0.35 · Closed eye ≈ 0.0

3. When the averaged EAR (left + right) drops below `EAR_THRESHOLD` for
   `EAR_CONSEC_FRAMES` consecutive frames, one blink is counted.
4. **Blink rate** = Total blinks ÷ Video duration (seconds).

### Task B — Facial Dimensions

Landmark pairs are used to measure pixel distances for each feature:

| Feature | Width landmarks | Height landmarks |
|---------|----------------|-----------------|
| Face    | #234 (L ear) → #454 (R ear) | #10 (forehead) → #152 (chin) |
| Eyes    | corner-to-corner | max of two vertical EAR pairs |
| Nose    | #129 (L ala) → #358 (R ala) | #6 (bridge) → #2 (tip) |
| Mouth   | #61 (L corner) → #291 (R corner) | #13 (upper lip) → #14 (lower lip) |

The **maximum** value observed across all frames is reported per video
(captures the frame where the face is best oriented toward the camera).

### Pixel → Real-World Conversion

Pixel distances can be converted to centimetres using a known reference:

```
scale_factor = known_reference_cm / known_reference_px

# Example: if the subject's interpupillary distance (≈6.3 cm)
# measures 120 px in the video:
scale_factor = 6.3 / 120 = 0.0525 cm/px

face_width_cm = face_width_px × scale_factor
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'cv2'` | Activate the venv and run `pip install -r requirements.txt` |
| `No face detected` warning for every video | Check lighting; try lowering `MP_DETECTION_CONFIDENCE` to `0.3` |
| Blink count seems too high | Increase `EAR_CONSEC_FRAMES` to 3 or raise `EAR_THRESHOLD` slightly |
| Blink count seems too low | Lower `EAR_THRESHOLD` to `0.18` or reduce `EAR_CONSEC_FRAMES` to `1` |
| Script is slow on long videos | Set `FRAME_SKIP = 2` or `3` to process fewer frames |
| PowerShell script execution blocked | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Dependencies

| Package | Version (min) | Purpose |
|---------|--------------|---------|
| `opencv-python` | ≥ 4.8.0 | Video decoding, frame I/O, BGR→RGB conversion |
| `mediapipe` | ≥ 0.10.0 | 468-point Face Mesh landmark detection |
| `numpy` | ≥ 1.24.0 | Pixel coordinate arrays and arithmetic |
| `scipy` | ≥ 1.10.0 | Euclidean distance computation |
| `tqdm` | ≥ 4.65.0 | Terminal progress bars |

---

_Facial Analysis Pipeline — built with OpenCV + MediaPipe Face Mesh_
