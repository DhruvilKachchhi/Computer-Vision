# 🎥 Optical Flow Computation

A Python script that computes and visualises **dense** (Farneback) and **sparse** (Lucas-Kanade) optical flow from any video file.  
All outputs are automatically saved to the `Output/` sub-folder.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Output Files](#output-files)
- [Quick Start (Batch File)](#quick-start-batch-file)
- [Manual Setup](#manual-setup)
- [Command-Line Usage](#command-line-usage)
- [Visual Graphs & Charts](#visual-graphs--charts)
- [How the Visualisation Works](#how-the-visualisation-works)
- [Interpreting the Results](#interpreting-the-results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [References](#references)

---

## Overview

**Optical flow** describes the apparent motion of pixels between consecutive video frames.  
This tool implements two complementary methods:

| Method | Type | Algorithm |
|---|---|---|
| **Dense flow** | Every pixel gets a flow vector | Farneback polynomial expansion \[1\] |
| **Sparse flow** | Tracked keypoints only | Lucas-Kanade pyramidal \[2\] on Shi-Tomasi corners \[3\] |

All results are written to the `Output/` sub-folder so you can inspect each visualisation independently.

---

## Output Files

All outputs are saved to the **`Output/`** sub-folder created automatically next to the script.

| File | Type | Description |
|---|---|---|
| `optical_flow_output.mp4` | Video | Side-by-side: **left** = source frame with moving/static region overlay; **right** = dense flow in HSV colour + sparse arrow overlay |
| `flow_heatmap_output.mp4` | Video | Magnitude heatmap blended over the source video (JET colormap) with a live colour-scale legend |
| `motion_bbox_output.mp4` | Video | Motion bounding boxes drawn around every detected moving region, labelled with mean magnitude |
| `flow_magnitude_plot.png` | Chart | ⭐ Improved time-series of average flow magnitude per frame — dark theme, smoothed trend, spike annotations, activity bands |
| `polar_direction_histogram.png` | Chart | ⭐ Improved polar rose chart — dark theme, HSV-matched bar colours, top-3 direction annotations, mean-direction needle |
| `accumulated_motion_energy.png` | Chart | ⭐ **NEW** — Single spatial heatmap: per-pixel sum of all flow magnitudes across the whole video |
| Console | — | Per-frame table (magnitude, dominant direction, % moving pixels, box count) + summary statistics |

### Main composite frame layout

```
┌──────────────────────────┬──────────────────────────┐
│  Source + region mask    │  Dense flow (HSV) +      │
│                          │  Lucas-Kanade arrows      │
│  Green tint = moving     │  Hue  = direction         │
│  Dark  tint = static     │  Bright = fast motion     │
└──────────────────────────┴──────────────────────────┘
```

---

## Visual Graphs & Charts

### 1 · Per-Frame Magnitude Time-Series (`flow_magnitude_plot.png`)

Line chart of **average flow magnitude over time** (in px/frame).  
**Best for:** activity detection, motion bursts, scene changes.

| Feature | Description |
|---|---|
| **Raw signal** (thin blue line) | Per-frame average magnitude |
| **Smoothed trend** (orange line) | Rolling mean over ~1 s window — removes noise, shows true motion envelope |
| **Shaded bands** | 🟢 Green = static (< threshold) · 🟠 Orange = active · 🔴 Red = high-activity spikes |
| **Gold markers** | Top-5 spike frames annotated with frame number and exact value |
| **Secondary axis** | Cumulative motion energy (purple dotted line) — shows total motion accumulation |
| **Footer bar** | Mean / Median / Peak / total active seconds at a glance |

---

### 2 · Polar Direction Histogram (`polar_direction_histogram.png`)

Bar chart plotted on a circle (0°–360°) — a **Rose Diagram**.  
**Best for:** distinguishing camera pan vs. object motion vs. zoom.

| Feature | Description |
|---|---|
| **36 bars** (10° each) | Each bar's radius = magnitude-weighted frequency of motion in that direction |
| **HSV-matched colours** | Bar hue matches the optical-flow colour wheel in the main video |
| **Top-3 annotations** | Gold / Silver / Bronze labels showing the dominant directions and their share % |
| **Mean-direction needle** | White arrow pointing to the circular mean of all flow vectors |
| **Dark background** | High-contrast design matching the other output charts |

| Pattern | Interpretation |
|---|---|
| Single dominant bar | Consistent unidirectional motion — camera pan or object moving one way |
| Two opposing bars (180° apart) | Back-and-forth or zoom motion |
| Symmetric ring | Radial / zoom flow — camera moving forward or backward |
| Scattered bars | Complex or chaotic motion |

---

### 3 · Accumulated Motion Energy Heatmap (`accumulated_motion_energy.png`)

A single static image: **sum of all per-frame flow magnitudes** at every pixel across the entire video.  
**Best for:** identifying moving object travel paths at a glance.

| Feature | Description |
|---|---|
| **Inferno colormap** | Black → purple → orange → yellow — high contrast on dark backgrounds |
| **Contour lines** | Cyan / green / yellow / red outlines at the 50th / 75th / 90th / 98th percentile |
| **Colour-bar** | Labelled scale in accumulated px/frame units |
| **Gaussian smoothing** | σ = 3 px removes single-pixel noise before plotting |

Bright yellow/white areas = where objects moved most or fastest over the whole video.  
Dark/black areas = background regions that were always static.

---

## Quick Start (Batch File)

> **Windows only.** Double-click or run from any terminal.

### Just use the video already in the folder

```bat
run_optical_flow.bat
```

The batch file will:
1. Verify Python is installed.
2. Create (or reuse) the `venv\` virtual environment.
3. Install / update all required packages (including `scipy`).
4. Auto-detect the first video in the folder and run the analysis.
5. Save all outputs to `Output\`.

### Pass your own video

```bat
run_optical_flow.bat "my video.mp4"
```

### Full argument form

```bat
run_optical_flow.bat  [input]  [output]  [plot]  [heatmap]  [bbox]  [polar]  [energy]  [arrow-interval]
```

| Position | Maps to | Default |
|---|---|---|
| `%1` | `--input` | first video in folder |
| `%2` | `--output` | `Output\optical_flow_output.mp4` |
| `%3` | `--plot` | `Output\flow_magnitude_plot.png` |
| `%4` | `--heatmap` | `Output\flow_heatmap_output.mp4` |
| `%5` | `--bbox` | `Output\motion_bbox_output.mp4` |
| `%6` | `--polar` | `Output\polar_direction_histogram.png` |
| `%7` | `--energy` | `Output\accumulated_motion_energy.png` |
| `%8` | `--arrow-interval` | `5` |

---

## Manual Setup

### 1 — Clone / download the project

```bash
git clone <repo-url>
cd "Optical Flow Computation"
```

### 2 — Create a virtual environment

```bash
python -m venv venv
```

### 3 — Activate the virtual environment

**Windows (Command Prompt / batch)**

```bat
venv\Scripts\activate.bat
```

**Windows (PowerShell)**

```powershell
venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Command-Line Usage

```
python optical_flow.py [OPTIONS]
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--input PATH` | `-i` | *(auto-detect)* | Path to the input video. If omitted the script searches the folder for any `.mp4 / .avi / .mov / .mkv` file; if none found it downloads a sample. |
| `--output PATH` | `-o` | `Output/optical_flow_output.mp4` | Main composite output video (HSV flow + region mask). |
| `--plot PATH` | `-p` | `Output/flow_magnitude_plot.png` | Where to save the magnitude time-series PNG chart. |
| `--heatmap PATH` | | `Output/flow_heatmap_output.mp4` | Magnitude heatmap overlay video. |
| `--bbox PATH` | | `Output/motion_bbox_output.mp4` | Motion bounding-box video. |
| `--polar PATH` | | `Output/polar_direction_histogram.png` | Polar direction histogram PNG. |
| `--energy PATH` | | `Output/accumulated_motion_energy.png` | Accumulated motion energy heatmap PNG. |
| `--arrow-interval N` | `-n` | `5` | Draw Lucas-Kanade sparse arrows every **N** frames. |

### Examples

```bash
# Auto-detect the first video in the folder
python optical_flow.py

# Use a specific input video
python optical_flow.py --input "input/people-detection.mp4"

# Custom output paths, arrows every 10 frames
python optical_flow.py -i "input/car-detection.mp4" -o result.mp4 -p plot.png -n 10

# Override only the energy map path
python optical_flow.py -i "Video 1.mp4" --energy my_energy.png
```

---

## How the Visualisation Works

### Dense Flow → HSV Colour Encoding \[6\]

Each pixel in the flow field produces a 2-D vector `(dx, dy)`.  
These vectors are encoded as a colour image in **HSV** space:

| HSV Channel | Encodes | Interpretation |
|---|---|---|
| **Hue** | Flow direction (angle) | Colour wheel shows which way pixels moved |
| **Saturation** | Fixed at 255 | Always fully saturated |
| **Value** | Flow magnitude | Bright = fast; dark = slow / static |

### Sparse Flow → Lucas-Kanade Arrows \[2, 3\]

Every `--arrow-interval` frames:
1. **Shi-Tomasi** corner detection finds strong features in the previous frame.
2. **Lucas-Kanade** pyramidal tracker estimates where each feature moved.
3. **Cyan arrows** are drawn from old position → new position on the HSV frame.

### Moving / Static Region Overlay

A magnitude threshold (`2.0 px/frame` by default) creates a binary mask:

- **Green tint** → pixel magnitude > threshold → moving region
- **Dark red tint** → pixel magnitude ≤ threshold → static background

### Magnitude Heatmap (Video)

Flow magnitude at every pixel is normalised to `[0, 255]` against a running exponential moving-average peak, then mapped through OpenCV's `COLORMAP_JET`.  
The result is alpha-blended with the source frame (`α = 0.55`).  
A colour-scale legend bar is drawn on the right edge for reference.

### Motion Bounding Boxes

1. Compute per-pixel magnitude; threshold to get a binary motion mask.
2. Apply morphological **close** (fill holes) then **open** (remove noise) then **dilate** (expand slightly).
3. Find external contours; filter by minimum area (`500 px²` default).
4. Draw a labelled green rectangle around each surviving contour.

### Per-Frame Magnitude Time-Series (Improved)

- Raw per-frame average magnitudes are plotted as a thin signal line.
- A rolling-mean **smoothed trend** (window ≈ 1 s) is overlaid.
- Activity is colour-coded into static / active / spike bands.
- The **top-5 peak frames** are annotated with frame number and value.
- A **secondary axis** tracks cumulative motion energy over time.

### Polar Direction Histogram (Improved)

At each frame, every pixel's flow angle is added—weighted by its magnitude—into one of 36 bins (10° each).  
After all frames, the accumulated bin counts are plotted on a polar axes with:
- HSV-matched bar colours (consistent with the flow video hue wheel).
- A **mean-direction needle** (circular mean of all weighted vectors).
- **Top-3 direction annotations** showing their percentage share.

### Accumulated Motion Energy Heatmap (New)

`energy_accumulator[y, x] += magnitude[y, x]` for every frame.  
After processing, the accumulated map is lightly Gaussian-smoothed (σ = 3 px) and rendered with the **Inferno** colormap plus percentile contour lines to delineate the most active spatial zones.

---

## Interpreting the Results

### Flow Magnitude

> *"How fast did each pixel move?"*

- **High** average magnitude → rapid camera movement or fast object motion.
- **Low** average magnitude → mostly static scene.
- The **magnitude plot** shows this over time with colour-coded activity bands.
- The **heatmap video** shows spatial distribution of speed per frame.
- The **accumulated energy map** shows where speed was concentrated over the whole video.

### Flow Direction (Hue / Polar Histogram)

> *"Which way did pixels move?"*

| Pattern | Meaning |
|---|---|
| Uniform hue across the frame | Camera pan (horizontal) or tilt (vertical) |
| Radially **diverging** from centre | Camera moving **forward** (looming) |
| Radially **converging** toward centre | Camera moving **backward** |
| Local hue anomaly vs. background | Independently moving object (IMO) |
| Single dominant bar in polar chart | Consistent unidirectional motion (e.g. pan) |
| Symmetric opposing bars | Back-and-forth or zoom motion |

### Motion Bounding Boxes

> *"Where are the moving objects?"*

- Each box wraps a spatially contiguous group of moving pixels.
- The label shows the **mean flow magnitude** inside that region in px/frame.
- Morphological pre-processing means small noise blobs are suppressed; only genuine motion regions are boxed.

### Accumulated Energy Map

> *"Where did motion happen most over the whole video?"*

- Bright (yellow/white) regions = objects that moved most or fastest.
- Dark (black) regions = always-static background.
- Contour lines label the 50 / 75 / 90 / 98th percentile activity zones — useful for identifying travel paths or hotspots.

### Dominant Direction (Console Output)

Each frame reports the **weighted circular mean** of all flow angles as one of 8 compass directions:  
`E  NE  N  NW  W  SW  S  SE`

### Time-to-Collision (TTC) Estimation \[5\]

When the camera moves directly toward a surface, the FOE (Focus of Expansion) is the point where flow = 0.  
For any pixel at distance `d` from the FOE with local magnitude `m`:

```
τ (seconds-to-collision) ≈ d / m
```

- **Shrinking τ** → you are approaching a collision.
- **Growing τ** → you are receding.

---

## Project Structure

```
Optical Flow Computation/
├── optical_flow.py                  # Main Python script
├── run_optical_flow.bat             # One-click Windows launcher
├── requirements.txt                 # Pinned package versions (incl. scipy)
├── venv/                            # Virtual environment (auto-created)
├── input/                           # Sample input videos
└── Output/                          # ← All generated files go here
    ├── optical_flow_output.mp4          Generated: main composite video
    ├── flow_heatmap_output.mp4          Generated: magnitude heatmap video
    ├── motion_bbox_output.mp4           Generated: bounding-box video
    ├── flow_magnitude_plot.png          Generated: magnitude time-series chart
    ├── polar_direction_histogram.png    Generated: polar direction rose chart
    └── accumulated_motion_energy.png    Generated: spatial energy heatmap
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥ 4.8 | Video I/O, Farneback flow, Lucas-Kanade, corner detection, colormaps |
| `numpy` | ≥ 1.24 | Array maths, flow statistics, angle accumulation |
| `matplotlib` | ≥ 3.7 | All charts: time-series, polar histogram, energy heatmap |
| `scipy` | ≥ 1.11 | Gaussian smoothing of the accumulated energy map |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## References

| # | Citation |
|---|---|
| \[1\] | Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion.* SCIA, LNCS 2749, pp. 363–370. https://doi.org/10.1007/3-540-45103-X_50 |
| \[2\] | Lucas, B. D., & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision.* IJCAI, pp. 674–679. |
| \[3\] | Shi, J., & Tomasi, C. (1994). *Good Features to Track.* CVPR, pp. 593–600. https://doi.org/10.1109/CVPR.1994.323794 |
| \[4\] | Horn, B. K. P., & Schunck, B. G. (1981). *Determining Optical Flow.* Artificial Intelligence, 17(1–3), 185–203. https://doi.org/10.1016/0004-3702(81)90024-2 |
| \[5\] | Gibson, J. J. (1950). *The Perception of the Visual World.* Houghton Mifflin. |
| \[6\] | OpenCV Docs – `cv2.calcOpticalFlowFarneback`. https://docs.opencv.org/4.x/ |
| \[7\] | OpenCV Docs – `cv2.calcOpticalFlowPyrLK`. https://docs.opencv.org/4.x/ |
| \[8\] | OpenCV Docs – `cv2.goodFeaturesToTrack`. https://docs.opencv.org/4.x/ |
| \[9\] | Beauchemin, S. S., & Barron, J. L. (1995). *The Computation of Optical Flow.* ACM Computing Surveys, 27(3), 433–466. https://doi.org/10.1145/212094.212141 |
| \[10\] | Fleet, D. J., & Weiss, Y. (2006). *Optical Flow Estimation.* Handbook of Mathematical Models in Computer Vision, pp. 237–257. https://doi.org/10.1007/0-387-28831-7_15 |

---

> **Tip:** The `venv\` folder is local and can be safely deleted and regenerated by re-running `run_optical_flow.bat`.
