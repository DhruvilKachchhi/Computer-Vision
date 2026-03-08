# Lucas-Kanade Optical Flow — Theory, Visualisation & Validation

> **Derive** motion-tracking and bilinear-interpolation equations from first
> principles, then **validate** them numerically against consecutive video frames
> and produce six rich visualisations — three static PNGs and three animated MP4s.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Mathematical Derivations](#4-mathematical-derivations)
   - 4.1 [Motion Tracking — Lucas-Kanade Optical Flow](#41-motion-tracking--lucas-kanade-optical-flow)
   - 4.2 [Bilinear Interpolation](#42-bilinear-interpolation)
   - 4.3 [Two-Frame Problem Setup](#43-two-frame-problem-setup)
5. [Implementation Details](#5-implementation-details)
6. [Validation Methodology](#6-validation-methodology)
7. [Output Files](#7-output-files)
8. [Command-Line Reference](#8-command-line-reference)
9. [Dependencies](#9-dependencies)

---

## 1. Project Overview

This project provides:

| Component | Description |
|-----------|-------------|
| **Full derivation** | Lucas-Kanade optical flow from the brightness constancy equation through the 2×2 normal equations |
| **Bilinear interpolation** | Built from 1-D linear interpolation, derived step-by-step |
| **Manual LK solver** | Educational single-point LK implementation that exactly mirrors the derivation |
| **Practical validation** | Tracks features between two consecutive frames, compares bilinear-interpolated intensities against nearest-neighbour ground truth |
| **Dense HSV map (PNG + MP4)** | Farneback per-pixel flow: hue = direction, brightness = speed |
| **Quiver arrow field (PNG + MP4)** | LK motion vectors overlaid on frames, coloured by magnitude |
| **Eigenvalue scatter plot (PNG)** | λ₁ vs λ₂ of the structure tensor — visualises corner/edge/flat classification |
| **Trajectory trails (MP4)** | LK comet-tail paths drawn in real time across all frames |

Both synthetic (known ground-truth flow) and real video sources are supported.

---

## 2. Repository Structure

```
Bilinear Interpolation/
├── lucas_kanade_tracker.py   # main script — theory + tracking + all visualisations
├── requirements.txt          # Python dependencies
├── setup.bat                 # one-click: create venv + install packages
├── run.bat                   # one-click: activate venv + run script
├── activate_env.bat          # open a shell with the venv active
├── venv/                     # virtual environment (created by setup.bat)
├── output/                   # all outputs saved here (created at runtime)
│   │
│   │  ── Static PNGs ──────────────────────────────────────────
│   ├── lk_theory.png                      full derivation as dark-themed figure
│   ├── lk_tracking_<tag>.png              side-by-side frames + motion arrows
│   ├── lk_bilinear_detail_<tag>.png       zoom into bilinear 2×2 neighbourhood
│   ├── lk_errors_<tag>.png                per-point Δ_interp + ΔBright bar charts
│   ├── lk_dense_hsv_<tag>.png             dense Farneback HSV colour map (single pair)
│   ├── lk_quiver_<tag>.png                quiver arrow field (Frame 1 → Frame 2)
│   └── lk_eigenvalue_scatter_<tag>.png    λ₁ vs λ₂ scatter + Shi-Tomasi bar chart
│   │
│   │  ── Animated MP4s ─────────────────────────────────────────
│   ├── vid_dense_hsv_<tag>.mp4            per-frame HSV flow blended onto video
│   ├── vid_trails_<tag>.mp4               LK comet-tail trajectory trails
│   └── vid_quiver_<tag>.mp4              LK quiver arrow field overlay (live)
└── README.md
```

---

## 3. Quick Start

### Windows (recommended)

```bat
setup.bat          REM one-time: creates venv/, installs packages

run.bat                                    REM synthetic ground-truth frames
run.bat --video "Video 1.mp4"              REM single real video
run.bat --video a.mp4 --video b.mp4        REM multiple videos
run.bat --video clip.mp4 --outdir results  REM custom output folder
run.bat --video clip.mp4 --no-theory       REM skip theory printout
```

### Any platform

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python lucas_kanade_tracker.py                          # synthetic
python lucas_kanade_tracker.py --video clip.mp4         # real video
python lucas_kanade_tracker.py --video clip.mp4 --frame1 10 --frame2 11
python lucas_kanade_tracker.py --outdir results --no-theory
```

---

## 4. Mathematical Derivations

### 4.1 Motion Tracking — Lucas-Kanade Optical Flow

#### Problem Setup: Two Consecutive Frames

Let **I₁(x, y)** and **I₂(x, y)** be two consecutive grayscale image frames.
A feature point is located at **(x₀, y₀)** in Frame 1.
The goal is to find its displaced location **(x₀ + u, y₀ + v)** in Frame 2,
where **(u, v)** is the unknown optical flow vector.

```
Frame 1                     Frame 2
┌──────────────────┐        ┌──────────────────┐
│                  │        │                  │
│   ● (x₀,y₀)     │   →    │       ● (x₀+u, y₀+v)
│                  │  flow  │                  │
└──────────────────┘  (u,v) └──────────────────┘
```

---

#### Step 1 — Brightness Constancy Assumption

The fundamental assumption: a physical scene point preserves its brightness
as it moves between frames.

```
I(x, y, t)  =  I(x + dx,  y + dy,  t + dt)                   … (1)
```

where `(dx, dy)` is the 2-D displacement during time step `dt`.

This is exact for rigid objects under constant illumination. Real scenes
introduce small violations due to lighting changes, motion blur, and occlusion.

---

#### Step 2 — First-Order Taylor Expansion

Expand the right-hand side of (1) about the point `(x, y, t)`,
retaining only first-order (linear) terms:

```
I(x+dx, y+dy, t+dt)
    ≈  I(x,y,t)
     + (∂I/∂x)·dx
     + (∂I/∂y)·dy
     + (∂I/∂t)·dt                                              … (2)
```

Substitute (2) into (1) and cancel `I(x,y,t)` from both sides:

```
(∂I/∂x)·dx  +  (∂I/∂y)·dy  +  (∂I/∂t)·dt  =  0              … (3)
```

---

#### Step 3 — Optical Flow Constraint Equation (OFCE)

Divide (3) by `dt` and define pixel velocity `u = dx/dt`, `v = dy/dt`:

```
Iₓ·u  +  Iᵧ·v  +  I_t  =  0                                  … (OFCE)
```

**Notation:**
- `Iₓ = ∂I/∂x` — horizontal spatial gradient
- `Iᵧ = ∂I/∂y` — vertical spatial gradient
- `I_t = ∂I/∂t ≈ I₂(x,y) − I₁(x,y)` — temporal gradient

> ⚠ **The Aperture Problem.** The OFCE is **one equation in two unknowns** (u, v).
> Geometrically, the solution lies on a *constraint line* in (u,v)-space:
> `u·Iₓ + v·Iᵧ = −I_t`. Only the component of motion *perpendicular to an
> edge* can be recovered from a single pixel.

---

#### Step 4 — Lucas-Kanade: Constant-Flow Window Assumption

**Key insight** (Lucas & Kanade, 1981):
Assume `(u, v)` is **constant** over a small neighbourhood window **W** of *N* pixels.

Each pixel `i ∈ W` yields one OFCE:

```
Iₓᵢ·u  +  Iᵧᵢ·v  =  −I_tᵢ       for i = 1, 2, …, N
```

Stacked into matrix form **A · p = b**:

```
    ⎡ Iₓ₁  Iᵧ₁ ⎤ ⎡ u ⎤   ⎡ −I_t₁ ⎤
    ⎢ Iₓ₂  Iᵧ₂ ⎥ ⎣ v ⎦ = ⎢ −I_t₂ ⎥
    ⎢  ⋮    ⋮  ⎥           ⎢   ⋮   ⎥
    ⎣ Iₓₙ  Iᵧₙ ⎦           ⎣ −I_tₙ ⎦
```

This is an **over-determined** system (N ≫ 2) — solved by least squares.

---

#### Step 5 — Weighted Least-Squares Solution

Minimise the weighted squared residual over the window, with a **Gaussian
weight kernel W** (pixels near the centre matter more):

```
min Σᵢ wᵢ (Iₓᵢu + Iᵧᵢv + I_tᵢ)²
```

Setting ∂/∂u = 0 and ∂/∂v = 0 gives the **normal equations**:

```
AᵀWA · p  =  AᵀW b
```

Expanding (writing Σ for Σᵢ wᵢ):

```
⎡ ΣIₓ²    ΣIₓIᵧ ⎤ ⎡ u ⎤   ⎡ −ΣIₓI_t ⎤
⎣ ΣIₓIᵧ  ΣIᵧ²  ⎦ ⎣ v ⎦ = ⎣ −ΣIᵧI_t ⎦
```

Define the **Structure Tensor**:

```
        ⎡ ΣIₓ²    ΣIₓIᵧ ⎤
M  =    ⎣ ΣIₓIᵧ  ΣIᵧ²  ⎦    (= AᵀWA)
```

The **closed-form solution** is:

```
⎡ u ⎤              ⎡ ΣIᵧ²  −ΣIₓIᵧ ⎤ ⎡ −ΣIₓI_t ⎤
⎣ v ⎦  =  (1/det M) ⎣ −ΣIₓIᵧ  ΣIₓ² ⎦ ⎣ −ΣIᵧI_t ⎦

where  det M  =  ΣIₓ² · ΣIᵧ²  −  (ΣIₓIᵧ)²
```

**Trackability from eigenvalues λ₁ ≥ λ₂ of M:**

| Condition | Scene type | Trackability |
|-----------|-----------|--------------|
| λ₁ ≫ 1, λ₂ ≫ 1 | Corner | ✅ Well-conditioned |
| λ₁ ≫ λ₂ ≈ 0 | Edge | ⚠ Aperture problem |
| λ₁ ≈ λ₂ ≈ 0 | Flat region | ❌ No texture |

**Shi-Tomasi feature selection:** `min(λ₁, λ₂) > threshold`
→ implemented as `cv2.goodFeaturesToTrack`.

---

#### Step 6 — Pyramid Extension for Large Displacements

The Taylor expansion assumes small displacements (`dx, dy ≪ 1`).
For large motions, build a **Gaussian image pyramid**:

```
Level L (½ res):   solve for (u_L, v_L)  (coarse estimate)
Level L-1:         warp by 2·(u_L, v_L), solve for residual
…
Level 0 (full):    accumulate → final (u, v)
```

This is `cv2.calcOpticalFlowPyrLK` (Bouguet, 2001), used throughout this project.

---

### 4.2 Bilinear Interpolation

#### Why Interpolation is Needed

LK tracking produces **sub-pixel** coordinates, e.g., `(34.73, 21.18)`.
Image data exists only at **integer** grid points.
We must estimate intensity at fractional locations by interpolating neighbours.

---

#### Step A — 1-D Linear Interpolation

Given values `f₀` at `x = 0` and `f₁` at `x = 1`,
estimate `f` at a fractional position `x = a` where `0 ≤ a ≤ 1`:

```
f(a)  =  (1 − a) · f₀  +  a · f₁                             … (L1D)
```

**Geometric meaning:** each weight equals the distance to the *opposite* endpoint.
The two weights `(1−a)` and `a` always sum to 1 → convex combination.

---

#### Step B — Extend to 2-D

Query point: `(x, y)` with:
- `x₀ = floor(x)`,  fractional offset `a = x − x₀`
- `y₀ = floor(y)`,  fractional offset `b = y − y₀`

**Four surrounding integer-grid neighbours:**

```
              x₀          x₀+1
        y₀    Q₁₁ ──────── Q₂₁
               │              │
               │   P(x,y)     │
               │     ★        │
        y₀+1  Q₁₂ ──────── Q₂₂
```

**Step 1 — Interpolate along top row** (y = y₀, varying x):

```
R₁  =  (1 − a) · Q₁₁  +  a · Q₂₁
```

**Step 2 — Interpolate along bottom row** (y = y₀+1, varying x):

```
R₂  =  (1 − a) · Q₁₂  +  a · Q₂₂
```

**Step 3 — Interpolate vertically** between R₁ and R₂:

```
f(x, y)  =  (1 − b) · R₁  +  b · R₂
```

**Substituting R₁ and R₂ and expanding:**

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│  f(x,y) = (1−a)(1−b) · Q₁₁  +  a(1−b) · Q₂₁                 │
│         + (1−a)·b    · Q₁₂  +  a·b    · Q₂₂                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Area / weight interpretation:**

Each weight equals the **area of the opposite sub-rectangle** in the unit cell:

```
 Weight of Q₁₁  =  (1−a)(1−b)   ← area of rectangle opposite to Q₁₁
 Weight of Q₂₁  =  a(1−b)
 Weight of Q₁₂  =  (1−a)·b
 Weight of Q₂₂  =  a·b
```

All four weights sum to `(1−a)(1−b) + a(1−b) + (1−a)b + ab = 1`.

> This is a **convex combination** — the result always lies within the range
> `[min(Q₁₁,Q₂₁,Q₁₂,Q₂₂), max(Q₁₁,Q₂₁,Q₁₂,Q₂₂)]`. No extrapolation occurs.

---

### 4.3 Two-Frame Problem Setup

Given consecutive frames **I₁** and **I₂**:

| Symbol | Meaning |
|--------|---------|
| `(x₀, y₀)` | Feature location in Frame 1 (integer, from Shi-Tomasi) |
| `Iₓ, Iᵧ` | Spatial gradients computed from Frame 1 via Sobel filters |
| `I_t` | Temporal difference: `I₂(x,y) − I₁(x,y)` |
| `W` | 21×21 Gaussian-weighted window around `(x₀, y₀)` |
| `M = AᵀWA` | 2×2 Structure Tensor |
| `(u, v)` | Predicted sub-pixel displacement (LK solution) |
| `(px, py)` | Predicted location in Frame 2: `px = x₀ + u`, `py = y₀ + v` |
| `I_bilinear` | Intensity at `(px, py)` in Frame 2 via bilinear interpolation |
| `I_NN` | Intensity at `(round(px), round(py))` in Frame 2 (nearest neighbour) |

**Validation checks:**

```
Δ_interp  =  |I_bilinear − I_NN|           (should be small, < ~5 units)
ΔBright   =  |I₁(x₀,y₀) − I_bilinear|     (brightness constancy residual)
u_manual  ≈  u_opencv                       (derivation cross-check)
```

---

## 5. Implementation Details

### `bilinear_interp(img, x, y)`
Hand-coded implementation of the formula derived in §4.2.
Boundary-clamps indices; fractional offsets `a, b ∈ [0, 1]` guaranteed.

### `lk_manual(img1, img2, pt, win)`
Educational single-point LK solver:
1. Computes Sobel spatial gradients `Iₓ, Iᵧ` and temporal difference `I_t`
2. Extracts a `(2·win+1)²` window with Gaussian weights
3. Builds the 2×2 structure tensor M
4. Solves via explicit 2×2 inversion (Cramer's rule)
5. Returns `(u, v, condition_number)`

Cross-checking `(u_man, v_man)` against OpenCV's `(dx, dy)` validates the derivation.

### `make_synthetic_frames()`
Generates Gaussian-blob frames with known constant flow `dx=2, dy=1 px/frame`.
The synthetic ground truth lets you confirm the algorithm recovers the exact displacement.

### `run_video_visualisations(video_path, outdir)`
Processes every frame of the source video using OpenCV alone (no matplotlib overhead):
- **Dense HSV** — Farneback flow, built into HSV frame, blended 70 % flow / 30 % original
- **Trajectory trails** — 120 LK points, 25-frame fading comet tails drawn onto a persistent canvas; points re-detected periodically to maintain coverage
- **Quiver field** — `cv2.arrowedLine` from previous to current point per track, arrow length proportional to speed

---

## 6. Validation Methodology

For each pair of consecutive frames and each tracked feature point:

```
Frame 1                   Frame 2
  ●  (x₀, y₀)   →LK→   ★  (px, py)   [sub-pixel, predicted]
  I₁_src                   I_bilinear  [bilinear formula]
                           I_NN        [nearest-neighbour ground truth]
```

**Three independent checks:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `Δ_interp` | `|I_bilinear − I_NN|` | Bilinear accuracy vs integer pixel |
| `ΔBright` | `|I₁_src − I_bilinear|` | Brightness constancy validity |
| `u_man ≈ dx` | `|u_manual − dx_opencv|` | Manual LK matches cv2 |

**Expected results on synthetic data:**
- `Δ_interp` < 2 intensity units (sub-pixel shift within smooth blob)
- `ΔBright` < 5 intensity units (Gaussian blobs shift cleanly)
- `u_man ≈ 2.0`, `v_man ≈ 1.0` matching known ground truth

---

## 7. Output Files

All files are saved to `output/` (or `--outdir` path).

### Static PNGs — generated for every run

| File | Contents |
|------|----------|
| `lk_theory.png` | Full derivation rendered as a dark-themed monospace figure |
| `lk_tracking_<tag>.png` | Side-by-side frames with labelled feature points and motion arrows |
| `lk_bilinear_detail_<tag>.png` | Zoom into each tracked point's 2×2 bilinear neighbourhood with annotated weights |
| `lk_errors_<tag>.png` | Bar charts of `Δ_interp` and `ΔBright` per tracked point |
| `lk_dense_hsv_<tag>.png` | Dense Farneback HSV colour map (frames 0↔1 pair) |
| `lk_quiver_<tag>.png` | Quiver arrow field on Frame 1 + Frame 2 side-by-side with colourbar |
| `lk_eigenvalue_scatter_<tag>.png` | λ₁ vs λ₂ scatter plot with corner/edge/flat regions + Shi-Tomasi bar chart |

### Animated MP4s — generated when `--video` is supplied

| File | Contents |
|------|----------|
| `vid_dense_hsv_<tag>.mp4` | 🥇 Every frame rendered as Farneback HSV flow blended onto the original — the definitive optical flow video |
| `vid_trails_<tag>.mp4` | 🥈 120 LK tracked points draw growing comet-tail paths across all frames — long-exposure photograph effect |
| `vid_quiver_<tag>.mp4` | 🥉 Per-frame quiver arrows overlaid on the scene — fast regions get long arrows, slow regions get short ones |

---

## 8. Command-Line Reference

```
python lucas_kanade_tracker.py [options]

Options:
  --video PATH      Video file to process (mp4, avi, mov, …).
                    Repeat to process multiple videos in one run.
                    Omit to use the built-in synthetic sequence.
  --frame1 N        First frame index for static PNGs (default: 0).
  --frame2 N        Second frame index for static PNGs (default: 1).
  --outdir DIR      Output directory for all outputs (default: output/).
  --no-theory       Skip printing the theory derivation to stdout.
  -h, --help        Show this help message.
```

**Examples:**

```bat
REM Synthetic ground-truth validation (no video needed)
run.bat

REM Single real video — produces all 10 outputs
run.bat --video "Video 1.mp4"

REM Two videos compared
run.bat --video scene_a.mp4 --video scene_b.mp4

REM Pick specific frame pair for static PNGs
run.bat --video footage.mp4 --frame1 25 --frame2 26

REM Save to custom folder, no theory printout
run.bat --video footage.mp4 --outdir my_results --no-theory
```

---

## 9. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.8 | Feature detection, LK pyramid, Farneback flow, video I/O |
| `numpy` | ≥ 1.24 | Array maths, gradient computation |
| `matplotlib` | ≥ 3.7 | Static figure generation and saving |

Install via:
```bash
pip install -r requirements.txt
```
or use `setup.bat` on Windows to create the virtual environment automatically.

---

## References

- B. D. Lucas and T. Kanade, *"An Iterative Image Registration Technique with an Application to Stereo Vision,"* IJCAI 1981.
- J.-Y. Bouguet, *"Pyramidal Implementation of the Lucas-Kanade Feature Tracker,"* Intel Corporation, 2001.
- J. Shi and C. Tomasi, *"Good Features to Track,"* CVPR 1994.
- G. Farneback, *"Two-Frame Motion Estimation Based on Polynomial Expansion,"* SCIA 2003.
