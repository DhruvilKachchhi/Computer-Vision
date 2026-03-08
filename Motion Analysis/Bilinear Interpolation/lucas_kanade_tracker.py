"""
Lucas-Kanade Optical Flow — Theory, Bilinear Interpolation & Practical Validation
==================================================================================
Derives motion-tracking and bilinear-interpolation equations from first principles,
then validates them against real (or synthetic) video frames.

Usage
-----
    python lucas_kanade_tracker.py                        # synthetic frames
    python lucas_kanade_tracker.py --video clip.mp4       # one video
    python lucas_kanade_tracker.py --video a.mp4 --video b.mp4   # multiple
    python lucas_kanade_tracker.py --video clip.mp4 --frame1 10 --frame2 11
    python lucas_kanade_tracker.py --no-theory            # skip theory print
    python lucas_kanade_tracker.py --outdir results       # custom output folder

Dependencies
------------
    pip install opencv-python numpy matplotlib
    (or use:  setup.bat  on Windows)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path
from typing import Optional

import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np

# Suppress cosmetic matplotlib warnings that do not affect output quality.
# 1. Missing glyphs in the theory PNG (special Unicode in monospace font)
warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from font",
    category=UserWarning,
)
# 2. tight_layout margin warning (figures still render correctly)
warnings.filterwarnings(
    "ignore",
    message="Tight layout not applied",
    category=UserWarning,
)


# ─────────────────────────────────────────────────────────────────────────────
# PART 0 — THEORY TEXT
# ─────────────────────────────────────────────────────────────────────────────

THEORY = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        LUCAS-KANADE OPTICAL FLOW — DERIVATION FROM FIRST PRINCIPLES         ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — Brightness Constancy Assumption
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A scene point with intensity I keeps the same brightness as it moves:

    I(x, y, t) = I(x + dx,  y + dy,  t + dt)                     … (1)

where (dx, dy) is the 2-D displacement during dt.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — First-Order Taylor Expansion of the RHS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expand I(x+dx, y+dy, t+dt) about (x, y, t), keeping first-order terms:

    I(x+dx, y+dy, t+dt) ≈ I(x,y,t)
                          + (∂I/∂x)·dx + (∂I/∂y)·dy + (∂I/∂t)·dt   … (2)

Substitute (2) into (1) and cancel I(x,y,t):

    (∂I/∂x)·dx  +  (∂I/∂y)·dy  +  (∂I/∂t)·dt  =  0               … (3)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — Optical Flow Constraint Equation (OFCE)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Divide (3) by dt and define velocity u = dx/dt,  v = dy/dt:

    Iₓ·u  +  Iᵧ·v  +  I_t  =  0                                    … (OFCE)

Notation:
    Iₓ  = ∂I/∂x   (spatial gradient, horizontal)
    Iᵧ  = ∂I/∂y   (spatial gradient, vertical)
    I_t = ∂I/∂t   (temporal gradient ≈ frame₂ − frame₁)

⚠  The OFCE is ONE equation in TWO unknowns (u, v) — the "aperture problem".
   The constraint lies on a line in (u,v)-space: u·Iₓ + v·Iᵧ = −I_t

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — Lucas-Kanade: Neighbourhood Constant-Flow Assumption
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Key insight (Lucas & Kanade, 1981):
    Assume (u, v) is CONSTANT over a small window W of N pixels.

Each pixel i ∈ W contributes one equation:
    Iₓᵢ·u  +  Iᵧᵢ·v  =  −I_tᵢ       for i = 1, 2, …, N

In matrix form   A · p = b :

    A = ⎡ Iₓ₁  Iᵧ₁ ⎤     b = ⎡ −I_t₁ ⎤     p = ⎡ u ⎤
        ⎢ Iₓ₂  Iᵧ₂ ⎥         ⎢ −I_t₂ ⎥         ⎣ v ⎦
        ⎢  ⋮    ⋮  ⎥         ⎢   ⋮   ⎥
        ⎣ Iₓₙ  Iᵧₙ ⎦         ⎣ −I_tₙ ⎦

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5 — Weighted Least-Squares → 2×2 Normal Equations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimise the weighted residual  Σ wᵢ (Iₓᵢu + Iᵧᵢv + I_tᵢ)²
via normal equations  AᵀWA · p = AᵀW b  (W = diag of Gaussian weights):

    ⎡ ΣwIₓ²    ΣwIₓIᵧ ⎤ ⎡ u ⎤   ⎡ −ΣwIₓI_t ⎤
    ⎣ ΣwIₓIᵧ  ΣwIᵧ²  ⎦ ⎣ v ⎦ = ⎣ −ΣwIᵧI_t ⎦

Solve explicitly (2×2 matrix inversion):

    M = AᵀWA  (the Structure Tensor)

    ⎡ u ⎤   M⁻¹ ⎡ −ΣwIₓI_t ⎤
    ⎣ v ⎦ =      ⎣ −ΣwIᵧI_t ⎦

Trackability from eigenvalues λ₁ ≥ λ₂ of M:
    • λ₁ ≫ 1,  λ₂ ≫ 1  →  corner  →  well-conditioned  ✓
    • λ₁ ≫ λ₂ ≈ 0      →  edge    →  aperture problem
    • λ₁ ≈ λ₂ ≈ 0      →  flat    →  untrackable

Shi-Tomasi criterion: select points where  min(λ₁,λ₂) > threshold.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Pyramid (Coarse-to-Fine) for Large Displacements
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Large motions violate the small-displacement Taylor assumption.
Solution: build an image Gaussian pyramid (downsample by ×2 each level).

    Level L (coarsest): solve for (u_L, v_L)
    Level L-1         : warp frame by 2·(u_L, v_L), refine residual
    …
    Level 0           : accumulate full-resolution (u, v)

cv2.calcOpticalFlowPyrLK implements this (Bouguet, 2001).


╔══════════════════════════════════════════════════════════════════════════════╗
║              BILINEAR INTERPOLATION — DERIVATION FROM SCRATCH                ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY do we need interpolation?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LK tracking returns sub-pixel coordinates (e.g. x=34.7, y=21.3).
Pixel intensities are defined only at integer grid points.
→ We must INTERPOLATE between neighbours to evaluate I at fractional coords.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP A — 1-D Linear Interpolation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Given values f₀ at x=0 and f₁ at x=1, query at x=a  (0 ≤ a ≤ 1):

    f(a) = (1−a)·f₀ + a·f₁                                        … (L1D)

Weight = proportional to OPPOSITE distance. Weights sum to 1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP B — 2-D Bilinear Interpolation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Query point: (x, y) with x₀ ≤ x ≤ x₀+1,  y₀ ≤ y ≤ y₀+1.
Fractional offsets:  a = x − x₀,   b = y − y₀.

Four integer-grid neighbours (unit cell):

    Q₁₁ = I(x₀,   y₀  )   ┌───────────────────┐  y₀
    Q₂₁ = I(x₀+1, y₀  )   │  Q₁₁        Q₂₁   │
    Q₁₂ = I(x₀,   y₀+1)   │        P           │  ← query (x,y)
    Q₂₂ = I(x₀+1, y₀+1)   │  Q₁₂        Q₂₂   │
                           └───────────────────┘  y₀+1
                           x₀                x₀+1

Interpolate along the TOP row (y = y₀):
    R₁ = (1−a)·Q₁₁ + a·Q₂₁

Interpolate along the BOTTOM row (y = y₀+1):
    R₂ = (1−a)·Q₁₂ + a·Q₂₂

Interpolate vertically between R₁ and R₂:
    f(x,y) = (1−b)·R₁ + b·R₂

Expanding fully:

    ┌────────────────────────────────────────────────────────────┐
    │  f(x,y) = (1−a)(1−b)·Q₁₁  +  a(1−b)·Q₂₁                 │
    │         +  (1−a)b  ·Q₁₂   +  a·b    ·Q₂₂                 │
    └────────────────────────────────────────────────────────────┘

Area interpretation:
    Each weight = area of the OPPOSITE sub-rectangle in the unit cell:
        w(Q₁₁) = (1−a)(1−b),   w(Q₂₁) = a(1−b)
        w(Q₁₂) = (1−a)·b,      w(Q₂₂) = a·b
    All four weights sum to 1  →  convex combination, no extrapolation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALIDATION CONNECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After LK predicts sub-pixel location (px, py) in frame 2:
  1. Compute  a = px − floor(px),   b = py − floor(py)
  2. Read four integer neighbours from frame 2
  3. Apply the bilinear formula  →  I_bilinear
  4. Compare with nearest-neighbour  I_round = I(round(px), round(py))
  5. MAE = mean|I_bilinear − I_round| over all tracked points

  Low MAE validates:
    (a) Bilinear interpolation is consistent with the discrete image grid.
    (b) The predicted LK position is accurate — the point genuinely moved there.

  Additionally we check the residual intensity error at the tracked location:
    Δ = |I₁(x₀,y₀) − I₂_bilinear(px,py)|
  Brightness constancy predicts Δ ≈ 0.  Real scenes show small Δ due to
  lighting changes, occlusion, and noise.
"""


def print_theory() -> None:
    """Print the full theory derivation to stdout."""
    print(THEORY)


# ─────────────────────────────────────────────────────────────────────────────
# BILINEAR INTERPOLATION — hand-coded (matches theory above exactly)
# ─────────────────────────────────────────────────────────────────────────────

def bilinear_interp(img: np.ndarray, x: float, y: float) -> float:
    """
    Evaluate a grayscale image at sub-pixel position (x, y) using bilinear
    interpolation.

    Formula (from derivation above):
        f = (1-a)(1-b)*Q11 + a(1-b)*Q21 + (1-a)*b*Q12 + a*b*Q22
    where  a = x - floor(x),   b = y - floor(y).

    Parameters
    ----------
    img : np.ndarray  (H×W, uint8 or float)
        Grayscale image.
    x   : float   Column coordinate (horizontal, increasing rightward).
    y   : float   Row coordinate (vertical, increasing downward).

    Returns
    -------
    float   Interpolated intensity.
    """
    H, W = img.shape[:2]
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    # Clamp to valid pixel indices
    x0c = max(0, min(x0, W - 1))
    x1c = max(0, min(x1, W - 1))
    y0c = max(0, min(y0, H - 1))
    y1c = max(0, min(y1, H - 1))

    # Fractional offsets; clamp x,y so a,b ∈ [0,1] at boundaries
    a = max(0.0, min(float(x) - x0, 1.0))
    b = max(0.0, min(float(y) - y0, 1.0))

    Q11 = float(img[y0c, x0c])
    Q21 = float(img[y0c, x1c])
    Q12 = float(img[y1c, x0c])
    Q22 = float(img[y1c, x1c])

    return (1 - a) * (1 - b) * Q11 \
         + a       * (1 - b) * Q21 \
         + (1 - a) * b       * Q12 \
         + a       * b       * Q22


# ─────────────────────────────────────────────────────────────────────────────
# MANUAL LK IMPLEMENTATION (educational; validates cv2 results)
# ─────────────────────────────────────────────────────────────────────────────

def lk_manual(img1: np.ndarray, img2: np.ndarray,
               pt: tuple[float, float],
               win: int = 11) -> tuple[float, float, float]:
    """
    Compute Lucas-Kanade optical flow at a single point using the 2×2 normal
    equations derived in STEP 5 above.

    Parameters
    ----------
    img1, img2 : grayscale float32 images
    pt         : (x, y) point in img1
    win        : half-window size (full window = 2*win+1)

    Returns
    -------
    (u, v, condition_number)
    """
    x0, y0 = int(round(pt[0])), int(round(pt[1]))
    H, W = img1.shape

    # Spatial gradients via Sobel
    Ix = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    It = img2.astype(np.float64) - img1.astype(np.float64)

    r0 = max(0, y0 - win);  r1 = min(H, y0 + win + 1)
    c0 = max(0, x0 - win);  c1 = min(W, x0 + win + 1)

    ix = Ix[r0:r1, c0:c1].ravel()
    iy = Iy[r0:r1, c0:c1].ravel()
    it = It[r0:r1, c0:c1].ravel()

    # Gaussian weights centred at window
    h, w = r1 - r0, c1 - c0
    gy = cv2.getGaussianKernel(h, -1)
    gx = cv2.getGaussianKernel(w, -1)
    G  = (gy @ gx.T).ravel()

    # Structure tensor  M = AᵀWA
    Sxx = float(np.sum(G * ix * ix))
    Sxy = float(np.sum(G * ix * iy))
    Syy = float(np.sum(G * iy * iy))
    Sxt = float(np.sum(G * ix * it))
    Syt = float(np.sum(G * iy * it))

    M = np.array([[Sxx, Sxy],
                  [Sxy, Syy]])

    det = Sxx * Syy - Sxy ** 2
    if abs(det) < 1e-10:
        return 0.0, 0.0, float("inf")

    u =  (Syy * (-Sxt) - Sxy * (-Syt)) / det
    v = (-Sxy * (-Sxt) + Sxx * (-Syt)) / det

    eigvals = np.linalg.eigvalsh(M)
    cond = float(eigvals[-1]) / (float(eigvals[0]) + 1e-12)
    return float(u), float(v), cond


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO LOADING
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-SELECT BEST FRAME PAIR (maximum inter-frame motion)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_frame_pair(video_path: str,
                         sample_every: int = 1) -> tuple[int, int, float]:
    """
    Scan *video_path* and return the consecutive frame pair with the highest
    mean Farneback optical-flow magnitude — i.e. where the most visible change
    is happening.

    Parameters
    ----------
    video_path   : str   Path to the video file.
    sample_every : int   Check every Nth frame pair (1 = every pair).
                         Increase to speed up scanning on long videos.

    Returns
    -------
    (idx1, idx2, best_mean_magnitude)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 1, 0.0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Scanning {total} frames to find the pair with maximum motion …",
          flush=True)

    best_idx1, best_idx2, best_mag = 0, 1, -1.0

    ret, prev_bgr = cap.read()
    if not ret:
        cap.release()
        return 0, 1, 0.0

    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    while True:
        ret, curr_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_every != 0:
            prev_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
            continue

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=13,
            iterations=2, poly_n=5, poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mean_mag = float(mag.mean())

        if mean_mag > best_mag:
            best_mag  = mean_mag
            best_idx1 = frame_idx - 1
            best_idx2 = frame_idx

        prev_gray = curr_gray

    cap.release()
    print(f"  Best frame pair: {best_idx1} ↔ {best_idx2}  "
          f"(mean flow = {best_mag:.3f} px/frame)")
    return best_idx1, best_idx2, best_mag


def make_synthetic_frames(n: int = 10, h: int = 240, w: int = 320) -> list[np.ndarray]:
    """
    Generate synthetic video frames: Gaussian blobs on dark background with
    known constant translational flow (dx=2 px/frame, dy=1 px/frame).
    """
    rng = np.random.default_rng(42)
    centres = rng.integers(50, min(h, w) - 50, size=(12, 2))   # (y, x)
    frames: list[np.ndarray] = []

    for t in range(n):
        canvas = np.zeros((h, w), dtype=np.float32)
        dx, dy = t * 2, t * 1           # ground-truth displacement
        for cy, cx in centres:
            ny, nx = int(cy + dy), int(cx + dx)
            if 15 < ny < h - 15 and 15 < nx < w - 15:
                canvas[ny, nx] = 255.0
        # Gaussian blur to create smooth blobs (rich gradient texture)
        canvas = cv2.GaussianBlur(canvas, (21, 21), 5)
        # Normalise to [0,220]
        if canvas.max() > 0:
            canvas = canvas / canvas.max() * 220.0
        frames.append(canvas.astype(np.uint8))

    return frames


def load_two_frames(
    video_path: Optional[str],
    frame1_idx: int = 0,
    frame2_idx: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load two frames from a video file (or generate synthetic ones).

    Returns
    -------
    (f1_gray, f2_gray, f1_bgr, f2_bgr, source_name)
    """
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ⚠  Could not open '{video_path}'. Falling back to synthetic.")
            video_path = None
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx2 = min(frame2_idx, total - 1)
            idx1 = min(frame1_idx, idx2 - 1)

            def _grab(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frm = cap.read()
                return frm if ret else None

            frm1 = _grab(cap, idx1)
            frm2 = _grab(cap, idx2)
            cap.release()

            if frm1 is None or frm2 is None:
                print("  ⚠  Could not read requested frames. Falling back to synthetic.")
                video_path = None
            else:
                f1_gray = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)
                f2_gray = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY)
                src = f"{Path(video_path).name}  [frames {idx1}↔{idx2}]"
                return f1_gray, f2_gray, frm1, frm2, src

    # ── Synthetic fallback ───────────────────────────────────────────────────
    print("  ℹ  Generating synthetic video (10 frames, 240×320, dx=2 dy=1 px/frame).")
    synth = make_synthetic_frames()
    f1_gray = synth[0]
    f2_gray = synth[1]
    f1_bgr = cv2.cvtColor(f1_gray, cv2.COLOR_GRAY2BGR)
    f2_bgr = cv2.cvtColor(f2_gray, cv2.COLOR_GRAY2BGR)
    return f1_gray, f2_gray, f1_bgr, f2_bgr, "synthetic  [frames 0↔1, true flow dx=2 dy=1]"


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRACKING + VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def run_tracking(
    video_path: Optional[str],
    outdir: Path,
    frame1_idx: int = 0,
    frame2_idx: int = 1,
    label: str = "",
) -> dict:
    """
    Track features between two consecutive frames, validate with bilinear
    interpolation and manual LK, then save figures.

    Returns a summary dict with keys:
        n_detected, n_tracked, mae_bilinear, mae_brightness, rows
    """
    tag = label or (Path(video_path).stem if video_path else "synthetic")
    bar = "═" * 78

    print(f"\n{bar}")
    print(f"  TRACKING VALIDATION  —  {tag}")
    print(bar)

    # 1. Load frames ─────────────────────────────────────────────────────────
    f1_gray, f2_gray, f1_bgr, f2_bgr, src = load_two_frames(
        video_path, frame1_idx, frame2_idx
    )
    H, W = f1_gray.shape
    print(f"\n  Source      : {src}")
    print(f"  Frame size  : {W} × {H} px")

    # 2. Detect Shi-Tomasi corners ────────────────────────────────────────────
    feat_params = dict(maxCorners=25, qualityLevel=0.04,
                       minDistance=15, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(f1_gray, mask=None, **feat_params)

    if p0 is None or len(p0) == 0:
        print("  ⚠  No features found.  Try a video with more texture.")
        return {}

    # 3. Track with OpenCV LK pyramid ────────────────────────────────────────
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p1, status, _err = cv2.calcOpticalFlowPyrLK(
        f1_gray, f2_gray, p0, None, **lk_params
    )

    mask = status.ravel() == 1
    good_p0 = p0[mask]       # shape (N, 1, 2)
    good_p1 = p1[mask]
    N = len(good_p0)

    print(f"\n  Features detected   : {len(p0)}")
    print(f"  Successfully tracked: {N}")

    if N == 0:
        print("  ⚠  No points tracked.")
        return {}

    # 4. Tabular validation ──────────────────────────────────────────────────
    col_w = 79
    hdr = (f"{'Pt':>3}  {'F1 (x,y)':>14}  {'F2 pred (x,y)':>16}"
           f"  {'dx':>6} {'dy':>6}"
           f"  {'Blin':>7}  {'NN':>6}  {'Δ_interp':>9}"
           f"  {'ΔBright':>8}  {'u_man':>6} {'v_man':>6}")
    sep = "─" * len(hdr)

    print(f"\n{sep}")
    print(hdr)
    print(sep)

    errors_interp: list[float] = []
    errors_bright: list[float] = []
    rows = []

    f1_f32 = f1_gray.astype(np.float32)
    f2_f32 = f2_gray.astype(np.float32)

    for i, (pt0, pt1) in enumerate(zip(good_p0, good_p1)):
        x0, y0 = float(pt0[0][0]), float(pt0[0][1])
        px, py = float(pt1[0][0]), float(pt1[0][1])
        dx = px - x0
        dy = py - y0

        # --- Bilinear interpolated intensity at predicted (px,py) in frame 2
        i_blin = bilinear_interp(f2_gray, px, py)

        # --- Nearest-neighbour intensity (integer coords)
        xi = max(0, min(int(round(px)), W - 1))
        yi = max(0, min(int(round(py)), H - 1))
        i_nn = float(f2_gray[yi, xi])

        # Δ_interp: difference between bilinear and NN
        d_interp = abs(i_blin - i_nn)
        errors_interp.append(d_interp)

        # --- Brightness constancy residual: I₁(x₀,y₀) vs I₂_blin(px,py)
        i1_src = bilinear_interp(f1_gray, x0, y0)
        d_bright = abs(i1_src - i_blin)
        errors_bright.append(d_bright)

        # --- Manual LK flow at this point (educational cross-check)
        u_m, v_m, _ = lk_manual(f1_f32, f2_f32, (x0, y0), win=10)

        row_str = (
            f"{i+1:>3}  ({x0:6.1f},{y0:6.1f})  ({px:7.2f},{py:7.2f})"
            f"  {dx:+6.2f} {dy:+6.2f}"
            f"  {i_blin:7.2f}  {i_nn:6.1f}  {d_interp:9.4f}"
            f"  {d_bright:8.2f}  {u_m:+6.2f} {v_m:+6.2f}"
        )
        print(row_str)
        rows.append(dict(
            idx=i+1, x0=x0, y0=y0, px=px, py=py,
            dx=dx, dy=dy,
            i_blin=i_blin, i_nn=i_nn, d_interp=d_interp,
            i1_src=i1_src, d_bright=d_bright,
            u_man=u_m, v_man=v_m,
        ))

    print(sep)
    mae_blin   = float(np.mean(errors_interp))
    mae_bright = float(np.mean(errors_bright))
    print(f"\n  MAE (bilinear vs nearest-neighbour) : {mae_blin:.4f} intensity units")
    print(f"  MAE (brightness constancy residual) : {mae_bright:.4f} intensity units")
    print(
        "\n  Interpretation:\n"
        "  • Small Δ_interp  → bilinear ≈ nearest-pixel  (interpolation is accurate)\n"
        "  • Small ΔBright   → I₁(src) ≈ I₂(dest)       (brightness constancy holds)\n"
        "  • u_man ≈ dx      → manual LK matches OpenCV  (derivation is correct)\n"
    )

    # 5. Figures ─────────────────────────────────────────────────────────────
    _fig_tracking(f1_bgr, f2_bgr, good_p0, good_p1,
                  mae_blin, mae_bright, src, outdir, tag)
    _fig_dense_hsv(f1_gray, f2_gray, outdir, tag)
    _fig_quiver(f1_bgr, f2_bgr, f1_gray, f2_gray, good_p0, good_p1, outdir, tag)
    _fig_eigenvalue_scatter(f1_gray, good_p0, outdir, tag)

    return dict(n_detected=len(p0), n_tracked=N,
                mae_bilinear=mae_blin, mae_brightness=mae_bright,
                rows=rows)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG   = "#0d1117"
_PANEL_BG  = "#161b22"
_GRID_COL  = "#30363d"
_TEXT_COL  = "#c9d1d9"
_ACCENT    = "#58a6ff"
_ORANGE    = "#f0883e"


def _apply_dark(fig: plt.Figure, axes) -> None:
    fig.patch.set_facecolor(_DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(_TEXT_COL)
        ax.yaxis.label.set_color(_TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID_COL)
        ax.grid(color=_GRID_COL, linestyle="--", linewidth=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# NEW VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _fig_dense_hsv(f1_gray: np.ndarray, f2_gray: np.ndarray,
                   outdir: Path, tag: str) -> None:
    """
    Dense Optical Flow — HSV colour map.

    Hue  → direction of motion  (0°–360°)
    Saturation → 1 (always full colour)
    Value (brightness) → magnitude, normalised to [0, 255]

    Uses cv2.calcOpticalFlowFarneback for dense per-pixel flow.
    """
    flow = cv2.calcOpticalFlowFarneback(
        f1_gray, f2_gray, None,
        pyr_scale=0.5, levels=5, winsize=15,
        iterations=3, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    fx, fy = flow[..., 0], flow[..., 1]
    magnitude, angle = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    # Build HSV image: H = direction, S = 1, V = normalised magnitude
    hsv = np.zeros((*f1_gray.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2).astype(np.uint8)          # H: 0-179 (OpenCV scale)
    hsv[..., 1] = 255                                    # S: full saturation
    norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = norm_mag.astype(np.uint8)              # V: speed

    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor(_DARK_BG)
    gs = GridSpec(1, 3, figure=fig, wspace=0.04)

    ax_f1  = fig.add_subplot(gs[0, 0])
    ax_f2  = fig.add_subplot(gs[0, 1])
    ax_hsv = fig.add_subplot(gs[0, 2])

    for ax, img, title, cmap in [
        (ax_f1,  f1_gray, "Frame 1",        "gray"),
        (ax_f2,  f2_gray, "Frame 2",        "gray"),
        (ax_hsv, flow_rgb,"Dense Flow  (HSV)", None),
    ]:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=_TEXT_COL, fontsize=12, pad=6)
        ax.axis("off")
        ax.set_facecolor(_PANEL_BG)

    # ── Colour-wheel legend ───────────────────────────────────────────────────
    legend_ax = fig.add_axes([0.968, 0.18, 0.03, 0.64])   # thin right strip
    theta = np.linspace(0, 2 * np.pi, 256)
    wheel_data = np.tile(np.linspace(0, 179, 256, dtype=np.uint8), (20, 1))
    wheel_hsv  = np.dstack([wheel_data,
                            np.full_like(wheel_data, 255),
                            np.full_like(wheel_data, 200)])
    wheel_rgb  = cv2.cvtColor(wheel_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    legend_ax.imshow(wheel_rgb.transpose(1, 0, 2), aspect="auto",
                     extent=[0, 1, 0, 360])
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 360)
    legend_ax.yaxis.set_label_position("right")
    legend_ax.yaxis.tick_right()
    legend_ax.set_yticks([0, 90, 180, 270, 360])
    legend_ax.set_yticklabels(["0°\n(→)", "90°\n(↓)", "180°\n(←)", "270°\n(↑)", "360°"],
                               fontsize=7, color=_TEXT_COL)
    legend_ax.set_xticks([])
    for sp in legend_ax.spines.values():
        sp.set_edgecolor(_GRID_COL)
    legend_ax.set_facecolor(_PANEL_BG)
    legend_ax.set_title("dir", color=_TEXT_COL, fontsize=7, pad=3)
    legend_ax.patch.set_facecolor(_DARK_BG)

    # ── Magnitude colourbar ──────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap="gray",
        norm=plt.Normalize(vmin=0, vmax=float(magnitude.max())),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_hsv, fraction=0.03, pad=0.015, aspect=30)
    cbar.set_label("Flow magnitude  [px / frame]",
                   color=_TEXT_COL, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COL, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COL)
    cbar.outline.set_edgecolor(_GRID_COL)
    cbar.ax.set_facecolor(_PANEL_BG)

    fig.suptitle(
        f"Dense Optical Flow — HSV Colour Map  |  {tag}\n"
        "Hue = direction  •  Brightness = speed",
        color=_TEXT_COL, fontsize=11, y=1.01,
    )
    _save(fig, outdir / f"lk_dense_hsv_{_safe(tag)}.png")


def _fig_quiver(f1_bgr: np.ndarray, f2_bgr: np.ndarray,
                f1_gray: np.ndarray, f2_gray: np.ndarray,
                good_p0: np.ndarray, good_p1: np.ndarray,
                outdir: Path, tag: str) -> None:
    """
    Quiver / Arrow Field overlaid on both frames.

    • Frame 1 panel: feature points coloured by motion magnitude.
    • Frame 2 panel: matplotlib quiver arrows (coloured by magnitude) drawn
      from each tracked point, overlaid on the frame.
    • A shared colourbar shows the motion scale in pixels/frame.
    """
    fig = plt.figure(figsize=(17, 7))
    fig.patch.set_facecolor(_DARK_BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.06,
                  left=0.03, right=0.91)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Convert BGR → RGB for matplotlib
    f1_rgb = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2RGB)
    f2_rgb = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2RGB)

    pts0 = good_p0.reshape(-1, 2)   # (N, 2): [x, y]
    pts1 = good_p1.reshape(-1, 2)
    dx   = pts1[:, 0] - pts0[:, 0]
    dy   = pts1[:, 1] - pts0[:, 1]
    mag  = np.sqrt(dx**2 + dy**2)

    norm     = plt.Normalize(vmin=0, vmax=max(mag.max(), 1e-6))
    cmap_q   = plt.cm.plasma

    # ── Frame 1: scatter coloured by magnitude ───────────────────────────────
    ax1.imshow(f1_rgb)
    sc = ax1.scatter(pts0[:, 0], pts0[:, 1],
                     c=mag, cmap=cmap_q, norm=norm,
                     s=60, zorder=5, edgecolors="white", linewidths=0.6)
    for i, (x, y) in enumerate(pts0):
        ax1.text(x + 5, y - 5, str(i + 1),
                 color="white", fontsize=6, zorder=6)
    ax1.set_title("Frame 1 — Tracked Points\n(colour = motion magnitude)",
                  color=_TEXT_COL, fontsize=11, pad=6)
    ax1.axis("off")
    ax1.set_facecolor(_PANEL_BG)

    # ── Frame 2: quiver arrows ────────────────────────────────────────────────
    ax2.imshow(f2_rgb)
    # Draw individual arrows via annotate for better control
    colours = cmap_q(norm(mag))
    for i, (x0, y0, ddx, ddy, col) in enumerate(
            zip(pts0[:, 0], pts0[:, 1], dx, dy, colours)):
        ax2.annotate(
            "", xy=(x0 + ddx, y0 + ddy), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=1.8,
                mutation_scale=14,
            ),
            zorder=5,
        )
        ax2.plot(x0 + ddx, y0 + ddy, "o", ms=5,
                 color=col, zorder=6,
                 markeredgecolor="white", markeredgewidth=0.5)
        ax2.text(x0 + ddx + 5, y0 + ddy - 5, str(i + 1),
                 color="white", fontsize=6, zorder=7)

    ax2.set_title("Frame 2 — Motion Vectors\n(colour = magnitude, arrow = direction)",
                  color=_TEXT_COL, fontsize=11, pad=6)
    ax2.axis("off")
    ax2.set_facecolor(_PANEL_BG)

    # ── Shared colourbar ─────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    sm = plt.cm.ScalarMappable(cmap=cmap_q, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Motion magnitude  [px / frame]",
                   color=_TEXT_COL, fontsize=9, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color=_TEXT_COL, labelsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=_TEXT_COL)
    cbar.outline.set_edgecolor(_GRID_COL)
    cbar.ax.set_facecolor(_PANEL_BG)

    fig.suptitle(
        f"Lucas-Kanade Quiver Field  |  {tag}",
        color=_TEXT_COL, fontsize=12, y=1.01,
    )
    _save(fig, outdir / f"lk_quiver_{_safe(tag)}.png")


def _fig_eigenvalue_scatter(f1_gray: np.ndarray,
                            good_p0: np.ndarray,
                            outdir: Path, tag: str,
                            block_size: int = 7) -> None:
    """
    Eigenvalue Scatter Plot — λ₁ vs λ₂ of the Structure Tensor.

    For every detected point we compute the 2×2 structure tensor M from the
    Sobel gradients inside a block_size window and extract both eigenvalues.
    Points are categorised as:
        CORNER : λ₂  > corner_thresh
        EDGE   : λ₁  > corner_thresh  and  λ₂ ≤ corner_thresh
        FLAT   : λ₁  ≤ corner_thresh
    """
    f32 = f1_gray.astype(np.float32)
    Ix  = cv2.Sobel(f32, cv2.CV_64F, 1, 0, ksize=3)
    Iy  = cv2.Sobel(f32, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Gaussian-weighted sums over the block
    ksize  = block_size if block_size % 2 == 1 else block_size + 1
    sigma  = ksize / 3.0
    Sxx = cv2.GaussianBlur(Ixx, (ksize, ksize), sigma)
    Sxy = cv2.GaussianBlur(Ixy, (ksize, ksize), sigma)
    Syy = cv2.GaussianBlur(Iyy, (ksize, ksize), sigma)

    H, W = f1_gray.shape
    pts  = good_p0.reshape(-1, 2)   # (N, 2)
    lam1_list, lam2_list, labels = [], [], []

    corner_thresh = 100.0   # tuned to typical uint8 Sobel magnitudes

    for x, y in pts:
        xi = int(round(float(x))); yi = int(round(float(y)))
        xi = max(0, min(xi, W - 1)); yi = max(0, min(yi, H - 1))

        M = np.array([[Sxx[yi, xi], Sxy[yi, xi]],
                      [Sxy[yi, xi], Syy[yi, xi]]])
        eigs = np.linalg.eigvalsh(M)      # sorted ascending → [λ₂, λ₁]
        lam2_val = float(eigs[0])
        lam1_val = float(eigs[1])
        lam1_list.append(lam1_val)
        lam2_list.append(lam2_val)

        if lam2_val > corner_thresh:
            labels.append("corner")
        elif lam1_val > corner_thresh:
            labels.append("edge")
        else:
            labels.append("flat")

    lam1 = np.array(lam1_list)
    lam2 = np.array(lam2_list)
    labels = np.array(labels)

    # Colours / markers per category
    CAT_STYLE = {
        "corner": dict(color="#3fb950", marker="*",  s=180, label="Corner  (both λ large)"),
        "edge":   dict(color=_ORANGE,   marker="D",  s=80,  label="Edge    (λ₁≫λ₂≈0)"),
        "flat":   dict(color="#6e7681", marker="o",  s=60,  label="Flat    (both λ small)"),
    }

    fig, (ax_scatter, ax_bar) = plt.subplots(
        1, 2, figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    _apply_dark(fig, [ax_scatter, ax_bar])

    # ── Scatter λ₁ vs λ₂ ────────────────────────────────────────────────────
    for cat, style in CAT_STYLE.items():
        mask = labels == cat
        if mask.sum() == 0:
            continue
        ax_scatter.scatter(lam1[mask], lam2[mask],
                           color=style["color"],
                           marker=style["marker"],
                           s=style["s"],
                           label=style["label"],
                           alpha=0.88, edgecolors="white", linewidths=0.5,
                           zorder=5)

    # ── Reference line λ₁ = λ₂ (equal eigenvalues → isotropic texture) ─────
    vmax = max(lam1.max(), lam2.max()) * 1.05 if len(lam1) > 0 else 1
    ax_scatter.plot([0, vmax], [0, vmax], "--",
                    color=_GRID_COL, linewidth=1.2, label="λ₁ = λ₂", zorder=3)

    # ── Threshold lines ──────────────────────────────────────────────────────
    ax_scatter.axhline(corner_thresh, color="#8b949e", linewidth=0.8,
                       linestyle=":", label=f"threshold = {corner_thresh:.0f}")
    ax_scatter.axvline(corner_thresh, color="#8b949e", linewidth=0.8,
                       linestyle=":")

    # ── Region annotations ────────────────────────────────────────────────────
    _reg_text_kw = dict(
        transform=ax_scatter.transAxes,
        fontsize=8, alpha=0.75,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_PANEL_BG,
                  edgecolor=_GRID_COL, alpha=0.8),
    )
    ax_scatter.text(0.72, 0.93, "CORNER\n(trackable ✓)",
                    color="#3fb950", **_reg_text_kw)
    ax_scatter.text(0.72, 0.35, "EDGE\n(aperture ⚠)",
                    color=_ORANGE, **_reg_text_kw)
    ax_scatter.text(0.10, 0.12, "FLAT\n(untrackable ✗)",
                    color="#6e7681", **_reg_text_kw)

    ax_scatter.set_xlabel("λ₁  (larger eigenvalue)", color=_TEXT_COL, fontsize=10)
    ax_scatter.set_ylabel("λ₂  (smaller eigenvalue)", color=_TEXT_COL, fontsize=10)
    ax_scatter.set_title("Structure Tensor Eigenvalues  (λ₁ vs λ₂)",
                         color=_TEXT_COL, fontsize=11)
    ax_scatter.set_xlim(left=0)
    ax_scatter.set_ylim(bottom=0)
    ax_scatter.legend(fontsize=8, facecolor=_PANEL_BG,
                      labelcolor=_TEXT_COL, edgecolor=_GRID_COL, loc="upper left")

    # Point labels
    for i, (l1, l2, lbl) in enumerate(zip(lam1, lam2, labels)):
        ax_scatter.text(l1 * 1.01, l2 * 1.01, str(i + 1),
                        fontsize=6.5, color=_TEXT_COL, zorder=6)

    # ── Bar chart: min(λ₁,λ₂) = Shi-Tomasi score ────────────────────────────
    shi_tomasi = np.minimum(lam1, lam2)
    cat_colors  = [CAT_STYLE[c]["color"] for c in labels]
    x_idx = np.arange(len(shi_tomasi))
    bars = ax_bar.bar(x_idx, shi_tomasi,
                      color=cat_colors, edgecolor=_GRID_COL,
                      linewidth=0.6, alpha=0.88, width=0.7)
    ax_bar.axhline(corner_thresh, color="white", linewidth=1.2,
                   linestyle="--", label=f"threshold = {corner_thresh:.0f}")
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels([str(i + 1) for i in range(len(shi_tomasi))],
                            fontsize=7, color=_TEXT_COL)
    ax_bar.set_xlabel("Point index", color=_TEXT_COL, fontsize=9)
    ax_bar.set_ylabel("min(λ₁, λ₂)  — Shi-Tomasi score",
                      color=_TEXT_COL, fontsize=8)
    ax_bar.set_title("Shi-Tomasi Score\nper Tracked Point",
                     color=_TEXT_COL, fontsize=10)
    ax_bar.legend(fontsize=7, facecolor=_PANEL_BG,
                  labelcolor=_TEXT_COL, edgecolor=_GRID_COL)
    for bar, val in zip(bars, shi_tomasi):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + shi_tomasi.max() * 0.02,
                    f"{val:.0f}", ha="center", va="bottom",
                    fontsize=5.5, color=_TEXT_COL)

    fig.suptitle(
        f"Eigenvalue Analysis — Structure Tensor  |  {tag}\n"
        "Points with large λ₂ are corners (well-trackable); "
        "λ₁≫λ₂≈0 → edge; λ₁≈λ₂≈0 → flat",
        color=_TEXT_COL, fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _save(fig, outdir / f"lk_eigenvalue_scatter_{_safe(tag)}.png")


def _fig_tracking(f1_bgr, f2_bgr, good_p0, good_p1,
                  mae_blin, mae_bright, src, outdir: Path, tag: str) -> None:
    """Side-by-side frame visualisation with motion arrows."""
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(_DARK_BG)
    gs = GridSpec(1, 2, figure=fig, wspace=0.04)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    def _show(ax, bgr, title):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color=_TEXT_COL, fontsize=12, pad=6)
        ax.axis("off")

    f1_vis, f2_vis = f1_bgr.copy(), f2_bgr.copy()
    cmap_pts = plt.cm.plasma(np.linspace(0.05, 0.95, len(good_p0)))

    for i, (pt0, pt1) in enumerate(zip(good_p0, good_p1)):
        x0, y0 = int(round(pt0[0][0])), int(round(pt0[0][1]))
        x1, y1 = int(round(pt1[0][0])), int(round(pt1[0][1]))
        c_bgr = (int(cmap_pts[i][2]*255), int(cmap_pts[i][1]*255),
                 int(cmap_pts[i][0]*255))

        cv2.circle(f1_vis, (x0, y0), 5, c_bgr, -1)
        cv2.circle(f1_vis, (x0, y0), 7, (255, 255, 255), 1)
        cv2.putText(f1_vis, str(i+1), (x0+8, y0-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.arrowedLine(f2_vis, (x0, y0), (x1, y1), c_bgr, 2, tipLength=0.3)
        cv2.circle(f2_vis, (x1, y1), 5, c_bgr, -1)
        cv2.circle(f2_vis, (x1, y1), 7, (255, 255, 255), 1)
        cv2.putText(f2_vis, str(i+1), (x1+8, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    _show(ax1, f1_vis, "Frame 1 — Detected Interest Points")
    _show(ax2, f2_vis, "Frame 2 — LK Predicted Locations  (arrows = motion)")

    patches = [mpatches.Patch(color=cmap_pts[i][:3], label=f"Pt {i+1}")
               for i in range(len(good_p0))]
    ax2.legend(handles=patches, loc="lower right", fontsize=7, ncol=5,
               facecolor=_PANEL_BG, labelcolor=_TEXT_COL, edgecolor=_GRID_COL)

    fig.suptitle(
        f"Lucas-Kanade Optical Flow  |  {src}\n"
        f"Bilinear MAE={mae_blin:.4f}   Brightness residual MAE={mae_bright:.4f}  (intensity units)",
        color=_TEXT_COL, fontsize=11, y=0.98,
    )
    _save(fig, outdir / f"lk_tracking_{_safe(tag)}.png")


def _fig_bilinear_detail(f2_gray: np.ndarray,
                         good_p1: np.ndarray,
                         rows: list[dict],
                         outdir: Path, tag: str) -> None:
    """
    For the first (up to 6) tracked points: zoom into the 3×3 neighbourhood
    in frame 2 and annotate the bilinear weights and interpolated value.
    """
    n_show = min(6, len(rows))
    cols = 3
    r_count = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(r_count, cols,
                             figsize=(cols * 4, r_count * 3.8))
    fig.patch.set_facecolor(_DARK_BG)
    axes_flat = np.array(axes).ravel()

    for idx in range(len(axes_flat)):
        ax = axes_flat[idx]
        ax.set_facecolor(_PANEL_BG)
        ax.tick_params(colors=_TEXT_COL, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRID_COL)

        if idx >= n_show:
            ax.axis("off")
            continue

        row = rows[idx]
        px, py = row["px"], row["py"]
        x0, y0 = int(np.floor(px)), int(np.floor(py))
        a = px - x0
        b = py - y0

        # 5×5 crop centred on the 2×2 neighbourhood
        H, W = f2_gray.shape
        r0 = max(0, y0 - 1);  r1 = min(H, y0 + 3)
        c0 = max(0, x0 - 1);  c1 = min(W, x0 + 3)
        crop = f2_gray[r0:r1, c0:c1].astype(np.float32)

        ax.imshow(crop, cmap="gray", vmin=0, vmax=255,
                  extent=[c0-0.5, c1-0.5, r1-0.5, r0-0.5],
                  interpolation="nearest")

        # Draw the 2×2 bilinear cell
        rect = plt.Rectangle((x0-0.5, y0-0.5), 1, 1,
                              linewidth=1.5, edgecolor=_ACCENT, facecolor="none")
        ax.add_patch(rect)

        # Mark the four corners with their weights
        corners = [
            (x0,   y0,   (1-a)*(1-b), "Q₁₁"),
            (x0+1, y0,       a*(1-b), "Q₂₁"),
            (x0,   y0+1, (1-a)*b,     "Q₁₂"),
            (x0+1, y0+1,     a*b,     "Q₂₂"),
        ]
        for cx, cy, w, lbl in corners:
            if 0 <= cy < H and 0 <= cx < W:
                v = float(f2_gray[cy, cx])
                ax.plot(cx, cy, "o", ms=7, color=_ORANGE, zorder=5)
                ax.text(cx, cy - 0.42,
                        f"{lbl}\nI={v:.0f}\nw={w:.3f}",
                        ha="center", va="bottom", fontsize=5.5,
                        color=_ORANGE, fontfamily="monospace")

        # Mark sub-pixel query point
        ax.plot(px, py, "*", ms=11, color="#3fb950", zorder=6,
                label=f"P=({px:.2f},{py:.2f})")

        ax.set_title(
            f"Pt {row['idx']}  bilinear={row['i_blin']:.1f}  NN={row['i_nn']:.0f}  "
            f"Δ={row['d_interp']:.3f}",
            color=_TEXT_COL, fontsize=7.5,
        )
        ax.set_xlim(c0 - 0.5, c1 - 0.5)
        ax.set_ylim(r1 - 0.5, r0 - 0.5)
        ax.legend(fontsize=6, facecolor=_PANEL_BG, labelcolor=_TEXT_COL,
                  edgecolor=_GRID_COL, loc="upper right")
        ax.set_xlabel("column (x)", color=_TEXT_COL, fontsize=7)
        ax.set_ylabel("row (y)",    color=_TEXT_COL, fontsize=7)

    fig.suptitle(
        f"Bilinear Interpolation — Neighbourhood Detail  |  {tag}",
        color=_TEXT_COL, fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _save(fig, outdir / f"lk_bilinear_detail_{_safe(tag)}.png")


def _fig_error_bars(rows: list[dict], outdir: Path, tag: str) -> None:
    """
    Bar chart of per-point errors: Δ_interp (bilinear vs NN) and
    ΔBright (brightness constancy residual).
    """
    if not rows:
        return
    idxs = [r["idx"] for r in rows]
    d_interp = [r["d_interp"] for r in rows]
    d_bright = [r["d_bright"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _apply_dark(fig, axes)

    x = np.arange(len(idxs))
    w = 0.6

    for ax, data, color, ylabel, title in [
        (axes[0], d_interp, _ACCENT,
         "| bilinear − nearest-neighbour |  [intensity]",
         "Interpolation Error  Δ_interp"),
        (axes[1], d_bright, _ORANGE,
         "| I₁(src) − I₂_bilinear(dest) |  [intensity]",
         "Brightness Constancy Residual  ΔBright"),
    ]:
        bars = ax.bar(x, data, width=w, color=color, alpha=0.85,
                      edgecolor=_GRID_COL, linewidth=0.6)
        ax.axhline(np.mean(data), color="white", linewidth=1.2,
                   linestyle="--", label=f"MAE = {np.mean(data):.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Pt{i}" for i in idxs], fontsize=7, color=_TEXT_COL)
        ax.set_ylabel(ylabel, color=_TEXT_COL, fontsize=8)
        ax.set_title(title, color=_TEXT_COL, fontsize=10)
        ax.legend(fontsize=8, facecolor=_PANEL_BG, labelcolor=_TEXT_COL,
                  edgecolor=_GRID_COL)
        for bar, val in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6, color=_TEXT_COL)

    fig.suptitle(f"Per-Point Error Analysis  |  {tag}",
                 color=_TEXT_COL, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, outdir / f"lk_errors_{_safe(tag)}.png")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔  Saved → {path}")


def _safe(s: str) -> str:
    """Make a string safe for use as a filename."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


# ─────────────────────────────────────────────────────────────────────────────
# THEORY FIGURE
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# FULL-VIDEO VISUALISATIONS  (output: three .mp4 files)
# ─────────────────────────────────────────────────────────────────────────────

def run_video_visualisations(video_path: str, outdir: Path) -> None:
    """
    Process every frame of *video_path* and write three output videos:

      1. vid_dense_hsv_<name>.mp4   — Farneback HSV colour map blended onto frame
      2. vid_trails_<name>.mp4      — LK comet-tail trajectory trails
      3. vid_quiver_<name>.mp4      — LK quiver arrow field overlay

    Parameters
    ----------
    video_path : str   Path to the source video file.
    outdir     : Path  Directory to save the three output .mp4 files.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠  Cannot open '{video_path}' for video generation.")
        return

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem  = Path(video_path).stem
    tag   = _safe(stem)

    print(f"\n{'═'*78}")
    print(f"  VIDEO VISUALISATIONS  —  {stem}")
    print(f"{'═'*78}")
    print(f"  {W}×{H} px  @  {fps:.1f} fps  ·  {total} frames")

    outdir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path_hsv    = outdir / f"vid_dense_hsv_{tag}.mp4"
    path_trails = outdir / f"vid_trails_{tag}.mp4"
    path_quiver = outdir / f"vid_quiver_{tag}.mp4"

    out_hsv    = cv2.VideoWriter(str(path_hsv),    fourcc, fps, (W, H))
    out_trails = cv2.VideoWriter(str(path_trails), fourcc, fps, (W, H))
    out_quiver = cv2.VideoWriter(str(path_quiver), fourcc, fps, (W, H))

    # ── Feature detection / LK parameters ────────────────────────────────────
    feat_params = dict(maxCorners=120, qualityLevel=0.02,
                       minDistance=10,  blockSize=7)
    lk_params   = dict(
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )

    # ── Per-track colour palette (seeded for reproducibility) ─────────────────
    rng    = np.random.default_rng(0)
    COLORS = rng.integers(80, 255, size=(2000, 3)).tolist()   # BGR

    # ── Trail state ───────────────────────────────────────────────────────────
    trail_canvas: Optional[np.ndarray] = None
    tracked_pts:  Optional[np.ndarray] = None   # (N, 1, 2) float32
    pt_ids:       list[int]            = []
    next_id:      int                  = 0
    trail_hist:   dict[int, list[tuple[int,int]]] = {}

    # ── Read first frame ──────────────────────────────────────────────────────
    ret, prev_bgr = cap.read()
    if not ret:
        cap.release()
        return

    prev_gray    = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    trail_canvas = np.zeros_like(prev_bgr)

    # Initial point detection
    def _detect_pts(gray: np.ndarray) -> tuple[Optional[np.ndarray], list[int], int]:
        nonlocal next_id
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **feat_params)
        if pts is None:
            return None, [], next_id
        ids: list[int] = []
        for _ in pts:
            trail_hist[next_id] = []
            ids.append(next_id)
            next_id += 1
        return pts, ids, next_id

    tracked_pts, pt_ids, next_id = _detect_pts(prev_gray)

    frame_idx   = 0
    redetect_every = max(1, int(fps))   # refresh features every ~1 s

    print(f"\n  Writing frames …  ", end="", flush=True)

    while True:
        ret, curr_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"{frame_idx}/{total}", end="  ", flush=True)

        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)

        # ── 1. Dense Farneback HSV ────────────────────────────────────────────
        flow      = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=13,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
        )
        fx, fy    = flow[..., 0], flow[..., 1]
        mag, ang  = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        hsv_f     = np.zeros((H, W, 3), dtype=np.uint8)
        hsv_f[..., 0] = (ang / 2).astype(np.uint8)           # direction
        hsv_f[..., 1] = 255                                   # saturation
        hsv_f[..., 2] = cv2.normalize(mag, None, 0, 255,
                                      cv2.NORM_MINMAX).astype(np.uint8)  # speed
        bgr_flow  = cv2.cvtColor(hsv_f, cv2.COLOR_HSV2BGR)
        hsv_out   = cv2.addWeighted(curr_bgr, 0.30, bgr_flow, 0.70, 0)
        cv2.putText(hsv_out,
                    f"Dense Optical Flow (HSV)  |  frame {frame_idx}/{total}",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)
        out_hsv.write(hsv_out)

        # ── 2 & 3. LK-based: trails + quiver ──────────────────────────────────
        quiver_frame = curr_bgr.copy()
        trail_canvas = (trail_canvas * 0.90).astype(np.uint8)   # fade tails

        if tracked_pts is not None and len(tracked_pts) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, tracked_pts, None, **lk_params
            )
            good = (status.ravel() == 1)

            valid_new: list[np.ndarray] = []
            valid_ids: list[int]        = []

            for ok, npt, opt, pid in zip(good, new_pts, tracked_pts, pt_ids):
                if not ok:
                    continue
                nx, ny = int(npt[0][0]), int(npt[0][1])
                ox, oy = int(opt[0][0]), int(opt[0][1])
                col    = COLORS[pid % len(COLORS)]

                # ── Trail ─────────────────────────────────────────────────────
                hist = trail_hist.setdefault(pid, [])
                hist.append((nx, ny))
                if len(hist) >= 2:
                    # Draw fading line with thickness proportional to recency
                    n_seg = min(len(hist) - 1, 25)
                    for k in range(n_seg):
                        alpha = (k + 1) / n_seg            # 0 → 1 (older → newer)
                        thick = max(1, int(alpha * 3))
                        fade_col = [int(c * (0.4 + 0.6 * alpha)) for c in col]
                        cv2.line(trail_canvas,
                                 hist[-(n_seg + 1 - k)],
                                 hist[-(n_seg - k)],
                                 fade_col, thick)

                # ── Quiver ────────────────────────────────────────────────────
                speed = float(np.hypot(nx - ox, ny - oy))
                if speed > 0.4:
                    tip = min(0.5, 6.0 / (speed + 1e-6))
                    cv2.arrowedLine(quiver_frame, (ox, oy), (nx, ny),
                                    col, 2, tipLength=tip, line_type=cv2.LINE_AA)
                cv2.circle(quiver_frame, (nx, ny), 3, col, -1)

                valid_new.append(npt)
                valid_ids.append(pid)

            # ── Merge trail onto frame ─────────────────────────────────────────
            trails_out = cv2.add(curr_bgr, trail_canvas)
            for p, pid in zip(valid_new, valid_ids):
                cx, cy = int(p[0][0]), int(p[0][1])
                cv2.circle(trails_out, (cx, cy), 4, COLORS[pid % len(COLORS)], -1)

            # ── Labels ────────────────────────────────────────────────────────
            cv2.putText(trails_out,
                        f"Motion Trajectory Trails  |  frame {frame_idx}/{total}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(quiver_frame,
                        f"Quiver Arrow Field  |  frame {frame_idx}/{total}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            out_trails.write(trails_out)
            out_quiver.write(quiver_frame)

            # Update tracked points
            if len(valid_new) >= 8:
                tracked_pts = np.array(valid_new, dtype=np.float32)
                pt_ids      = valid_ids
            else:
                tracked_pts, pt_ids, next_id = _detect_pts(curr_gray)

        else:
            # No tracked points — write plain frame and try to re-detect
            cv2.putText(curr_bgr,
                        f"Motion Trajectory Trails  |  frame {frame_idx}/{total}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            out_trails.write(curr_bgr)
            out_quiver.write(curr_bgr)
            tracked_pts, pt_ids, next_id = _detect_pts(curr_gray)

        # ── Periodic re-detection (inject fresh points into the set) ──────────
        if frame_idx % redetect_every == 0:
            fresh = cv2.goodFeaturesToTrack(curr_gray, mask=None, **feat_params)
            if fresh is not None and tracked_pts is not None:
                existing = tracked_pts.reshape(-1, 2)
                add_pts, add_ids = [], []
                for fp in fresh.reshape(-1, 2):
                    if len(existing) == 0 or \
                       float(np.min(np.linalg.norm(existing - fp, axis=1))) > 15:
                        add_pts.append(fp.reshape(1, 1, 2).astype(np.float32))
                        trail_hist[next_id] = []
                        add_ids.append(next_id)
                        next_id += 1
                        existing = np.vstack([existing, fp.reshape(1, 2)])
                if add_pts:
                    tracked_pts = np.vstack([tracked_pts] + add_pts)
                    pt_ids      = pt_ids + add_ids

        prev_gray = curr_gray.copy()

    cap.release()
    out_hsv.release()
    out_trails.release()
    out_quiver.release()

    print(f"\n  ✔  Saved → {path_hsv}")
    print(f"  ✔  Saved → {path_trails}")
    print(f"  ✔  Saved → {path_quiver}")


def save_theory_figure(outdir: Path) -> None:
    """Render the full theory derivation text as a dark-themed PNG."""
    lines = THEORY.strip().split("\n")
    n = len(lines)
    fig, ax = plt.subplots(figsize=(14, n * 0.215 + 0.5))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.axis("off")

    for i, line in enumerate(lines):
        y_pos = 1.0 - (i + 0.5) / n
        if any(kw in line for kw in ("STEP", "WHY", "VALIDATION", "STEP A", "STEP B")):
            colour = _ACCENT
        elif line.startswith("╔") or line.startswith("╚") or line.startswith("║"):
            colour = _ORANGE
        elif "━" in line:
            colour = _GRID_COL
        elif line.strip().startswith(("│", "┌", "└")):
            colour = "#8b949e"
        elif "⚠" in line or "✓" in line or "→" in line:
            colour = "#3fb950"
        else:
            colour = _TEXT_COL

        ax.text(0.01, y_pos, line, transform=ax.transAxes,
                fontsize=7.2, fontfamily="monospace",
                color=colour, va="center", ha="left")

    _save(fig, outdir / "lk_theory.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lucas-Kanade optical flow — theory, bilinear interpolation "
                    "and practical frame-to-frame validation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python lucas_kanade_tracker.py
              python lucas_kanade_tracker.py --video clip.mp4
              python lucas_kanade_tracker.py --video a.mp4 --video b.mp4
              python lucas_kanade_tracker.py --video clip.mp4 --frame1 5 --frame2 6
              python lucas_kanade_tracker.py --outdir results --no-theory
        """),
    )
    p.add_argument(
        "--video", action="append", dest="videos", metavar="PATH",
        help="Path to a video file. Repeat for multiple videos. "
             "Omit to use synthetic frames.",
    )
    p.add_argument(
        "--frame1", type=int, default=0, metavar="N",
        help="Index of the first frame to use (default: 0).",
    )
    p.add_argument(
        "--frame2", type=int, default=1, metavar="N",
        help="Index of the second frame to use (default: 1).",
    )
    p.add_argument(
        "--outdir", type=str, default="output",
        help="Directory to save output images (default: output/).",
    )
    p.add_argument(
        "--no-theory", action="store_true",
        help="Skip printing the theory derivation to stdout.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Theory ───────────────────────────────────────────────────────────────
    if not args.no_theory:
        print_theory()

    print(f"\n  Saving theory figure …")
    save_theory_figure(outdir)

    # ── Tracking + validation ─────────────────────────────────────────────────
    videos: list[Optional[str]] = args.videos if args.videos else [None]

    # Sentinel: did the user explicitly override frame indices?
    user_set_frames = not (args.frame1 == 0 and args.frame2 == 1)

    summaries = []
    for vid in videos:
        lbl = Path(vid).stem if vid else "synthetic"

        f1_idx = args.frame1
        f2_idx = args.frame2

        # ── Auto-select best frame pair when video supplied + no manual override
        if vid is not None and not user_set_frames:
            f1_idx, f2_idx, _ = find_best_frame_pair(vid)

        result = run_tracking(
            video_path=vid,
            outdir=outdir,
            frame1_idx=f1_idx,
            frame2_idx=f2_idx,
            label=lbl,
        )
        if result:
            summaries.append((lbl, result))

        # ── Full-video visualisations (three .mp4 outputs) ───────────────────
        if vid is not None:
            run_video_visualisations(vid, outdir)

    # ── Overall summary ───────────────────────────────────────────────────────
    if len(summaries) > 1:
        print("\n" + "═" * 78)
        print("  MULTI-VIDEO SUMMARY")
        print("═" * 78)
        print(f"  {'Source':<30}  {'Tracked':>8}  {'MAE blin':>10}  {'MAE bright':>11}")
        print("  " + "─" * 64)
        for lbl, r in summaries:
            print(f"  {lbl:<30}  {r['n_tracked']:>8}  "
                  f"{r['mae_bilinear']:>10.4f}  {r['mae_brightness']:>11.4f}")

    print(f"\n  All outputs saved to: {outdir.resolve()}\n")


if __name__ == "__main__":
    main()
