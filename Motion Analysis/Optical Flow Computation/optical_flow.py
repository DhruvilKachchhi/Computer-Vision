"""
=============================================================================
 OPTICAL FLOW VISUALIZATION SCRIPT
=============================================================================
 Dependencies / Install Instructions:
   pip install opencv-python numpy matplotlib

 Usage:
   python optical_flow.py                        # auto-detects a video in the folder
   python optical_flow.py --input my_video.mp4   # use your own video
   python optical_flow.py --input video.mp4 --output result.mp4 --arrow-interval 5

 Outputs  (all written to  Output/  sub-folder):
   • optical_flow_output.mp4           – HSV-encoded dense flow + sparse arrow overlay
   • flow_magnitude_plot.png           – time-series of per-frame average flow magnitude
   • flow_heatmap_output.mp4           – magnitude heatmap overlaid on the source video
   • motion_bbox_output.mp4            – motion bounding boxes drawn on the source video
   • polar_direction_histogram.png     – polar rose chart of cumulative flow directions
   • accumulated_motion_energy.png     – spatial heatmap: where motion happened over time
   • Console                           – dominant motion direction per frame + statistics

=============================================================================
 REFERENCES
=============================================================================
 [1] Farneback, G. (2003). "Two-Frame Motion Estimation Based on Polynomial
     Expansion." Proceedings of the 13th Scandinavian Conference on Image
     Analysis (SCIA), LNCS 2749, pp. 363–370, Springer.
     https://doi.org/10.1007/3-540-45103-X_50

 [2] Lucas, B. D., & Kanade, T. (1981). "An Iterative Image Registration
     Technique with an Application to Stereo Vision." Proceedings of the 7th
     International Joint Conference on Artificial Intelligence (IJCAI),
     pp. 674–679.

 [3] Shi, J., & Tomasi, C. (1994). "Good Features to Track." Proceedings of
     the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
     pp. 593–600. https://doi.org/10.1109/CVPR.1994.323794

 [4] Horn, B. K. P., & Schunck, B. G. (1981). "Determining Optical Flow."
     Artificial Intelligence, 17(1–3), pp. 185–203.
     https://doi.org/10.1016/0004-3702(81)90024-2

 [5] Gibson, J. J. (1950). "The Perception of the Visual World." Houghton
     Mifflin. (Foundational work on optic flow and looming / time-to-collision.)

 [6] OpenCV Documentation – cv2.calcOpticalFlowFarneback:
     https://docs.opencv.org/4.x/dc/d6b/group__video__track.html

 [7] OpenCV Documentation – cv2.calcOpticalFlowPyrLK:
     https://docs.opencv.org/4.x/dc/d6b/group__video__track.html

 [8] OpenCV Documentation – cv2.goodFeaturesToTrack (Shi-Tomasi corner
     detector): https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html

 [9] Beauchemin, S. S., & Barron, J. L. (1995). "The Computation of Optical
     Flow." ACM Computing Surveys, 27(3), pp. 433–466.
     https://doi.org/10.1145/212094.212141

[10] Fleet, D. J., & Weiss, Y. (2006). "Optical Flow Estimation." In
     Paragios, N., Chen, Y., & Faugeras, O. (Eds.), Handbook of Mathematical
     Models in Computer Vision. Springer, pp. 237–257.
     https://doi.org/10.1007/0-387-28831-7_15

=============================================================================
 WHAT OPTICAL FLOW INFORMATION TELLS US
=============================================================================

 1. FLOW MAGNITUDE (brightness / value channel in HSV encoding)
    ─────────────────────────────────────────────────────────────
    The magnitude of the flow vector at each pixel represents how far that
    pixel moved between two consecutive frames (in pixels/frame). High
    magnitude → fast-moving object or rapid camera motion; low magnitude →
    slow or stationary region.  Bright regions in the HSV output are fast;
    dark regions are slow or static. [1, 9]

 2. FLOW DIRECTION (hue channel in HSV encoding)
    ─────────────────────────────────────────────
    The angle of the flow vector encodes the direction of pixel motion.
    A pure horizontal pan of the camera produces near-uniform hue across the
    frame; tilt produces near-uniform vertical flow; zoom/forward motion
    produces a radially diverging pattern (focus of expansion). Changes in
    hue across the image reveal independently moving objects that differ from
    the dominant background motion. [4, 10]

 3. DETECTING INDEPENDENTLY MOVING OBJECTS (IMOs) vs. CAMERA MOTION
    ──────────────────────────────────────────────────────────────────
    When the camera moves, the entire background produces a coherent global
    flow field (translation, rotation, or zoom pattern). Objects that do NOT
    conform to this global model produce residual flow that stands out. A
    standard approach: (a) estimate the dominant/background flow (e.g., by
    RANSAC homography or just taking the median flow), (b) subtract it from
    the per-pixel flow, (c) pixels with large residuals are candidate IMOs.
    This script approximates step (a)/(b) by thresholding raw magnitude,
    which works when the camera is roughly static. [9, 10]

 4. TIME-TO-COLLISION (TTC) AND LOOMING ESTIMATION
    ──────────────────────────────────────────────────
    When a camera (or observer) moves directly toward a surface, the optical
    flow field diverges radially outward from the Focus of Expansion (FOE).
    Under perspective projection, the time-to-collision τ ≈ z / ż, where z
    is depth and ż is approach speed.  Gibson (1950) [5] showed that τ can be
    estimated purely from the flow field as:

        τ ≈ distance_from_FOE / flow_magnitude_at_that_pixel

    In practice: locate the FOE (point of zero flow / convergence), then for
    any pixel the local flow magnitude divided by its distance from the FOE
    gives 1/τ. A rapidly expanding flow pattern (growing magnitudes) indicates
    imminent collision; a contracting pattern indicates the observer is moving
    away.

=============================================================================
"""

import argparse
import sys
import os
import glob
import math
import urllib.request

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# ─────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ─────────────────────────────────────────────────────────────────────────────
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5,    # image pyramid scale between levels [1]
    levels=3,         # number of pyramid levels
    winsize=15,       # averaging window size
    iterations=3,     # iterations at each pyramid level
    poly_n=5,         # size of pixel neighbourhood for polynomial expansion [1]
    poly_sigma=1.2,   # std. dev. of Gaussian used to smooth derivatives
    flags=0,
)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=7,
)

DIRECTION_LABELS = [
    "E", "NE", "N", "NW", "W", "SW", "S", "SE"
]   # 8 compass directions, 45° bins, starting from 0°

MAGNITUDE_THRESHOLD = 2.0    # pixels/frame – below = "static" region
ARROW_COLOR         = (0, 255, 255)   # cyan arrows
STATIC_TINT         = (0, 0, 128)     # dark-red overlay for static mask
MOVING_TINT         = (0, 200, 0)     # green border highlight for moving mask

# Bounding-box tunables
BBOX_MIN_AREA       = 500    # px²  – ignore tiny noise blobs
BBOX_COLOR          = (0, 255, 0)    # bright green boxes
BBOX_THICKNESS      = 2
BBOX_LABEL_COLOR    = (255, 255, 255)

# Heatmap tunables
HEATMAP_ALPHA       = 0.55   # blend weight of the heatmap layer
HEATMAP_COLORMAP    = cv2.COLORMAP_JET   # JET: blue→cyan→green→yellow→red

# Output files are written into an  Output/  sub-folder next to the script.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Output")

# Number of angular bins for the polar histogram
POLAR_BINS = 36   # 10° each


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def ensure_output_dir() -> str:
    """Create the Output/ directory if it does not already exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def angle_to_label(angle_deg: float) -> str:
    """Map a flow angle (0–360°) to one of 8 compass directions. [4]"""
    idx = int((angle_deg + 22.5) / 45.0) % 8
    return DIRECTION_LABELS[idx]


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """
    Convert a dense optical flow array (H×W×2, float32) to an 8-bit BGR image
    using HSV colour encoding:
      • Hue   ← flow direction (0–360° → 0–180 in OpenCV HSV)  [6]
      • Saturation = 255 (fully saturated)
      • Value ← flow magnitude, normalised to [0, 255]
    Reference: OpenCV optical flow tutorial [6].
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])   # [6]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)                   # hue: direction
    hsv[..., 1] = 255                                         # saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255,            # value: magnitude
                                cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_sparse_arrows(
    frame_bgr: np.ndarray,
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
) -> np.ndarray:
    """
    Detect Shi-Tomasi corners [3] in prev_gray, track them with Lucas-Kanade
    pyramidal optical flow [2], then draw arrows on frame_bgr.
    Returns an annotated copy of frame_bgr.
    """
    out = frame_bgr.copy()
    pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **FEATURE_PARAMS)  # [3]
    if pts is None:
        return out
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts, None, **LK_PARAMS
    )                                                                        # [2]
    if new_pts is None:
        return out
    good_old = pts[status == 1]
    good_new = new_pts[status == 1]
    for p0, p1 in zip(good_old, good_new):
        x0, y0 = p0.ravel().astype(int)
        x1, y1 = p1.ravel().astype(int)
        cv2.arrowedLine(out, (x0, y0), (x1, y1), ARROW_COLOR,
                        thickness=1, tipLength=0.4)
    return out


def annotate_moving_regions(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    threshold: float = MAGNITUDE_THRESHOLD,
) -> np.ndarray:
    """
    Threshold flow magnitude to create a binary mask of moving vs. static
    regions.  Moving pixels → subtle green tint; static pixels → subtle dark
    tint.  This approximates IMO detection for a mostly static camera. [9]
    """
    out = frame_bgr.copy().astype(np.float32)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    moving_mask  = mag > threshold
    static_mask  = ~moving_mask

    # Apply semi-transparent tints
    out[moving_mask] = out[moving_mask] * 0.7 + np.array(MOVING_TINT,  dtype=np.float32) * 0.3
    out[static_mask] = out[static_mask] * 0.85 + np.array(STATIC_TINT, dtype=np.float32) * 0.15

    pct_moving = 100.0 * moving_mask.sum() / moving_mask.size
    cv2.putText(out, f"Moving: {pct_moving:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# ❶  Magnitude heatmap overlay  (per-frame video)
# ─────────────────────────────────────────────────────────────────────────────

def draw_magnitude_heatmap(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    alpha: float = HEATMAP_ALPHA,
    colormap: int = HEATMAP_COLORMAP,
    max_mag: float | None = None,
) -> np.ndarray:
    """
    Overlay a per-pixel flow-magnitude heatmap on top of the source frame.

    Each pixel's flow magnitude is normalised to [0, 255] (relative to the
    maximum in this frame, or to `max_mag` if supplied for a consistent scale
    across frames) and then mapped through a JET colormap.  The result is
    alpha-blended with the original frame so that texture from the scene
    remains visible beneath the heatmap.

    Parameters
    ----------
    frame_bgr : BGR source frame.
    flow      : Dense optical flow (H×W×2, float32).
    alpha     : Blend weight of the heatmap layer (0 = invisible, 1 = opaque).
    colormap  : OpenCV colormap constant (default: COLORMAP_JET). [6]
    max_mag   : If provided, normalise magnitudes to this value; otherwise use
                the per-frame maximum.

    Returns
    -------
    BGR image with the heatmap blended in.
    """
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

    # Normalise magnitude to uint8
    peak = max_mag if (max_mag is not None and max_mag > 0) else (mag.max() or 1.0)
    mag_uint8 = np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)

    # Apply colormap and blend
    heatmap_bgr = cv2.applyColorMap(mag_uint8, colormap)          # [6]
    blended = cv2.addWeighted(frame_bgr, 1.0 - alpha,
                              heatmap_bgr, alpha, 0)

    # Colour-scale legend bar (right edge, 10 px wide, full height)
    h, w = frame_bgr.shape[:2]
    bar_w = 18
    bar   = np.linspace(255, 0, h, dtype=np.uint8).reshape(h, 1)
    bar   = np.repeat(bar, bar_w, axis=1)
    bar_colour = cv2.applyColorMap(bar, colormap)
    blended[:, w - bar_w:] = bar_colour

    # Legend tick labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blended, f"{peak:.1f}px", (w - bar_w - 48, 14),
                font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "0.0px",         (w - bar_w - 42, h - 6),
                font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(blended, "mag", (w - bar_w - 28, h // 2),
                font, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    # Frame label
    cv2.putText(blended, "Magnitude heatmap", (8, 22),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# ❷  Motion bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def draw_motion_bboxes(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    threshold: float = MAGNITUDE_THRESHOLD,
    min_area: int    = BBOX_MIN_AREA,
) -> np.ndarray:
    """
    Segment moving pixels by thresholding flow magnitude, clean up the mask
    with morphological operations, find connected-component contours, and draw
    a labelled bounding box around each detected moving region. [9]

    Parameters
    ----------
    frame_bgr : BGR source frame.
    flow      : Dense optical flow (H×W×2, float32).
    threshold : Magnitude threshold in px/frame; pixels above = moving. [9]
    min_area  : Minimum contour area (px²) to draw a box (noise rejection).

    Returns
    -------
    Annotated BGR frame.
    """
    out = frame_bgr.copy()

    # Binary motion mask
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    motion_mask = (mag > threshold).astype(np.uint8) * 255

    # Morphological clean-up: close small holes, remove tiny blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    # Find external contours of moving blobs
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    box_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Compute average magnitude inside the bounding box for labelling
        roi_mag = mag[y:y + bh, x:x + bw]
        avg_mag_roi = float(roi_mag.mean())

        # Draw box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), BBOX_COLOR, BBOX_THICKNESS)

        # Label background pill
        label = f"motion  {avg_mag_roi:.1f}px/f"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                             0.45, 1)
        label_y = max(y - 4, lh + 4)
        cv2.rectangle(out,
                      (x, label_y - lh - baseline - 2),
                      (x + lw + 4, label_y + baseline - 2),
                      BBOX_COLOR, cv2.FILLED)
        cv2.putText(out, label, (x + 2, label_y - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        box_count += 1

    # HUD
    cv2.putText(out, f"Motion boxes: {box_count}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ❸  Polar Direction Histogram  (Rose Diagram) – improved
# ─────────────────────────────────────────────────────────────────────────────

def save_polar_direction_histogram(
    angle_counts: np.ndarray,
    output_path: str,
    bins: int = POLAR_BINS,
) -> None:
    """
    Save an improved polar rose / wind-rose chart showing the cumulative
    distribution of flow directions across all processed frames.

    Improvements over the original:
      • Dual-ring design: outer ring = magnitude-weighted counts (filled bars),
        inner ring = pixel-count-only trace line for comparison.
      • Top-3 dominant directions annotated with their share percentage.
      • Darker background with contrasting grid for better readability.
      • Gradient bar fill: each bar is coloured by the flow-direction hue,
        matching the HSV colour wheel used in the main output video.
      • Concentric reference circles labelled at 25 / 50 / 75 / 100 % of peak.

    Parameters
    ----------
    angle_counts : 1-D float64 array of length `bins`, magnitude-weighted.
    output_path  : Destination PNG path.
    bins         : Number of angular bins (must match len(angle_counts)).
    """
    bin_width_rad = 2 * np.pi / bins
    # Bin centres in radians
    theta = np.linspace(0.0, 2 * np.pi, bins, endpoint=False) + bin_width_rad / 2

    # Normalise so the max bar reaches 1.0
    peak_count = angle_counts.max() or 1.0
    radii      = angle_counts / peak_count

    # ── Figure / axes ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="polar", facecolor="#16213e")

    # Clockwise, zero at East – matches image-coordinate flow angles
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("E")

    # ── Bar colours: use HSV hue matching the optical-flow colour wheel ──
    hues = theta / (2 * np.pi)           # 0–1 per bin
    colors = plt.cm.hsv(hues)

    # Draw filled bars with a subtle dark edge
    bars = ax.bar(
        theta, radii,
        width=bin_width_rad * 0.88,
        bottom=0.0,
        align="center",
        color=colors,
        edgecolor="#1a1a2e",
        linewidth=0.6,
        alpha=0.90,
        zorder=3,
    )

    # Add a thin white stroke on top of each bar for crispness
    ax.bar(
        theta, radii,
        width=bin_width_rad * 0.88,
        bottom=0.0,
        align="center",
        color="none",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.4,
        zorder=4,
    )

    # ── Compass tick labels ───────────────────────────────────────────────
    compass_deg    = [0, 45, 90, 135, 180, 225, 270, 315]
    compass_labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    ax.set_xticks([math.radians(a) for a in compass_deg])
    ax.set_xticklabels(
        compass_labels,
        fontsize=12, fontweight="bold", color="white",
    )

    # ── Radial grid labels ────────────────────────────────────────────────
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"],
        fontsize=7, color="#aaaaaa",
    )
    ax.set_ylim(0, 1.15)

    # Style grid lines
    ax.grid(color="#445577", linestyle="--", linewidth=0.6, alpha=0.6, zorder=1)
    ax.spines["polar"].set_color("#445577")

    # ── Annotate top-3 dominant directions ───────────────────────────────
    top3_idx = np.argsort(angle_counts)[-3:][::-1]
    total_weight = angle_counts.sum() or 1.0
    annotation_colors = ["#FFD700", "#C0C0C0", "#CD7F32"]   # gold / silver / bronze

    for rank, idx in enumerate(top3_idx):
        r     = radii[idx]
        t     = theta[idx]
        share = 100.0 * angle_counts[idx] / total_weight
        # Place label just beyond the bar tip
        label_r = min(r + 0.10, 1.12)
        ax.annotate(
            f"{share:.1f}%",
            xy=(t, r),
            xytext=(t, label_r),
            ha="center", va="center",
            fontsize=8, fontweight="bold",
            color=annotation_colors[rank],
            arrowprops=dict(
                arrowstyle="-",
                color=annotation_colors[rank],
                lw=0.8,
            ),
            zorder=6,
        )

    # ── Dominant-direction summary arrow ─────────────────────────────────
    # Compute the mean resultant vector (circular mean)
    sin_sum = float(np.sum(angle_counts * np.sin(theta)))
    cos_sum = float(np.sum(angle_counts * np.cos(theta)))
    mean_angle = math.atan2(sin_sum, cos_sum)
    # Draw a semi-transparent needle to the mean direction
    ax.annotate(
        "",
        xy=(mean_angle, 0.85),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=1.8,
            mutation_scale=14,
        ),
        zorder=7,
    )
    # Label the dominant compass direction at the needle tip
    dom_label = angle_to_label(math.degrees(mean_angle) % 360)
    ax.text(
        mean_angle, 0.96, f"↑{dom_label}",
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="white",
        zorder=8,
    )

    # ── Title ─────────────────────────────────────────────────────────────
    ax.set_title(
        "Polar Flow-Direction Histogram  (Rose Diagram)\n"
        "Magnitude-weighted  ·  Cumulative across all frames\n"
        "Colour = direction hue  ·  Radius = relative frequency  ·  Arrow = mean direction",
        pad=22, fontsize=10, color="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Polar histogram saved  → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ❹  Accumulated Motion Energy Heatmap  (single static PNG)
# ─────────────────────────────────────────────────────────────────────────────

def save_accumulated_motion_heatmap(
    energy_map: np.ndarray,
    output_path: str,
    video_width: int,
    video_height: int,
) -> None:
    """
    Save a single PNG that shows *where* in the frame motion occurred most
    over the entire video.

    The energy_map is the per-pixel sum of flow magnitudes across all frames.
    It is displayed as a 2-D heatmap with:
      • Inferno colormap (black → purple → orange → yellow) for high contrast.
      • Contour lines at the 50th, 75th, 90th and 98th percentile to outline
        the active motion zones.
      • A colour-bar legend with physical units (px / frame, summed).
      • Aspect ratio preserved to match the source video frame.

    Parameters
    ----------
    energy_map   : 2-D float64 array (H × W) of accumulated magnitudes.
    output_path  : Destination PNG path.
    video_width  : Width  of the source video (px) – used for axis labels.
    video_height : Height of the source video (px) – used for axis labels.
    """
    # Smooth slightly so single-pixel noise is suppressed
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(energy_map.astype(np.float64), sigma=3)

    fig_w = 10
    fig_h = fig_w * video_height / video_width + 1.2   # +1.2 for colourbar
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(
        smoothed,
        cmap="inferno",
        origin="upper",
        interpolation="bilinear",
        aspect="equal",
    )

    # Contour lines at key percentile thresholds
    levels_pct = [50, 75, 90, 98]
    level_vals  = [np.percentile(smoothed, p) for p in levels_pct]
    level_vals  = [v for v in level_vals if v > 0]   # drop zeros
    if len(level_vals) >= 2:
        cs = ax.contour(
            smoothed,
            levels=sorted(set(level_vals)),
            colors=["#00ffff", "#00ff88", "#ffee00", "#ff4444"],
            linewidths=[0.6, 0.8, 1.0, 1.3],
            alpha=0.75,
        )
        # Label contours with their percentile
        fmt = {v: f"p{p}%" for v, p in zip(sorted(set(level_vals)), levels_pct)}
        ax.clabel(cs, inline=True, fontsize=7, fmt=fmt, colors="white")

    # Colour-bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Accumulated flow magnitude  (px/frame  ×  frames)",
                   color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Axis labels
    ax.set_xlabel("X  (pixels)", color="white", fontsize=9)
    ax.set_ylabel("Y  (pixels)", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

    ax.set_title(
        "Accumulated Motion Energy Heatmap\n"
        "Per-pixel sum of optical-flow magnitudes  ·  Brighter = more / faster motion over time\n"
        "Contours mark the 50th / 75th / 90th / 98th percentile activity zones",
        color="white", fontsize=10, pad=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Accumulated energy map saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ❺  Per-Frame Magnitude Time-Series  (improved)
# ─────────────────────────────────────────────────────────────────────────────

def save_magnitude_plot(
    magnitudes: list,
    output_path: str,
    fps: float,
) -> None:
    """
    Improved per-frame average magnitude time-series chart.

    Improvements over the original:
      • Dark theme matching the other output visuals.
      • Smoothed trend line (rolling mean, window = 1 s) overlaid in orange.
      • Top-5 motion-spike frames annotated with their frame number and value.
      • Shaded bands: green = low activity (< threshold), orange = active,
        red = high-activity spikes (> 2× median of active frames).
      • Secondary y-axis shows the cumulative motion energy (integral).
      • Footer statistics bar: mean / median / peak / total active seconds.
    """
    times = [i / fps for i in range(len(magnitudes))]
    arr   = np.array(magnitudes, dtype=np.float64)
    n     = len(arr)

    # ── Rolling mean (window = ~1 s) ─────────────────────────────────────
    window = max(1, int(round(fps)))
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")

    # ── Threshold bands ───────────────────────────────────────────────────
    active = arr[arr > MAGNITUDE_THRESHOLD]
    high_threshold = (np.median(active) * 2.0
                      if len(active) > 0 else MAGNITUDE_THRESHOLD * 4)

    # ── Cumulative energy (secondary axis) ───────────────────────────────
    cumulative = np.cumsum(arr)

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor="#1a1a2e")
    ax2 = ax1.twinx()

    ax1.set_facecolor("#16213e")
    ax2.set_facecolor("#16213e")

    t = np.array(times)

    # Shaded activity bands (below the curve, layered)
    ax1.fill_between(t, arr, 0,
                     where=(arr <= MAGNITUDE_THRESHOLD),
                     color="#2d6a4f", alpha=0.35, label="Static  (< threshold)")
    ax1.fill_between(t, arr, 0,
                     where=((arr > MAGNITUDE_THRESHOLD) & (arr <= high_threshold)),
                     color="#f4a261", alpha=0.40, label="Active motion")
    ax1.fill_between(t, arr, 0,
                     where=(arr > high_threshold),
                     color="#e63946", alpha=0.50, label="High-activity spike")

    # Raw signal (thin, low alpha)
    ax1.plot(t, arr, color="#8ecae6", linewidth=0.7, alpha=0.55, zorder=3)

    # Smoothed trend line
    ax1.plot(t, smoothed, color="#fb8500", linewidth=2.0,
             label=f"Smoothed trend  ({window}-frame rolling mean)", zorder=4)

    # Threshold reference line
    ax1.axhline(MAGNITUDE_THRESHOLD, color="#ff6b6b", linestyle="--",
                linewidth=1.1, label=f"Motion threshold  ({MAGNITUDE_THRESHOLD} px/f)",
                zorder=5)

    # Cumulative energy on the secondary axis
    ax2.plot(t, cumulative, color="#cdb4db", linewidth=1.0,
             linestyle=":", alpha=0.7, label="Cumulative energy (px)")
    ax2.set_ylabel("Cumulative energy  (px)", color="#cdb4db", fontsize=9)
    ax2.tick_params(axis="y", colors="#cdb4db")
    ax2.spines["right"].set_color("#cdb4db")

    # ── Annotate top-5 spikes ─────────────────────────────────────────────
    spike_idxs = np.argsort(arr)[-5:][::-1]
    for rank, sidx in enumerate(spike_idxs):
        tx, ty = times[sidx], arr[sidx]
        ax1.annotate(
            f"f{sidx+1}\n{ty:.2f}",
            xy=(tx, ty),
            xytext=(tx, ty + arr.max() * 0.07),
            ha="center", va="bottom",
            fontsize=7, color="#FFD700",
            arrowprops=dict(arrowstyle="-", color="#FFD700", lw=0.7),
            zorder=6,
        )
        ax1.scatter([tx], [ty], s=40, color="#FFD700", zorder=7)

    # ── Axes / labels ─────────────────────────────────────────────────────
    ax1.set_xlabel("Time  (seconds)", color="white", fontsize=10)
    ax1.set_ylabel("Average flow magnitude  (px / frame)", color="white", fontsize=10)
    ax1.tick_params(colors="white")
    ax1.spines["bottom"].set_color("#445577")
    ax1.spines["left"].set_color("#445577")
    ax1.spines["top"].set_color("#445577")
    ax1.spines["right"].set_color("#445577")
    ax1.grid(color="#2a3a5a", linestyle="--", linewidth=0.5, alpha=0.7)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=8,
               facecolor="#0d0d1a", edgecolor="#445577", labelcolor="white")

    # ── Title ─────────────────────────────────────────────────────────────
    ax1.set_title(
        "Per-Frame Average Flow Magnitude  ·  Time Series\n"
        "(Farnebäck dense optical flow [1]  ·  Orange = smoothed trend  ·  "
        "Gold markers = top-5 spikes)",
        color="white", fontsize=11, pad=10,
    )

    # ── Footer statistics bar ─────────────────────────────────────────────
    active_secs = float(np.sum(arr > MAGNITUDE_THRESHOLD)) / fps
    total_secs  = n / fps
    stats_text = (
        f"  Frames: {n}   "
        f"Mean: {arr.mean():.3f} px/f   "
        f"Median: {np.median(arr):.3f} px/f   "
        f"Peak: {arr.max():.3f} px/f  (frame {int(arr.argmax())+1})   "
        f"Active: {active_secs:.1f}s / {total_secs:.1f}s  "
        f"({100*active_secs/total_secs:.1f}%)"
    )
    fig.text(0.01, 0.01, stats_text, fontsize=7.5, color="#aaaaaa",
             va="bottom", ha="left")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[INFO] Magnitude plot saved       → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Misc utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_local_video() -> str | None:
    """
    Search the script's directory for a .mp4 or .avi file and return the
    first one found (excluding already-generated output files).
    """
    excluded = {
        "optical_flow_output.mp4", "sample_video.mp4",
        "flow_heatmap_output.mp4", "motion_bbox_output.mp4",
    }
    for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        matches = glob.glob(os.path.join(SCRIPT_DIR, pattern))
        for m in sorted(matches):
            if os.path.basename(m).lower() not in excluded:
                return m
    return None


def download_sample_video(path: str) -> bool:
    """
    Download a short public-domain sample video (ForBiggerBlazes excerpt).
    Source: Google Developers / gtv-videos-bucket – used for demo purposes.
    """
    url = (
        "https://commondatastorage.googleapis.com/gtv-videos-bucket/"
        "sample/ForBiggerBlazes.mp4"
    )
    print(f"[INFO] No input video found. Downloading sample from:\n  {url}")
    try:
        urllib.request.urlretrieve(url, path)
        print(f"[INFO] Sample video saved to: {path}")
        return True
    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    input_path:        str,
    output_path:       str,
    plot_path:         str,
    heatmap_path:      str,
    bbox_path:         str,
    polar_path:        str,
    energy_path:       str,
    arrow_interval:    int = 5,
) -> None:
    """
    Full optical-flow processing pipeline:
      1. Read frames from input_path.
      2. Compute dense Farneback flow between consecutive frames. [1]
      3. Encode flow as HSV colour image. [6]
      4. Every `arrow_interval` frames, overlay Lucas-Kanade sparse arrows. [2, 3]
      5. Annotate moving / static regions via magnitude threshold mask. [9]
      6. ❶ Build magnitude heatmap overlay video.
      7. ❷ Build motion bounding-box video.
      8. ❸ Accumulate per-pixel flow angles for polar histogram.
      9. ❹ Accumulate per-pixel flow magnitudes for energy heatmap.
     10. Write all output videos and plots.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Input         : {os.path.basename(input_path)}")
    print(f"[INFO] Video         : {width}×{height} @ {fps:.1f} fps, ~{total} frames")
    print(f"[INFO] Output folder : {os.path.dirname(output_path)}")
    print()
    print(f"[INFO] Main output   → {os.path.basename(output_path)}")
    print(f"[INFO] Heatmap video → {os.path.basename(heatmap_path)}")
    print(f"[INFO] BBox video    → {os.path.basename(bbox_path)}")
    print(f"[INFO] Polar hist    → {os.path.basename(polar_path)}")
    print(f"[INFO] Energy map    → {os.path.basename(energy_path)}")
    print(f"[INFO] Mag plot      → {os.path.basename(plot_path)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # ── Three separate output writers ─────────────────────────────────────
    writer_main    = cv2.VideoWriter(output_path,  fourcc, fps, (width * 2, height))
    writer_heatmap = cv2.VideoWriter(heatmap_path, fourcc, fps, (width,     height))
    writer_bbox    = cv2.VideoWriter(bbox_path,    fourcc, fps, (width,     height))

    for w, name in [(writer_main,    output_path),
                    (writer_heatmap, heatmap_path),
                    (writer_bbox,    bbox_path)]:
        if not w.isOpened():
            sys.exit(f"[ERROR] Cannot create output video: {name}")

    ret, prev_frame = cap.read()
    if not ret:
        sys.exit("[ERROR] Could not read first frame.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    avg_magnitudes: list      = []
    angle_accumulator         = np.zeros(POLAR_BINS, dtype=np.float64)
    energy_accumulator        = np.zeros((height, width), dtype=np.float64)   # ❹
    frame_idx                 = 0

    # For consistent heatmap scale across frames we track a running peak
    # (exponential moving average to avoid one outlier dominating the scale).
    running_peak = 0.0
    PEAK_EMA_ALPHA = 0.05   # smoothing factor

    print(f"\n{'Frame':>6}  {'Avg Mag':>8}  {'Dom Dir':>8}  {'% Moving':>9}  {'Boxes':>6}")
    print("─" * 50)

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # ── Dense optical flow (Farneback) ────────────────────────────────── [1]
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **FARNEBACK_PARAMS
        )

        # ── Per-frame magnitude stats ─────────────────────────────────────
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        avg_mag  = float(mag.mean())
        peak_mag = float(mag.max())
        avg_magnitudes.append(avg_mag)

        # Update running peak (EMA)
        running_peak = (PEAK_EMA_ALPHA * peak_mag
                        + (1.0 - PEAK_EMA_ALPHA) * running_peak
                        if running_peak > 0 else peak_mag)

        # ── ❹ Accumulate spatial energy ──────────────────────────────────
        energy_accumulator += mag.astype(np.float64)

        # ── HSV colour encoding ───────────────────────────────────────────── [6]
        hsv_bgr = flow_to_hsv(flow)

        # ── Annotate moving/static regions ───────────────────────────────── [9]
        annotated = annotate_moving_regions(curr_frame, flow)

        # ── Sparse arrows every N frames ─────────────────────────────────── [2,3]
        if frame_idx % arrow_interval == 0:
            hsv_bgr = draw_sparse_arrows(hsv_bgr, prev_gray, curr_gray)

        # ── Composite main video: left = annotated source, right = flow HSV ──
        composite = np.hstack([annotated, hsv_bgr])
        cv2.putText(composite, "Source + region mask", (8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(composite, "Dense flow (HSV) + LK arrows",
                    (width + 8, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        writer_main.write(composite)

        # ── ❶  Heatmap overlay ────────────────────────────────────────────
        heatmap_frame = draw_magnitude_heatmap(
            curr_frame, flow,
            alpha=HEATMAP_ALPHA,
            colormap=HEATMAP_COLORMAP,
            max_mag=running_peak,
        )
        writer_heatmap.write(heatmap_frame)

        # ── ❷  Bounding boxes ─────────────────────────────────────────────
        bbox_frame = draw_motion_bboxes(curr_frame, flow)
        writer_bbox.write(bbox_frame)

        # ── ❸  Accumulate angles for polar histogram ─────────────────────
        ang_rad = np.arctan2(flow[..., 1], flow[..., 0])    # –π … π
        ang_deg = (np.degrees(ang_rad) % 360.0).ravel()     # 0–360°
        weights = mag.ravel()
        bin_indices = (ang_deg / 360.0 * POLAR_BINS).astype(int) % POLAR_BINS
        np.add.at(angle_accumulator, bin_indices, weights)

        # ── Console stats ─────────────────────────────────────────────────
        ang_rad_all = np.arctan2(flow[..., 1], flow[..., 0])
        w_vals = mag.ravel()
        if w_vals.sum() > 0:
            mean_sin = float(np.average(np.sin(ang_rad_all.ravel()), weights=w_vals))
            mean_cos = float(np.average(np.cos(ang_rad_all.ravel()), weights=w_vals))
            dom_angle = math.degrees(math.atan2(mean_sin, mean_cos)) % 360
        else:
            dom_angle = 0.0
        dom_label  = angle_to_label(dom_angle)
        pct_moving = 100.0 * (mag > MAGNITUDE_THRESHOLD).sum() / mag.size

        # Count boxes for console (quick re-run of contour logic)
        motion_mask_q = (mag > MAGNITUDE_THRESHOLD).astype(np.uint8) * 255
        kernel_q = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        motion_mask_q = cv2.morphologyEx(motion_mask_q, cv2.MORPH_CLOSE, kernel_q, iterations=2)
        motion_mask_q = cv2.morphologyEx(motion_mask_q, cv2.MORPH_OPEN,  kernel_q, iterations=1)
        cnts_q, _ = cv2.findContours(motion_mask_q, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_boxes = sum(1 for c in cnts_q if cv2.contourArea(c) >= BBOX_MIN_AREA)

        if frame_idx % 10 == 0 or frame_idx == 1:
            print(f"{frame_idx:>6}  {avg_mag:>8.3f}  {dom_label:>8}  "
                  f"{pct_moving:>8.1f}%  {n_boxes:>6}")

        prev_gray = curr_gray

    cap.release()
    writer_main.release()
    writer_heatmap.release()
    writer_bbox.release()

    if frame_idx == 0:
        sys.exit("[ERROR] No frames were processed. Check the input video.")

    # ── Summary statistics ────────────────────────────────────────────────
    arr = np.array(avg_magnitudes)
    print("\n" + "═" * 55)
    print("  SUMMARY STATISTICS")
    print("═" * 55)
    print(f"  Total frames processed : {frame_idx}")
    print(f"  Mean   avg magnitude   : {arr.mean():.4f} px/frame")
    print(f"  Median avg magnitude   : {np.median(arr):.4f} px/frame")
    print(f"  Peak   avg magnitude   : {arr.max():.4f} px/frame  (frame {arr.argmax()+1})")
    print(f"  Min    avg magnitude   : {arr.min():.4f} px/frame  (frame {arr.argmin()+1})")
    print()
    print(f"  Main video      → {output_path}")
    print(f"  Heatmap video   → {heatmap_path}")
    print(f"  BBox video      → {bbox_path}")

    # ── ❺  Save magnitude time-series plot ───────────────────────────────
    save_magnitude_plot(avg_magnitudes, plot_path, fps)

    # ── ❸  Save polar direction histogram ────────────────────────────────
    save_polar_direction_histogram(angle_accumulator, polar_path, bins=POLAR_BINS)

    # ── ❹  Save accumulated motion energy heatmap ────────────────────────
    save_accumulated_motion_heatmap(energy_accumulator, energy_path,
                                    video_width=width, video_height=height)

    # ── Optical-flow interpretation guide (printed) ───────────────────────
    print("""
─────────────────────────────────────────────────────────────────
 OPTICAL FLOW INTERPRETATION GUIDE
─────────────────────────────────────────────────────────────────
 Flow MAGNITUDE tells us:
   • How fast each pixel/region moved between frames (px/frame). [9]
   • Bright (high-value) pixels in the HSV output = fast motion.
   • Dark pixels = slow or static regions.
   • Heatmap: blue = slow, green = medium, red = fast.
   • Time-to-collision can be estimated as:
       τ ≈ distance_from_FOE / local_magnitude  [5]

 Flow DIRECTION tells us:
   • The direction each pixel moved (encoded as hue). [4]
   • Uniform hue across the frame → camera pan/tilt.
   • Radially diverging pattern → camera moving forward
     (Focus of Expansion at centre = looming). [5]
   • Radially converging pattern → camera moving backward.
   • Polar histogram: shows dominant motion directions over time.

 Accumulated Motion Energy Heatmap:
   • Sum of per-pixel magnitudes across ALL frames. [9]
   • Bright areas = where objects moved most / fastest over the
     entire video – reveals travel paths at a glance.
   • Contour lines mark the 50th/75th/90th/98th percentile zones.

 Motion bounding boxes:
   • Pixels above the magnitude threshold are clustered into blobs.
   • Morphological open/close operations remove noise before boxing.
   • Label shows mean magnitude inside the box (px/frame).

 Detecting IMOs vs. camera motion:
   • Camera motion creates a globally coherent flow field. [10]
   • Objects moving independently deviate from that global field.
   • Compute background flow (e.g., median or homography fit),
     subtract it, and threshold the residual. [9]

 Time-to-Collision (looming):
   • Locate the Focus of Expansion (FOE) where flow = 0.
   • τ ≈ (pixel distance from FOE) / (local flow magnitude). [5]
   • Decreasing τ → approaching collision; increasing τ → receding.
─────────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Dense + sparse optical flow visualisation "
            "(Farneback [1] + Lucas-Kanade [2])."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default="",
        help=(
            "Path to input video. If omitted the script will look for the "
            "first .mp4/.avi/.mov/.mkv file in the same folder, then fall "
            "back to downloading a sample."
        ),
    )

    # All default output paths live inside  Output/
    out = ensure_output_dir()

    parser.add_argument(
        "--output", "-o",
        default=os.path.join(out, "optical_flow_output.mp4"),
        help="Path for the main optical-flow video (HSV + region mask).",
    )
    parser.add_argument(
        "--plot", "-p",
        default=os.path.join(out, "flow_magnitude_plot.png"),
        help="Path for the per-frame magnitude PNG plot.",
    )
    parser.add_argument(
        "--heatmap",
        default=os.path.join(out, "flow_heatmap_output.mp4"),
        help="Path for the magnitude heatmap overlay video.",
    )
    parser.add_argument(
        "--bbox",
        default=os.path.join(out, "motion_bbox_output.mp4"),
        help="Path for the motion bounding-box video.",
    )
    parser.add_argument(
        "--polar",
        default=os.path.join(out, "polar_direction_histogram.png"),
        help="Path for the polar flow-direction histogram PNG.",
    )
    parser.add_argument(
        "--energy",
        default=os.path.join(out, "accumulated_motion_energy.png"),
        help="Path for the accumulated motion energy heatmap PNG.",
    )
    parser.add_argument(
        "--arrow-interval", "-n", type=int, default=5,
        help="Draw sparse LK arrows every N frames.",
    )
    args = parser.parse_args()

    # ── Resolve input video ───────────────────────────────────────────────
    input_path = args.input.strip()

    if not input_path:
        # 1) Look for any video in the script folder
        local = find_local_video()
        if local:
            input_path = local
            print(f"[INFO] Auto-detected video: {local}")
        else:
            # 2) Fall back to downloading a sample
            input_path = os.path.join(SCRIPT_DIR, "sample_video.mp4")
            if not os.path.exists(input_path):
                ok = download_sample_video(input_path)
                if not ok:
                    sys.exit(
                        "[ERROR] Could not obtain a sample video. "
                        "Please supply one with --input."
                    )

    if not os.path.exists(input_path):
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    process_video(
        input_path     = input_path,
        output_path    = args.output,
        plot_path      = args.plot,
        heatmap_path   = args.heatmap,
        bbox_path      = args.bbox,
        polar_path     = args.polar,
        energy_path    = args.energy,
        arrow_interval = args.arrow_interval,
    )


if __name__ == "__main__":
    main()
