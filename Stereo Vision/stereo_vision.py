#!/usr/bin/env python3
"""
=============================================================================
 Uncalibrated Stereo Vision — Distance Estimation
=============================================================================
 Assignment  : Use an uncalibrated stereo setup to estimate the distance
               to a selected object (square) in a classroom.

 Camera      : Samsung Galaxy M34
 Specs       : 4080 × 3060  |  12 MP  |  ISO 150  |  27 mm (35 mm equiv.)
               f/1.8  |  0.0 EV
 Images      : Stored at 1600 × 1200 (JPEG)

 Stereo Setup
   Baseline (B)           = 1 foot  = 30.48 cm
   Ground-truth distance  = 7 feet  = 213.36 cm
   Object                 : square target on classroom surface

 Outputs
   • output_stereo.png  — annotated stereo pair (both images, detected square)
   • results.json       — serialised matrices for report generation
   • Console summary    — F, E, R, t and distance estimation
=============================================================================
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive / file-only backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
#  CAMERA INTRINSICS  — Samsung Galaxy M34
#
#  Sensor type    : 1/2.0"  →  width ≈ 6.40 mm,  height ≈ 4.80 mm
#  Sensor diagonal: sqrt(6.40² + 4.80²) ≈ 8.00 mm
#  35-mm-equivalent focal length given by EXIF: 27 mm
#  Crop factor    : 43.27 / 8.00 ≈ 5.41
#  Actual FL      : 27 / 5.41  ≈ 4.99 mm
#
#  Native resolution : 4080 × 3060 px
#  Pixel size (native): 6.40 / 4080 ≈ 0.001569 mm / px
#  f in px (native)  : 4.99 / 0.001569 ≈ 3179 px
#  Stored resolution : 1600 × 1200 px  (scale = 1600/4080 ≈ 0.3922)
#  f in px (stored)  : 3179 × 0.3922  ≈ 1247 px
# ─────────────────────────────────────────────────────────────────────────────
ORIG_W        = 4080
ORIG_H        = 3060
IMG_W         = 1600
IMG_H         = 1200

SENSOR_W_MM   = 6.40
SENSOR_H_MM   = 4.80
SENSOR_D_MM   = np.sqrt(SENSOR_W_MM**2 + SENSOR_H_MM**2)   # ≈ 8.00 mm
CROP_FACTOR   = 43.27 / SENSOR_D_MM                         # ≈ 5.409
F_35MM_EQUIV  = 27.0                                         # mm (EXIF)
F_ACTUAL_MM   = F_35MM_EQUIV / CROP_FACTOR                  # ≈ 4.99 mm
SCALE         = IMG_W / ORIG_W                              # ≈ 0.3922
F_PX_NATIVE   = (F_ACTUAL_MM * ORIG_W) / SENSOR_W_MM       # px at native res
F_PX          = F_PX_NATIVE * SCALE                         # px at 1600 wide
CX            = IMG_W / 2.0                                  # 800.0
CY            = IMG_H / 2.0                                  # 600.0

#  3 × 3 Camera Intrinsic Matrix
K = np.array([[F_PX,  0.0,  CX],
              [0.0,   F_PX, CY],
              [0.0,   0.0,  1.0]], dtype=np.float64)

# ─────────────────────────────────────────────────────────────────────────────
#  STEREO SETUP
# ─────────────────────────────────────────────────────────────────────────────
BASELINE_FT   = 1.0
BASELINE_CM   = BASELINE_FT * 30.48      # 30.48 cm
GT_FT         = 7.0                       # ground-truth distance (feet)
GT_CM         = GT_FT    * 30.48          # 213.36 cm


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1  —  LOAD IMAGES
# ═══════════════════════════════════════════════════════════════════════════════
def load_images(left_path: str, right_path: str):
    """
    Load left and right stereo images from disk.

    Returns
    -------
    left_bgr, right_bgr : (H, W, 3) uint8 BGR arrays
    """
    left  = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        sys.exit(f"[ERROR] Could not open images:\n  {left_path}\n  {right_path}")
    return left, right


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2  —  FEATURE DETECTION & MATCHING  (SIFT + BFMatcher + Lowe ratio)
# ═══════════════════════════════════════════════════════════════════════════════
def detect_and_match(gray_l: np.ndarray,
                     gray_r: np.ndarray,
                     ratio:      float = 0.75,
                     n_features: int   = 8000) -> tuple:
    """
    Detect SIFT keypoints in both images and establish correspondences.

    Algorithm
    ---------
    1. Detect SIFT keypoints and descriptors in each image.
    2. Match descriptors with Brute-Force L2 distance, k=2 nearest neighbours.
    3. Apply Lowe's ratio test: keep match m only if
           m.distance < ratio × n.distance
       where n is the second-best match.

    Parameters
    ----------
    gray_l, gray_r  : single-channel grayscale images
    ratio           : Lowe ratio threshold (default 0.75)
    n_features      : max SIFT features to detect per image

    Returns
    -------
    pts1, pts2 : (N, 2) float32 — matched pixel coordinates
    kp1,  kp2  : SIFT KeyPoint lists
    good       : list of accepted DMatch objects
    """
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(gray_l, None)
    kp2, des2 = sift.detectAndCompute(gray_r, None)

    bf  = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw if m.distance < ratio * n.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, kp1, kp2, good


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3  —  FUNDAMENTAL MATRIX  F  (Normalised 8-Point + RANSAC)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_fundamental(pts1: np.ndarray, pts2: np.ndarray) -> tuple:
    """
    Estimate the 3 × 3 Fundamental Matrix F using the normalised 8-point
    algorithm with RANSAC outlier rejection.

    Geometric meaning
    -----------------
    For any pair of corresponding image points  x  (left) and  x'  (right):

        x'ᵀ F x = 0          (epipolar constraint)

    F encodes the epipolar geometry without requiring knowledge of the
    camera intrinsics — it is estimated purely from point correspondences.

    Algorithm (OpenCV FM_RANSAC)
    ----------------------------
    1. Normalise input points (zero-mean, unit RMS distance).
    2. Build the 9-column design matrix A from each correspondence.
    3. Solve  min ‖Af‖  subject to  ‖f‖=1  via SVD → last right singular vector.
    4. Reshape to F̃ (3×3) and enforce rank-2 (zero smallest singular value).
    5. Denormalise: F = T'ᵀ F̃ T.
    6. RANSAC loop: iterate, keep the model with the most inliers satisfying
       Sampson distance < threshold.

    Parameters
    ----------
    pts1, pts2 : (N, 2) matched pixel coordinates (left, right)

    Returns
    -------
    F    : (3, 3) Fundamental Matrix  (rank 2)
    mask : boolean (N,) array — True for RANSAC inliers
    """
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method                = cv2.FM_RANSAC,
        ransacReprojThreshold = 1.0,
        confidence            = 0.999,
        maxIters              = 10000,
    )
    return F, mask.ravel().astype(bool)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4  —  ESSENTIAL MATRIX  E = Kᵀ · F · K
# ═══════════════════════════════════════════════════════════════════════════════
def compute_essential(F: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Compute the 3 × 3 Essential Matrix from F and the intrinsic matrix K.

    Relation to F
    -------------
    For identical cameras (same K):

        E = Kᵀ F K

    The Essential Matrix carries metric information (the camera rotation R
    and the unit translation direction t̂) rather than just projective
    geometry.

    Rank-2 enforcement
    ------------------
    Due to noise, E_raw may have three non-zero singular values.
    Theoretically E has exactly two equal non-zero singular values:

        SVD(E_raw) = U Σ Vᵀ,   Σ = diag(σ₁, σ₂, σ₃)

    Enforce:  σ₁ = σ₂ = (σ₁+σ₂)/2,  σ₃ = 0

        E = U · diag(σ̄, σ̄, 0) · Vᵀ

    Parameters
    ----------
    F : (3, 3) Fundamental Matrix
    K : (3, 3) Camera Intrinsic Matrix

    Returns
    -------
    E : (3, 3) Essential Matrix (rank 2, equal non-zero singular values)
    """
    E_raw = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E_raw)
    if np.linalg.det(U)  < 0: U  *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1
    s_avg = (S[0] + S[1]) / 2.0
    E = U @ np.diag([s_avg, s_avg, 0.0]) @ Vt
    return E


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5  —  ROTATION MATRIX R  and  TRANSLATION t  (SVD + cheirality)
# ═══════════════════════════════════════════════════════════════════════════════
def decompose_essential(E:      np.ndarray,
                        K:      np.ndarray,
                        pts1_in: np.ndarray,
                        pts2_in: np.ndarray) -> tuple:
    """
    Recover the camera rotation R and unit translation t̂ from E.

    SVD Decomposition
    -----------------
    E = U Σ Vᵀ,     Σ = diag(1, 1, 0)  (after normalisation)

    Define the auxiliary matrix:
        W = [[0, -1, 0],
             [1,  0, 0],
             [0,  0, 1]]

    The four candidate solutions are:
        (R₁, +t̂) ,  (R₁, −t̂)
        (R₂, +t̂) ,  (R₂, −t̂)
    where
        R₁ = U  W  Vᵀ
        R₂ = U  Wᵀ Vᵀ
        t̂  = U[:,2]  (third column of U)

    Cheirality Check
    ----------------
    Triangulate a sample of correspondences under each candidate.
    The correct (R, t) is the one for which the majority of triangulated
    3-D points lie in front of BOTH cameras (positive depth).

    Parameters
    ----------
    E        : (3, 3) Essential Matrix
    K        : (3, 3) Intrinsic Matrix
    pts1_in  : (N, 2) inlier points in left image
    pts2_in  : (N, 2) inlier points in right image

    Returns
    -------
    R     : (3, 3) rotation matrix, det(R) = +1
    t     : (3,)   unit translation vector
    pts3d : (3, M) triangulated 3-D points for the winning candidate
    """
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U)  < 0: U  *= -1
    if np.linalg.det(Vt) < 0: Vt *= -1

    W = np.array([[ 0., -1.,  0.],
                  [ 1.,  0.,  0.],
                  [ 0.,  0.,  1.]], dtype=np.float64)

    # Left camera projection matrix: P1 = K [I | 0]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])

    candidates = [
        (U @ W    @ Vt,  U[:, 2]),
        (U @ W    @ Vt, -U[:, 2]),
        (U @ W.T  @ Vt,  U[:, 2]),
        (U @ W.T  @ Vt, -U[:, 2]),
    ]

    best_R, best_t, best_pts3d = None, None, None
    best_count = -1

    samp_l = pts1_in[:60].T.astype(np.float64)
    samp_r = pts2_in[:60].T.astype(np.float64)

    for R_c, t_c in candidates:
        if abs(np.linalg.det(R_c) - 1.0) > 0.05:
            continue                        # skip degenerate candidates

        # Right camera projection matrix: P2 = K [R | t]
        P2    = K @ np.hstack([R_c, t_c.reshape(3, 1)])
        pts4d = cv2.triangulatePoints(P1, P2, samp_l, samp_r)
        pts3d = pts4d[:3] / pts4d[3]       # homogeneous → 3-D

        # Cheirality: Z > 0 in both camera frames
        in_front_1 = pts3d[2] > 0
        in_front_2 = (R_c @ pts3d + t_c.reshape(3, 1))[2] > 0
        count = int(np.sum(in_front_1 & in_front_2))

        if count > best_count:
            best_count = count
            best_R, best_t, best_pts3d = R_c, t_c, pts3d

    return best_R, best_t, best_pts3d


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6  —  DETECT SQUARE OBJECT
# ═══════════════════════════════════════════════════════════════════════════════
def detect_square(bgr_img: np.ndarray,
                  min_area: int = 1000) -> tuple:
    """
    Locate the most prominent near-square / rectangular object.

    Pipeline
    --------
    1. Convert to grayscale and apply Gaussian blur (σ=1.0, kernel 5×5).
    2. Canny edge detection (low=30, high=100).
    3. Morphological closing (3×3 rect, 2 iterations) to bridge edge gaps.
    4. Find all external contours (RETR_LIST, CHAIN_APPROX_SIMPLE).
    5. Approximate each contour as a polygon (ε = 5% of perimeter).
    6. Keep 4-sided polygons with aspect ratio < 2.0 and area ≥ min_area.
    7. Return the polygon with the largest contour area.

    Returns
    -------
    bbox     : (x, y, w, h)  bounding rectangle
    centroid : (cx, cy)      pixel coordinates of the bounding-box centre
    """
    gray  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1.0)
    edges = cv2.Canny(blur, 30, 100)
    kern  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kern, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_box  = None
    best_area = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri   = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            ar = max(w, h) / (min(w, h) + 1e-9)
            if ar < 2.0 and area > best_area:
                best_area = area
                best_box  = (x, y, w, h)

    # Fallback — bounding box of the largest contour overall
    if best_box is None and cnts:
        c        = max(cnts, key=cv2.contourArea)
        best_box = cv2.boundingRect(c)

    if best_box is None:
        raise RuntimeError("No contour found in image.")

    x, y, w, h = best_box
    return best_box, (x + w // 2, y + h // 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7  —  DEPTH ESTIMATION   Z = f · B / d
# ═══════════════════════════════════════════════════════════════════════════════
def estimate_distance(cen_l:      tuple,
                      cen_r:      tuple,
                      f_px:       float,
                      baseline_cm: float) -> tuple:
    """
    Estimate depth using the stereo depth formula:

        Z = (f × B) / d

    where
        f   focal length in pixels
        B   baseline in the same units as the output distance
        d   horizontal disparity  =  | x_left − x_right |  (pixels)

    Parameters
    ----------
    cen_l, cen_r  : (cx, cy) centroids in the left and right images
    f_px          : focal length in pixels
    baseline_cm   : camera separation in centimetres

    Returns
    -------
    Z_cm      : estimated depth in centimetres
    Z_ft      : estimated depth in feet
    disparity : horizontal disparity in pixels
    """
    disparity = abs(float(cen_l[0]) - float(cen_r[0]))
    if disparity < 0.5:
        raise ValueError("Disparity ≈ 0 — cannot estimate depth.")
    Z_cm = (f_px * baseline_cm) / disparity
    Z_ft = Z_cm / 30.48
    return Z_cm, Z_ft, disparity


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8  —  VISUALISE  (side-by-side annotated plot)
# ═══════════════════════════════════════════════════════════════════════════════
def visualise(left_bgr:  np.ndarray,
              right_bgr: np.ndarray,
              box_l:     tuple,
              cen_l:     tuple,
              box_r:     tuple,
              cen_r:     tuple,
              pts1_in:   np.ndarray,
              pts2_in:   np.ndarray,
              Z_cm:      float,
              Z_ft:      float,
              disparity: float,
              out_path:  str) -> None:
    """
    Render a side-by-side figure of the stereo pair showing:
      • Cyan crosses  — SIFT inlier correspondences (sub-sampled)
      • Green rect    — detected square bounding box
      • Red dot       — bounding-box centroid
      • Yellow box    — distance annotation (left image only)

    The figure is saved as a PNG at `out_path`.
    """
    left_rgb  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.patch.set_facecolor("#12122a")

    img_data = [
        (axes[0], left_rgb,  box_l, cen_l, pts1_in, "Left Image  —  Detected Square"),
        (axes[1], right_rgb, box_r, cen_r, pts2_in, "Right Image  —  Detected Square"),
    ]

    for ax, img, box, cen, pts, title in img_data:
        ax.imshow(img)
        ax.set_title(title, fontsize=15, fontweight="bold",
                     color="white", pad=10)
        ax.set_facecolor("#12122a")
        ax.axis("off")

        # ── SIFT inlier keypoints (sub-sampled)
        step = max(1, len(pts) // 180)
        ax.scatter(pts[::step, 0], pts[::step, 1],
                   c="cyan", s=12, marker="+",
                   linewidths=0.8, alpha=0.45,
                   label="SIFT inliers" if ax is axes[0] else "_")

        # ── Square bounding box
        x, y, w, h = box
        rect = mpatches.FancyBboxPatch(
            (x - 4, y - 4), w + 8, h + 8,
            boxstyle="square,pad=0",
            linewidth=3.5, edgecolor="lime", facecolor="none",
            label="Detected square" if ax is axes[0] else "_"
        )
        ax.add_patch(rect)

        # ── Centroid
        ax.plot(*cen, "o", color="#ff3333", markersize=11,
                markeredgecolor="white", markeredgewidth=1.8,
                label="Centroid" if ax is axes[0] else "_",
                zorder=6)

    # ── Distance annotation — left image only
    err_cm  = abs(Z_cm - GT_CM)
    err_pct = err_cm / GT_CM * 100.0
    ann = (
        f"  Estimated   : {Z_ft:.2f} ft  ({Z_cm:.1f} cm)  \n"
        f"  Ground Truth: {GT_FT:.2f} ft  ({GT_CM:.1f} cm)  \n"
        f"  Error       : {err_cm:.1f} cm  ({err_pct:.1f}%)  \n"
        f"  Disparity   : {disparity:.1f} px  "
    )
    tx = min(cen_l[0] + 50,  IMG_W - 420)
    ty = max(cen_l[1] - 150, 15)
    axes[0].annotate(
        ann,
        xy     = cen_l,
        xytext = (tx, ty),
        fontsize     = 12,
        color        = "yellow",
        bbox         = dict(boxstyle="round,pad=0.55",
                            fc="#111111", alpha=0.90,
                            edgecolor="yellow", linewidth=1.8),
        arrowprops   = dict(arrowstyle="->", color="yellow", lw=2.2),
    )

    axes[0].legend(loc="upper left", fontsize=10,
                   facecolor="#22223a", labelcolor="white", framealpha=0.85)

    fig.suptitle(
        f"Uncalibrated Stereo Vision  ·  Samsung Galaxy M34  ·  "
        f"Baseline = {BASELINE_FT:.0f} ft  ·  f ≈ {F_PX:.0f} px",
        fontsize=15, fontweight="bold", color="white", y=1.01,
    )

    plt.tight_layout(pad=1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Plot saved  →  {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 9  —  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def print_summary(F: np.ndarray,
                  E: np.ndarray,
                  R: np.ndarray,
                  t: np.ndarray,
                  Z_cm: float,
                  Z_ft: float,
                  disparity: float,
                  n_matches: int,
                  n_inliers: int) -> None:
    """Pretty-print all computed matrices and the distance estimate."""
    np.set_printoptions(precision=8, suppress=True, linewidth=110)
    sep  = "─" * 72
    dsep = "═" * 72

    # ── Camera Intrinsics
    print(f"\n{dsep}")
    print("  CAMERA INTRINSIC MATRIX  K  (Samsung Galaxy M34 @ 1600×1200)")
    print(dsep)
    print(K)
    print(f"\n  Focal length (px)  : {F_PX:.4f}")
    print(f"  Principal point    : ({CX:.1f}, {CY:.1f})")
    print(f"  Sensor diagonal    : {SENSOR_D_MM:.4f} mm")
    print(f"  Crop factor        : {CROP_FACTOR:.4f}")
    print(f"  Actual FL          : {F_ACTUAL_MM:.4f} mm")

    # ── Feature matching
    print(f"\n{sep}")
    print("  FEATURE MATCHING  (SIFT + BFMatcher + Lowe ratio 0.75)")
    print(sep)
    print(f"  Lowe-ratio matches  : {n_matches}")
    print(f"  RANSAC inliers      : {n_inliers}")

    # ── Fundamental Matrix
    print(f"\n{dsep}")
    print("  FUNDAMENTAL MATRIX  F   [Normalised 8-Point + RANSAC]")
    print(dsep)
    print(F)
    print(f"\n  Epipolar constraint : x'ᵀ F x = 0  for all correspondences")
    print(f"  rank(F) = {np.linalg.matrix_rank(F)}  (must be 2)")

    # ── Essential Matrix
    print(f"\n{dsep}")
    print("  ESSENTIAL MATRIX  E = Kᵀ · F · K   [rank-2 enforced]")
    print(dsep)
    print(E)
    sv = np.linalg.svd(E, compute_uv=False)
    print(f"\n  Singular values of E : {sv[0]:.6f},  {sv[1]:.6f},  {sv[2]:.6f}")
    print(f"  σ₁ ≈ σ₂,  σ₃ ≈ 0  ✓" if sv[2] < 1e-6 else "  Rank-2 constraint active")

    # ── Rotation Matrix
    print(f"\n{dsep}")
    print("  ROTATION MATRIX  R   [SVD(E) decomposition + cheirality check]")
    print(dsep)
    print(R)
    theta = np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
    print(f"\n  det(R)              = {np.linalg.det(R):.8f}   (should be +1.0)")
    print(f"  RᵀR (≈ I₃)         =\n{np.array2string(R.T @ R, precision=8, suppress_small=True)}")
    print(f"  Rotation angle θ    = {theta:.4f}°")

    # ── Translation
    print(f"\n{sep}")
    print("  TRANSLATION VECTOR  t̂  [unit direction from SVD(E)]")
    print(sep)
    print(t)
    print(f"\n  ‖t̂‖               = {np.linalg.norm(t):.8f}   (should be ≈ 1.0)")

    # ── Distance
    err_cm  = abs(Z_cm - GT_CM)
    err_pct = err_cm / GT_CM * 100.0
    print(f"\n{dsep}")
    print("  DISTANCE ESTIMATION   Z = f · B / d")
    print(dsep)
    print(f"  Focal length      f  = {F_PX:.4f} px")
    print(f"  Baseline          B  = {BASELINE_CM:.4f} cm  ({BASELINE_FT:.0f} ft)")
    print(f"  Disparity         d  = {disparity:.4f} px")
    print(f"  Z = {F_PX:.4f} × {BASELINE_CM:.4f} / {disparity:.4f}")
    print(f"    = {Z_cm:.4f} cm")
    print(f"    = {Z_ft:.4f} ft")
    print(f"  Ground truth         = {GT_CM:.4f} cm  ({GT_FT:.2f} ft)")
    print(f"  Absolute error       = {err_cm:.4f} cm  ({err_pct:.2f}%)")
    print(f"{dsep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 10 —  SAVE RESULTS (for report generation)
# ═══════════════════════════════════════════════════════════════════════════════
def save_results(path: str,
                 F: np.ndarray, E: np.ndarray,
                 R: np.ndarray, t: np.ndarray,
                 box_l, cen_l, box_r, cen_r,
                 Z_cm: float, Z_ft: float,
                 disparity: float,
                 n_matches: int, n_inliers: int) -> None:
    """Serialise key results to JSON for the report generator."""

    def arr(a):
        return np.array(a).tolist()

    data = {
        "camera": {
            "sensor_w_mm":   SENSOR_W_MM,
            "sensor_h_mm":   SENSOR_H_MM,
            "sensor_d_mm":   float(SENSOR_D_MM),
            "crop_factor":   float(CROP_FACTOR),
            "f_actual_mm":   float(F_ACTUAL_MM),
            "f_px":          float(F_PX),
            "cx":            CX,
            "cy":            CY,
            "orig_w":        ORIG_W,
            "orig_h":        ORIG_H,
            "img_w":         IMG_W,
            "img_h":         IMG_H,
        },
        "stereo": {
            "baseline_ft":  BASELINE_FT,
            "baseline_cm":  BASELINE_CM,
            "gt_ft":        GT_FT,
            "gt_cm":        GT_CM,
        },
        "matrices": {
            "K": arr(K),
            "F": arr(F),
            "E": arr(E),
            "R": arr(R),
            "t": arr(t),
        },
        "detection": {
            "box_l":  list(box_l),
            "cen_l":  list(cen_l),
            "box_r":  list(box_r),
            "cen_r":  list(cen_r),
        },
        "estimation": {
            "disparity_px": disparity,
            "Z_cm":         Z_cm,
            "Z_ft":         Z_ft,
            "error_cm":     abs(Z_cm - GT_CM),
            "error_pct":    abs(Z_cm - GT_CM) / GT_CM * 100.0,
        },
        "matching": {
            "n_matches":  n_matches,
            "n_inliers":  n_inliers,
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Results saved  →  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    base       = os.path.dirname(os.path.abspath(__file__))
    left_path  = os.path.join(base, "left.jpeg")
    right_path = os.path.join(base, "right.jpeg")
    out_png    = os.path.join(base, "output_stereo.png")
    out_json   = os.path.join(base, "results.json")

    hdr = "═" * 72
    print(f"\n{hdr}")
    print("  Uncalibrated Stereo Vision — Distance Estimation")
    print(f"  Samsung Galaxy M34  |  Baseline = {BASELINE_FT:.0f} ft  |"
          f"  Target ≈ {GT_FT:.0f} ft")
    print(f"{hdr}")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1]  Loading images …")
    left_bgr, right_bgr = load_images(left_path, right_path)
    h, w = left_bgr.shape[:2]
    print(f"     Resolution : {w} × {h}")

    left_gray  = cv2.cvtColor(left_bgr,  cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)

    # ── 2. Feature matching ──────────────────────────────────────────────────
    print("\n[2]  Detecting SIFT features and matching …")
    pts1, pts2, kp1, kp2, good = detect_and_match(left_gray, right_gray)
    print(f"     Lowe-ratio matches : {len(pts1)}")

    # ── 3. Fundamental Matrix ────────────────────────────────────────────────
    print("\n[3]  Computing Fundamental Matrix F (RANSAC) …")
    F, mask = compute_fundamental(pts1, pts2)
    pts1_in = pts1[mask]
    pts2_in = pts2[mask]
    print(f"     RANSAC inliers : {int(mask.sum())}")

    # ── 4. Essential Matrix ──────────────────────────────────────────────────
    print("\n[4]  Computing Essential Matrix E …")
    E = compute_essential(F, K)

    # ── 5. Rotation + Translation ────────────────────────────────────────────
    print("\n[5]  Decomposing E  →  R, t …")
    R, t, pts3d = decompose_essential(E, K, pts1_in, pts2_in)

    # ── 6. Square detection ──────────────────────────────────────────────────
    print("\n[6]  Detecting square object …")
    box_l, cen_l = detect_square(left_bgr)
    box_r, cen_r = detect_square(right_bgr)
    print(f"     Left   centroid : {cen_l}   bbox : {box_l}")
    print(f"     Right  centroid : {cen_r}   bbox : {box_r}")

    # ── 7. Depth estimation ──────────────────────────────────────────────────
    print("\n[7]  Estimating depth …")
    Z_cm, Z_ft, disparity = estimate_distance(cen_l, cen_r, F_PX, BASELINE_CM)

    # ── 8. Console summary ───────────────────────────────────────────────────
    print_summary(F, E, R, t, Z_cm, Z_ft, disparity,
                  n_matches=len(pts1), n_inliers=int(mask.sum()))

    # ── 9. Plot ──────────────────────────────────────────────────────────────
    print("[8]  Rendering annotated stereo plot …")
    visualise(left_bgr, right_bgr,
              box_l, cen_l, box_r, cen_r,
              pts1_in, pts2_in,
              Z_cm, Z_ft, disparity,
              out_path=out_png)

    # ── 10. Save JSON ─────────────────────────────────────────────────────────
    print("[9]  Saving results for report generator …")
    save_results(out_json, F, E, R, t,
                 box_l, cen_l, box_r, cen_r,
                 Z_cm, Z_ft, disparity,
                 n_matches=len(pts1), n_inliers=int(mask.sum()))

    print(f"\n[✓]  Pipeline complete.\n"
          f"     Stereo plot : {out_png}\n"
          f"     Results     : {out_json}\n")


if __name__ == "__main__":
    main()
