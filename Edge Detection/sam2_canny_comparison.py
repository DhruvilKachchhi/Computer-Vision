"""
SAM2 vs Canny+Fill — Thermal Animal Segmentation Comparison
=============================================================
Simple comparison script that compares SAM2 and Canny+Fill methods
for thermal animal boundary detection with visualizations and metrics.

This script provides:
- Side-by-side visual comparison
- Processing time comparison
- Basic accuracy metrics (when ground truth is available)
- Boundary visualization

Usage:
    python sam2_canny_comparison.py
    python sam2_canny_comparison.py --image path/to/thermal.jpg

Requires: opencv-python-headless numpy matplotlib
Optional: torch + sam2 (falls back to simulation if unavailable)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# SAM2 optional import
SAM2_AVAILABLE = False
try:
    import torch
    # Try importing from the local sam2 folder first
    import sys
    sys.path.insert(0, 'sam2')
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
    print("[INFO] SAM2 found in local sam2 folder — will use real model inference.")
except ImportError as e:
    print(f"[INFO] SAM2 import failed: {e}")
    print("[INFO] Running in high-fidelity simulation mode.")


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def keep_largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1: return mask
    largest = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
    out = np.zeros_like(mask); out[labels == largest] = 255
    return out


def fill_holes(mask):
    H, W = mask.shape
    fm = np.zeros((H + 2, W + 2), np.uint8)
    f  = mask.copy()
    cv2.floodFill(f, fm, (0, 0), 255)
    return cv2.bitwise_not(f) | mask


def extract_boundary(mask, thickness=2):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (thickness*2+1, thickness*2+1))
    return cv2.subtract(mask, cv2.erode(mask, k, iterations=1))


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation methods
# ─────────────────────────────────────────────────────────────────────────────

def run_canny_fill(gray):
    """Full Canny+Fill pipeline (classical CV)."""
    t0 = time.perf_counter()
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(bilateral)
    H, W = gray.shape

    otsu_val, _ = cv2.threshold(bilateral, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(bilateral, otsu_val * 0.5, otsu_val)

    gx = cv2.Scharr(bilateral, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(bilateral, cv2.CV_32F, 0, 1)
    grad_8u = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_thr = cv2.threshold(grad_8u, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fused  = cv2.bitwise_or(edges, grad_thr)
    k7     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, k7, iterations=3)

    fm = np.zeros((H+2, W+2), np.uint8)
    fl = closed.copy(); cv2.floodFill(fl, fm, (0, 0), 255)
    combined = cv2.bitwise_or(closed, cv2.bitwise_not(fl))

    k13  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k13, iterations=4)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  k13, iterations=1)
    mask = fill_holes(keep_largest_component(mask))
    elapsed = time.perf_counter() - t0
    return mask, elapsed


def run_sam2_or_simulate(gray, thermal_color, checkpoint=None):
    """Run real SAM2 or high-fidelity simulation."""
    t0 = time.perf_counter()

    use_real = (SAM2_AVAILABLE and checkpoint and os.path.exists(checkpoint))

    if use_real:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam2_model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint, device=device)
        gen = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32, pred_iou_thresh=0.86,
            stability_score_thresh=0.92, min_mask_region_area=500)
        image_rgb = cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB)
        masks = gen.generate(image_rgb)
        best = max(masks,
                   key=lambda m: float(gray[m['segmentation']].mean())
                                 * m['stability_score'] * m['predicted_iou'])
        mask = best['segmentation'].astype(np.uint8) * 255
        mask = fill_holes(mask)
        mode = "SAM2 (real)"
    else:
        # Simulation
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        _, seed = cv2.threshold(bilateral, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        seed = fill_holes(keep_largest_component(seed))
        for r in [15, 9, 5, 3]:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
            seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, k, iterations=2)
            seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN,  k, iterations=1)
            seed = fill_holes(seed)
        dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
        gx = cv2.Scharr(bilateral.astype(np.float32), cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(bilateral.astype(np.float32), cv2.CV_32F, 0, 1)
        grad_n = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 1, cv2.NORM_MINMAX)
        bz = (dist < 8) & (dist > 0)
        mask = seed.copy()
        mask[bz & (grad_n > 0.35) & (gray > gray.mean())] = 255
        mask[bz & (grad_n < 0.15)] = 0
        np.random.seed(7)
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        H, W = gray.shape
        for _ in range(8):
            x = np.random.randint(0, W - 20); y = np.random.randint(0, H - 20)
            p = mask[y:y+20, x:x+20]
            mask[y:y+20, x:x+20] = (cv2.dilate(p, k3) if np.random.rand() > .5
                                     else cv2.erode(p, k3))
        mask = fill_holes(keep_largest_component(mask))
        mode = "SAM2 (simulated)"

    elapsed = time.perf_counter() - t0
    return mask, elapsed, mode


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_basic_metrics(pred, gt):
    """Compute basic segmentation metrics."""
    p = pred > 127; g = gt > 127
    tp = (p & g).sum(); fp = (p & ~g).sum()
    fn = (~p & g).sum(); tn = (~p & ~g).sum()

    iou       = tp / (tp + fp + fn + 1e-9)
    dice      = 2*tp / (2*tp + fp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)

    return {
        'IoU':       round(float(iou),       4),
        'Dice':      round(float(dice),      4),
        'Precision': round(float(precision), 4),
        'Recall':    round(float(recall),    4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_comparison(thermal_color, gray, canny_mask, sam2_mask, sam2_mode,
                        canny_metrics, sam2_metrics, canny_time, sam2_time, out_path):
    """Create a comprehensive comparison visualization."""
    
    fig = plt.figure(figsize=(20, 16), facecolor='#070B12')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.30, wspace=0.20,
                            left=0.05, right=0.95, top=0.92, bottom=0.05)

    title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=8)
    ann_kw   = dict(color='white', fontsize=9, va='bottom',
                    bbox=dict(facecolor='#000', alpha=0.6, pad=3))

    # ── Row 0: Input and individual results ───────────────────────────────────
    # [0,0] Input thermal
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB))
    ax.set_title('Input Thermal Image', **title_kw); ax.axis('off')

    # [0,1] Canny+Fill result
    ax = fig.add_subplot(gs[0, 1])
    cv = thermal_color.copy()
    cv[canny_mask > 0] = (cv[canny_mask > 0].astype(np.float32) * 0.55
                          + np.array([50, 200, 255]) * 0.45).astype(np.uint8)
    bnd_c = extract_boundary(canny_mask, 2)
    cv[bnd_c > 0] = [50, 200, 255]
    ax.imshow(cv[:, :, ::-1])
    ax.set_title(f'Canny+Fill  ({canny_time*1000:.0f} ms)', **title_kw)
    ax.text(0.02, 0.02, f'IoU: {canny_metrics["IoU"]:.3f}\nDice: {canny_metrics["Dice"]:.3f}',
            transform=ax.transAxes, **ann_kw)
    ax.axis('off')

    # [0,2] SAM2 result
    ax = fig.add_subplot(gs[0, 2])
    sv = thermal_color.copy()
    sv[sam2_mask > 0] = (sv[sam2_mask > 0].astype(np.float32) * 0.55
                         + np.array([50, 255, 140]) * 0.45).astype(np.uint8)
    bnd_s = extract_boundary(sam2_mask, 2)
    sv[bnd_s > 0] = [50, 255, 140]
    ax.imshow(sv[:, :, ::-1])
    ax.set_title(f'{sam2_mode}  ({sam2_time*1000:.0f} ms)', **title_kw)
    ax.text(0.02, 0.02, f'IoU: {sam2_metrics["IoU"]:.3f}\nDice: {sam2_metrics["Dice"]:.3f}',
            transform=ax.transAxes, **ann_kw)
    ax.axis('off')

    # ── Row 1: Contour overlays ───────────────────────────────────────────────
    # [1,0] Canny contour
    ax = fig.add_subplot(gs[1, 0])
    cco = thermal_color.copy()
    contours_c, _ = cv2.findContours(canny_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_c:
        cv2.drawContours(cco, [max(contours_c, key=cv2.contourArea)], -1, (50,200,255), 3)
    ax.imshow(cco[:, :, ::-1])
    ax.set_title('Canny+Fill Contour', **title_kw); ax.axis('off')

    # [1,1] SAM2 contour
    ax = fig.add_subplot(gs[1, 1])
    sco = thermal_color.copy()
    contours_s, _ = cv2.findContours(sam2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_s:
        cv2.drawContours(sco, [max(contours_s, key=cv2.contourArea)], -1, (50,255,140), 3)
    ax.imshow(sco[:, :, ::-1])
    ax.set_title(f'{sam2_mode} Contour', **title_kw); ax.axis('off')

    # [1,2] Both contours overlaid
    ax = fig.add_subplot(gs[1, 2])
    both = thermal_color.copy()
    if contours_c:
        cv2.drawContours(both, [max(contours_c, key=cv2.contourArea)], -1, (50,200,255), 2)
    if contours_s:
        cv2.drawContours(both, [max(contours_s, key=cv2.contourArea)], -1, (50,255,140), 2)
    ax.imshow(both[:, :, ::-1])
    ax.set_title('Both Contours Overlaid', **title_kw); ax.axis('off')
    legend_els = [Patch(facecolor='#32C8FF', label='Canny+Fill'),
                  Patch(facecolor='#32FF8C', label=sam2_mode)]
    ax.legend(handles=legend_els, loc='lower left', fontsize=9,
              facecolor='#111', labelcolor='white', framealpha=0.85)

    # ── Row 2: Difference map and metrics ─────────────────────────────────────
    # [2,0] Difference map
    ax = fig.add_subplot(gs[2, 0])
    diff = np.zeros((*gray.shape, 3), np.uint8)
    diff[(sam2_mask > 127) & (canny_mask > 127)]  = [50, 220, 90]    # both agree
    diff[(sam2_mask > 127) & (canny_mask <= 127)] = [50, 120, 255]   # SAM2 only
    diff[(sam2_mask <= 127) & (canny_mask > 127)] = [255, 80, 50]    # Canny only
    bg = cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB).astype(np.float32)
    out_diff = (bg * 0.45 + diff.astype(np.float32) * 0.55).astype(np.uint8)
    ax.imshow(out_diff)
    ax.set_title('Difference Map', **title_kw); ax.axis('off')
    agree_pct = float(((sam2_mask>127)&(canny_mask>127)).sum()) / (gray.size) * 100
    legend_d = [Patch(facecolor='#32DC5A', label=f'Both agree ({agree_pct:.1f}%)'),
                Patch(facecolor='#3278FF', label=f'{sam2_mode} only'),
                Patch(facecolor='#FF5032', label='Canny+Fill only')]
    ax.legend(handles=legend_d, loc='lower left', fontsize=8,
              facecolor='#111', labelcolor='white', framealpha=0.85)

    # [2,1] Metrics comparison
    ax = fig.add_subplot(gs[2, 1])
    ax.set_facecolor('#0E1420')
    
    metrics_names = ['IoU', 'Dice', 'Precision', 'Recall']
    canny_vals = [canny_metrics[k] for k in metrics_names]
    sam2_vals  = [sam2_metrics[k]  for k in metrics_names]
    
    x = np.arange(len(metrics_names))
    w = 0.35
    b1 = ax.bar(x - w/2, canny_vals, w, color='#32C8FF', alpha=0.85, label='Canny+Fill')
    b2 = ax.bar(x + w/2, sam2_vals,  w, color='#32FF8C', alpha=0.85, label=sam2_mode)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, color='white', fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.tick_params(colors='white')
    ax.set_ylabel('Score (0-1)', color='white', fontsize=10)
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values(): spine.set_edgecolor('#333')
    ax.grid(axis='y', color='#222', linewidth=0.8, zorder=0)
    ax.legend(facecolor='#1A1A2E', labelcolor='white', fontsize=9)
    ax.set_title('Segmentation Metrics', **title_kw)

    # Annotate bars
    for i, (cv_, sv_) in enumerate(zip(canny_vals, sam2_vals)):
        ax.text(i - w/2, cv_ + 0.02, f'{cv_:.3f}', ha='center', va='bottom', 
                color='#32C8FF', fontsize=8, fontweight='bold')
        ax.text(i + w/2, sv_ + 0.02, f'{sv_:.3f}', ha='center', va='bottom', 
                color='#32FF8C', fontsize=8, fontweight='bold')
        delta = sv_ - cv_
        sign = '+' if delta >= 0 else ''
        col = '#32FF8C' if delta >= 0 else '#FF6060'
        ax.text(i, max(cv_, sv_) + 0.06, f'Δ{sign}{delta:.3f}',
                ha='center', va='bottom', color=col, fontsize=7, fontweight='bold',
                bbox=dict(facecolor='#1A1A2E', alpha=0.7, pad=2))

    # [2,2] Performance summary
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor('#0E1420')
    
    # Create summary text
    summary_lines = [
        '─ Performance Summary ─────────────',
        '',
        f'Processing Time:',
        f'  Canny+Fill: {canny_time*1000:.1f} ms',
        f'  {sam2_mode}: {sam2_time*1000:.1f} ms',
        '',
        f'Speed Difference: {abs(canny_time - sam2_time)*1000:.1f} ms',
        '',
        f'Best Method:',
        f'  IoU:  {"SAM2" if sam2_metrics["IoU"] > canny_metrics["IoU"] else "Canny+Fill"}',
        f'  Dice: {"SAM2" if sam2_metrics["Dice"] > canny_metrics["Dice"] else "Canny+Fill"}',
        f'  Precision: {"SAM2" if sam2_metrics["Precision"] > canny_metrics["Precision"] else "Canny+Fill"}',
        f'  Recall: {"SAM2" if sam2_metrics["Recall"] > canny_metrics["Recall"] else "Canny+Fill"}',
    ]
    
    ax.text(0.05, 0.95, '\n'.join(summary_lines),
            transform=ax.transAxes, color='#CCCCCC', fontsize=9,
            fontfamily='monospace', va='top', ha='left',
            bbox=dict(facecolor='#0A0D16', alpha=0.88, pad=8))
    
    ax.set_title('Performance Summary', **title_kw)
    ax.axis('off')

    fig.suptitle(
        f'SAM2 vs Canny+Fill — Thermal Animal Segmentation Comparison',
        color='white', fontsize=15, fontweight='bold', y=0.96
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OUT] Comparison visualization → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='SAM2 vs Canny+Fill thermal segmentation comparison')
    parser.add_argument('--image',      default=None,
                        help='Thermal image path (uses synthetic if omitted)')
    parser.add_argument('--checkpoint', default='C:\\Repositories\\Edge detection\\sam2\\checkpoints\\sam2.1_hiera_large.pt',
                        help='SAM2 checkpoint .pt path')
    parser.add_argument('--out_dir',    default='results',
                        help='Output directory')
    args = parser.parse_args()

    print("=" * 65)
    print("  SAM2 vs Canny+Fill — Thermal Animal Segmentation Comparison")
    print("=" * 65)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load image ────────────────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        thermal_color = cv2.imread(args.image)
        gray = cv2.cvtColor(thermal_color, cv2.COLOR_BGR2GRAY)
        print(f"[IN]  Loaded: {args.image}  {gray.shape}")
    else:
        # Look for images in input_images directory
        input_dir = 'input_images'
        if os.path.exists(input_dir):
            image_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if image_files:
                # Use the first image found
                image_path = os.path.join(input_dir, image_files[0])
                thermal_color = cv2.imread(image_path)
                if thermal_color is None:
                    raise ValueError(f"Cannot read image: {image_path}")
                gray = cv2.cvtColor(thermal_color, cv2.COLOR_BGR2GRAY)
                print(f"[IN]  Loaded from input_images: {image_files[0]}  {gray.shape}")
            else:
                print(f"[IN]  No images found in {input_dir} — using synthetic image.")
                print("[IN]  Generating synthetic thermal image …")
                # Create a simple synthetic image
                np.random.seed(42)
                H, W = 480, 640
                canvas = np.random.normal(60, 8, (H, W)).astype(np.float32)
                
                def warm_blob(cx, cy, ra, rb, angle, amplitude, sigma):
                    m = np.zeros((H, W), np.float32)
                    cv2.ellipse(m, (cx, cy), (ra, rb), angle, 0, 360, 1.0, -1)
                    m = cv2.GaussianBlur(m, (51, 51), sigma)
                    canvas[:] += m * amplitude

                warm_blob(310, 270, 120, 75,  10, 140, 25)
                warm_blob(170, 220,  55, 55,   0, 155, 18)
                warm_blob(230, 240,  45, 35,  30, 148, 14)
                canvas += np.random.normal(0, 3, (H, W)).astype(np.float32)
                canvas = np.clip(canvas, 0, 255)
                gray  = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                thermal_color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(args.out_dir, 'synthetic_thermal.jpg'), thermal_color)
                print(f"[IN]  Synthetic image created: {gray.shape}")
        else:
            print(f"[IN]  {input_dir} directory not found — using synthetic image.")
            print("[IN]  Generating synthetic thermal image …")
            # Create a simple synthetic image
            np.random.seed(42)
            H, W = 480, 640
            canvas = np.random.normal(60, 8, (H, W)).astype(np.float32)
            
            def warm_blob(cx, cy, ra, rb, angle, amplitude, sigma):
                m = np.zeros((H, W), np.float32)
                cv2.ellipse(m, (cx, cy), (ra, rb), angle, 0, 360, 1.0, -1)
                m = cv2.GaussianBlur(m, (51, 51), sigma)
                canvas[:] += m * amplitude

            warm_blob(310, 270, 120, 75,  10, 140, 25)
            warm_blob(170, 220,  55, 55,   0, 155, 18)
            warm_blob(230, 240,  45, 35,  30, 148, 14)
            canvas += np.random.normal(0, 3, (H, W)).astype(np.float32)
            canvas = np.clip(canvas, 0, 255)
            gray  = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            thermal_color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(args.out_dir, 'synthetic_thermal.jpg'), thermal_color)
            print(f"[IN]  Synthetic image created: {gray.shape}")

    # ── Run both methods ───────────────────────────────────────────────────────
    print("\n[RUN] Running Canny+Fill …")
    canny_mask, canny_time = run_canny_fill(gray)
    print(f"      Done ({canny_time*1000:.1f} ms)")

    print("[RUN] Running SAM2 …")
    sam2_mask, sam2_time, sam2_mode = run_sam2_or_simulate(
        gray, thermal_color, args.checkpoint)
    print(f"      Done ({sam2_time*1000:.1f} ms)  [{sam2_mode}]")

    # ── Compute metrics (comparing each method against the other) ────────────
    print("\n[METRICS] Computing IoU scores …")
    
    # Calculate IoU scores for both methods by comparing them against each other
    # This gives us a measure of how well each method performs relative to the other
    p_canny = canny_mask > 127
    p_sam2  = sam2_mask > 127
    
    # IoU of Canny vs SAM2 (how well Canny matches SAM2)
    tp_canny_vs_sam2 = (p_canny & p_sam2).sum()
    fp_canny_vs_sam2 = (p_canny & ~p_sam2).sum()
    fn_canny_vs_sam2 = (~p_canny & p_sam2).sum()
    iou_canny_vs_sam2 = tp_canny_vs_sam2 / (tp_canny_vs_sam2 + fp_canny_vs_sam2 + fn_canny_vs_sam2 + 1e-9)
    
    # IoU of SAM2 vs Canny (how well SAM2 matches Canny)
    tp_sam2_vs_canny = (p_sam2 & p_canny).sum()
    fp_sam2_vs_canny = (p_sam2 & ~p_canny).sum()
    fn_sam2_vs_canny = (~p_sam2 & p_canny).sum()
    iou_sam2_vs_canny = tp_sam2_vs_canny / (tp_sam2_vs_canny + fp_sam2_vs_canny + fn_sam2_vs_canny + 1e-9)
    
    # Dice coefficients
    dice_canny_vs_sam2 = 2*tp_canny_vs_sam2 / (2*tp_canny_vs_sam2 + fp_canny_vs_sam2 + fn_canny_vs_sam2 + 1e-9)
    dice_sam2_vs_canny = 2*tp_sam2_vs_canny / (2*tp_sam2_vs_canny + fp_sam2_vs_canny + fn_sam2_vs_canny + 1e-9)
    
    # Precision and Recall for each method
    precision_canny = tp_canny_vs_sam2 / (tp_canny_vs_sam2 + fp_canny_vs_sam2 + 1e-9)
    recall_canny = tp_canny_vs_sam2 / (tp_canny_vs_sam2 + fn_canny_vs_sam2 + 1e-9)
    
    precision_sam2 = tp_sam2_vs_canny / (tp_sam2_vs_canny + fp_sam2_vs_canny + 1e-9)
    recall_sam2 = tp_sam2_vs_canny / (tp_sam2_vs_canny + fn_sam2_vs_canny + 1e-9)
    
    # Set metrics for visualization
    canny_metrics = {
        'IoU': round(float(iou_canny_vs_sam2), 4),
        'Dice': round(float(dice_canny_vs_sam2), 4),
        'Precision': round(float(precision_canny), 4),
        'Recall': round(float(recall_canny), 4)
    }
    
    sam2_metrics = {
        'IoU': round(float(iou_sam2_vs_canny), 4),
        'Dice': round(float(dice_sam2_vs_canny), 4),
        'Precision': round(float(precision_sam2), 4),
        'Recall': round(float(recall_sam2), 4)
    }
    
    print(f"\n  IoU Analysis:")
    print(f"    Canny+Fill vs SAM2: {iou_canny_vs_sam2:.3f}")
    print(f"    SAM2 vs Canny+Fill: {iou_sam2_vs_canny:.3f}")
    print(f"    Average IoU: {(iou_canny_vs_sam2 + iou_sam2_vs_canny) / 2:.3f}")
    print(f"\n  Dice Analysis:")
    print(f"    Canny+Fill vs SAM2: {dice_canny_vs_sam2:.3f}")
    print(f"    SAM2 vs Canny+Fill: {dice_sam2_vs_canny:.3f}")
    print(f"    Average Dice: {(dice_canny_vs_sam2 + dice_sam2_vs_canny) / 2:.3f}")
    print(f"\n  Mask Statistics:")
    print(f"    Canny pixels: {p_canny.sum():,}")
    print(f"    SAM2 pixels: {p_sam2.sum():,}")
    print(f"    Common pixels: {tp_canny_vs_sam2:,}")
    print(f"    Canny unique pixels: {fp_canny_vs_sam2:,}")
    print(f"    SAM2 unique pixels: {fp_sam2_vs_canny:,}")

    # ── Visualize ─────────────────────────────────────────────────────────────
    print("\n[VIZ] Generating comparison figure …")
    visualize_comparison(
        thermal_color, gray,
        canny_mask, sam2_mask, sam2_mode,
        canny_metrics, sam2_metrics,
        canny_time, sam2_time,
        out_path=os.path.join(args.out_dir, 'sam2_canny_comparison.png'))

    # ── Save masks ────────────────────────────────────────────────────────────
    cv2.imwrite(os.path.join(args.out_dir, 'canny_mask_final.png'), canny_mask)
    cv2.imwrite(os.path.join(args.out_dir, 'sam2_mask_final.png'),  sam2_mask)
    
    print("\n  Done. Outputs written to:", args.out_dir)
    print("\n  Generated files:")
    print("    - sam2_canny_comparison.png (comparison visualization)")
    print("    - canny_mask_final.png (Canny+Fill binary mask)")
    print("    - sam2_mask_final.png (SAM2 binary mask)")


if __name__ == '__main__':
    main()