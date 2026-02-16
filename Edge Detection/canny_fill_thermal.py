"""
Canny + Fill: Thermal Animal Boundary Detection
================================================
Finds the exact boundary of an animal in a thermal infrared image using
classical computer vision only (no ML/DL).

Pipeline:
  1. Bilateral filter         — edge-preserving denoising
  2. CLAHE                    — contrast enhancement
  3. Auto Canny (Otsu-based)  — edge detection
  4. Scharr gradient          — additional edge strength
  5. Edge fusion + closing    — bridge gaps between edge fragments
  6. Flood fill + invert      — convert edge map → filled region
  7. Morphological cleanup    — remove noise, fill holes
  8. Largest component        — keep only the animal

Usage:
    python canny_fill_thermal.py                        # uses built-in synthetic image
    python canny_fill_thermal.py --image your_file.jpg  # use your own thermal image

Outputs:
    canny_fill_edge_detection.png   — 9-panel step-by-step visualization
    canny_fill_mask.png             — binary segmentation mask
    canny_fill_overlay.png          — contour overlay on original image

Requirements:
    pip install opencv-python-headless numpy scipy matplotlib
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import ndimage
import os
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic thermal image generator (used when no real image is provided)
# ─────────────────────────────────────────────────────────────────────────────

def create_synthetic_thermal(save_path="thermal_animal.jpg"):
    """
    Generate a realistic synthetic thermal image of a dog.
    Simulates FLIR-style Inferno false-color output where brighter = warmer.
    """
    np.random.seed(42)
    H, W = 480, 640
    canvas = np.random.normal(60, 8, (H, W)).astype(np.float32)

    def warm_blob(cx, cy, ra, rb, angle_deg, amplitude, sigma):
        mask = np.zeros((H, W), np.float32)
        cv2.ellipse(mask, (cx, cy), (ra, rb), angle_deg, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), sigma)
        canvas[:] += mask * amplitude

    warm_blob(310, 270, 120, 75,  10, 140, 25)   # torso
    warm_blob(170, 220,  55, 55,   0, 155, 18)   # head
    warm_blob(230, 240,  45, 35,  30, 148, 14)   # neck
    warm_blob(250, 360,  18, 55,   5, 125, 10)   # leg FL
    warm_blob(290, 370,  18, 55,   0, 125, 10)   # leg FR
    warm_blob(360, 355,  18, 55,  -5, 125, 10)   # leg RL
    warm_blob(390, 365,  18, 55,   0, 125, 10)   # leg RR
    warm_blob(440, 220,  60, 22, -30, 115, 12)   # tail
    warm_blob(155, 175,  22, 30,  15, 160,  8)   # ear
    warm_blob(118, 228,  15, 12,   0, 170,  6)   # nose (hottest)
    warm_blob(140, 228,  30, 20,   0, 145,  8)   # snout
    canvas += np.random.normal(0, 3, (H, W)).astype(np.float32)
    canvas = np.clip(canvas, 0, 255)

    gray  = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    cv2.imwrite(save_path, color)
    print(f"  Synthetic thermal image created → {save_path}")
    return gray, color


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def keep_largest_component(mask):
    """Retain only the largest connected foreground region."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest_label = stats[1:, cv2.CC_STAT_AREA].argmax() + 1
    result = np.zeros_like(mask)
    result[labels == largest_label] = 255
    return result


def fill_holes(mask):
    """Fill enclosed holes in a binary mask via flood fill from image border."""
    H, W = mask.shape
    flood_map = np.zeros((H + 2, W + 2), np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood_map, (0, 0), 255)
    return cv2.bitwise_not(filled) | mask


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(gray):
    """
    Bilateral filter: smooths homogeneous regions while keeping
    warm/cool boundaries sharp — essential for thermal images.
    CLAHE: boosts local contrast for low-emissivity regions.
    """
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(bilateral)
    return bilateral, enhanced


# ─────────────────────────────────────────────────────────────────────────────
# Canny + Fill segmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def canny_fill_segmentation(gray, bilateral, enhanced):
    """
    Full Canny + Fill pipeline returning intermediate stages for visualization.

    Returns
    -------
    mask         : final binary segmentation mask  (uint8, 0/255)
    edges_canny  : raw Canny edge map
    grad_8u      : Scharr gradient magnitude (8-bit)
    grad_thresh  : thresholded gradient edges
    fused_edges  : Canny ∪ gradient edges
    closed_edges : fused edges after morphological closing
    contour      : final boundary contour (list of points)
    otsu_val     : Otsu threshold used for Canny
    """
    H, W = bilateral.shape

    # ── Step 1: Auto Canny — thresholds derived from Otsu ────────────────────
    # Otsu's threshold is a good proxy for the high Canny threshold.
    otsu_val, _ = cv2.threshold(bilateral, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges_canny = cv2.Canny(bilateral,
                            threshold1=otsu_val * 0.5,
                            threshold2=otsu_val)

    # ── Step 2: Scharr gradient — better directional accuracy than Sobel ─────
    gx = cv2.Scharr(bilateral, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(bilateral, cv2.CV_32F, 0, 1)
    grad_mag  = cv2.magnitude(gx, gy)
    grad_8u   = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, grad_thresh = cv2.threshold(grad_8u, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ── Step 3: Fuse both edge sources ────────────────────────────────────────
    fused_edges = cv2.bitwise_or(edges_canny, grad_thresh)

    # ── Step 4: Morphological closing to bridge edge gaps ─────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_edges = cv2.morphologyEx(fused_edges, cv2.MORPH_CLOSE,
                                    kernel_close, iterations=3)

    # ── Step 5: Flood fill background, then invert → animal region ───────────
    flood_map = np.zeros((H + 2, W + 2), np.uint8)
    filled    = closed_edges.copy()
    cv2.floodFill(filled, flood_map, (0, 0), 255)   # fill background white
    filled_inv = cv2.bitwise_not(filled)             # invert → animal = white
    combined   = cv2.bitwise_or(closed_edges, filled_inv)

    # ── Step 6: Morphological cleanup ─────────────────────────────────────────
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_clean, iterations=4)
    mask = cv2.morphologyEx(mask,     cv2.MORPH_OPEN,  kernel_clean, iterations=1)

    # ── Step 7: Keep largest component + fill interior holes ──────────────────
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)

    # ── Step 8: Extract final boundary contour ────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    return (mask, edges_canny, grad_8u, grad_thresh,
            fused_edges, closed_edges, contour, otsu_val)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pipeline(thermal_color, gray, bilateral,
                       mask, edges_canny, grad_8u, grad_thresh,
                       fused_edges, closed_edges, contour, otsu_val,
                       out_path):
    """
    9-panel step-by-step visualization of the complete Canny+Fill pipeline.
    """
    H, W = gray.shape
    fig = plt.figure(figsize=(24, 20), facecolor='#0A0A14')
    gs  = gridspec.GridSpec(3, 3, figure=fig,
                            hspace=0.38, wspace=0.22,
                            left=0.04, right=0.96, top=0.92, bottom=0.04)
    title_kw = dict(color='white', fontsize=12, fontweight='bold', pad=8)
    ann_kw   = dict(color='white', fontsize=9, va='bottom',
                    bbox=dict(facecolor='black', alpha=0.55, pad=3))

    # ── ① Original thermal ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(cv2.cvtColor(thermal_color, cv2.COLOR_BGR2RGB))
    ax.set_title('① Input Thermal Image\n(Inferno false-color LUT)', **title_kw)
    ax.axis('off')

    # ── ② Bilateral filtered ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(cv2.applyColorMap(bilateral, cv2.COLORMAP_INFERNO)[:, :, ::-1])
    ax.set_title('② Bilateral Filtered\n(edge-preserving denoising)', **title_kw)
    ax.axis('off')

    # ── ③ Canny edges ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    canny_rgb = np.zeros((*edges_canny.shape, 3), np.uint8)
    canny_rgb[edges_canny > 0] = [255, 220, 50]          # yellow glow
    ax.imshow(canny_rgb)
    ax.set_title(f'③ Canny Edge Detection\n(thresholds: {otsu_val*0.5:.0f} / {otsu_val:.0f})',
                 **title_kw)
    ax.axis('off')
    n_pix = int((edges_canny > 0).sum())
    ax.text(0.02, 0.02, f'{n_pix:,} edge pixels', transform=ax.transAxes, **ann_kw)

    # ── ④ Scharr gradient magnitude ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(grad_8u, cmap='hot', vmin=0, vmax=255)
    ax.set_title('④ Scharr Gradient Magnitude\n(|∇I| edge strength map)', **title_kw)
    ax.axis('off')
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.yaxis.set_tick_params(color='white', labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # ── ⑤ Fused + closed edges ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    fused_rgb = np.zeros((*closed_edges.shape, 3), np.uint8)
    fused_rgb[edges_canny  > 0] = [255, 220,  50]       # Canny  → yellow
    fused_rgb[grad_thresh  > 0] = [100, 200, 255]       # Scharr → blue
    fused_rgb[closed_edges > 0] = [255, 255, 255]       # merged → white
    ax.imshow(fused_rgb)
    ax.set_title('⑤ Fused + Morphologically Closed\n(Canny ∪ Gradient, gaps filled)',
                 **title_kw)
    ax.axis('off')
    legend_els = [Patch(facecolor='#FFDC32', label='Canny'),
                  Patch(facecolor='#64C8FF', label='Gradient'),
                  Patch(facecolor='white',   label='Merged/Closed')]
    ax.legend(handles=legend_els, loc='lower left', fontsize=8,
              facecolor='#111', labelcolor='white', framealpha=0.8)

    # ── ⑥ Flood-filled mask ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(cv2.applyColorMap(mask, cv2.COLORMAP_PLASMA)[:, :, ::-1])
    ax.set_title('⑥ Flood-Fill + Morphology\n(edge map → filled region mask)',
                 **title_kw)
    ax.axis('off')
    area_pct = float(mask.sum()) / 255 / (H * W) * 100
    ax.text(0.02, 0.02, f'Animal area: {area_pct:.1f}% of frame',
            transform=ax.transAxes, **ann_kw)

    # ── ⑦ Final boundary on thermal ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    bnd_overlay = thermal_color.copy()
    if contour is not None:
        cv2.drawContours(bnd_overlay, [contour], -1, (0, 255, 100), 3)
    ax.imshow(cv2.cvtColor(bnd_overlay, cv2.COLOR_BGR2RGB))
    ax.set_title('⑦ Final Detected Boundary\n(contour on thermal image)', **title_kw)
    ax.axis('off')
    if contour is not None:
        ax.text(0.02, 0.02, f'{len(contour):,} contour points',
                transform=ax.transAxes, color='#00FF64', fontsize=9, va='bottom',
                bbox=dict(facecolor='black', alpha=0.55, pad=3))

    # ── ⑧ Mask + bounding box composite ──────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    composite = thermal_color.astype(np.float32).copy()
    heat = np.zeros_like(composite)
    heat[:, :, 1] = mask.astype(np.float32) * 0.35      # green channel tint
    composite = np.clip(composite + heat, 0, 255).astype(np.uint8)
    if contour is not None:
        cv2.drawContours(composite, [contour], -1, (0, 255, 100), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(composite, (x, y), (x + w, y + h), (255, 180, 0), 2)
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
    ax.set_title('⑧ Segmentation + Bounding Box\n(green fill + contour + bbox)',
                 **title_kw)
    ax.axis('off')

    # ── ⑨ Zoomed head region ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    y0, y1, x0, x1 = 150, 310, 80, 280
    zoom     = cv2.cvtColor(thermal_color[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).copy()
    z_edges  = edges_canny[y0:y1, x0:x1]
    z_mask   = mask[y0:y1, x0:x1]
    zoom[z_edges > 0] = [255, 220, 50]                  # yellow Canny edges
    z_contours, _ = cv2.findContours(z_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if z_contours:
        cv2.drawContours(zoom, z_contours, -1, (0, 255, 100), 2)
    ax.imshow(zoom)
    ax.set_title('⑨ Zoomed: Head Region\n(yellow=Canny, green=final contour)',
                 **title_kw)
    ax.axis('off')
    ax.axvline(zoom.shape[1] // 2, color='white', alpha=0.2, linewidth=0.5)
    ax.axhline(zoom.shape[0] // 2, color='white', alpha=0.2, linewidth=0.5)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.suptitle(
        'Canny + Fill: Step-by-Step Edge Detection Pipeline for Thermal Animal Segmentation',
        color='white', fontsize=15, fontweight='bold', y=0.96
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Visualization saved → {out_path}")


def save_individual_outputs(thermal_color, mask, contour, out_dir):
    """Save the binary mask and clean overlay as separate files."""
    os.makedirs(out_dir, exist_ok=True)

    # Binary mask
    mask_path = os.path.join(out_dir, 'canny_fill_mask.png')
    cv2.imwrite(mask_path, mask)
    print(f"  Binary mask saved  → {mask_path}")

    # Contour overlay
    overlay = thermal_color.copy()
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 100), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 180, 0), 2)
    overlay_path = os.path.join(out_dir, 'canny_fill_overlay.png')
    cv2.imwrite(overlay_path, overlay)
    print(f"  Contour overlay saved → {overlay_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Thermal animal segmentation via Canny + Fill (classical CV only)')
    parser.add_argument('--image',   default=None,
                        help='Path to a thermal image (JPG/PNG). '
                             'Omit to use the built-in synthetic dog image.')
    parser.add_argument('--out_dir', default='results',
                        help='Directory for output files (default: results)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Canny + Fill — Thermal Animal Boundary Detection")
    print("=" * 60)

    # ── Load image ────────────────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        thermal_color = cv2.imread(args.image)
        if thermal_color is None:
            raise ValueError(f"Cannot read image: {args.image}")
        gray = cv2.cvtColor(thermal_color, cv2.COLOR_BGR2GRAY)
        print(f"  Loaded: {args.image}  {gray.shape}")
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
                print(f"  Loaded from input_images: {image_files[0]}  {gray.shape}")
            else:
                print(f"  No images found in {input_dir} — using synthetic image.")
                print("  Generating synthetic thermal image …")
                gray, thermal_color = create_synthetic_thermal()
        else:
            print(f"  {input_dir} directory not found — using synthetic image.")
            print("  Generating synthetic thermal image …")
            gray, thermal_color = create_synthetic_thermal()

    # ── Pre-process ───────────────────────────────────────────────────────────
    print("  Pre-processing (bilateral filter + CLAHE) …")
    bilateral, enhanced = preprocess(gray)

    # ── Canny + Fill ──────────────────────────────────────────────────────────
    print("  Running Canny + Fill segmentation …")
    (mask, edges_canny, grad_8u, grad_thresh,
     fused_edges, closed_edges, contour, otsu_val) = canny_fill_segmentation(
         gray, bilateral, enhanced)

    # ── Report ────────────────────────────────────────────────────────────────
    H, W = gray.shape
    area_px  = int((mask > 0).sum())
    area_pct = area_px / (H * W) * 100
    print(f"\n  Otsu threshold:  {otsu_val:.0f}")
    print(f"  Canny thresholds: {otsu_val*0.5:.0f} / {otsu_val:.0f}")
    print(f"  Canny edge pixels: {int((edges_canny > 0).sum()):,}")
    print(f"  Contour points:  {len(contour):,}" if contour is not None else "  No contour found")
    print(f"  Segmented area:  {area_px:,} px  ({area_pct:.1f}% of frame)")
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        print(f"  Bounding box:    x={x}, y={y}, w={w}, h={h}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n  Saving outputs …")
    os.makedirs(args.out_dir, exist_ok=True)
    vis_path = os.path.join(args.out_dir, 'canny_fill_edge_detection.png')
    visualize_pipeline(
        thermal_color, gray, bilateral,
        mask, edges_canny, grad_8u, grad_thresh,
        fused_edges, closed_edges, contour, otsu_val,
        out_path=vis_path
    )
    save_individual_outputs(thermal_color, mask, contour, args.out_dir)

    print("\n  Done.")


if __name__ == '__main__':
    main()
