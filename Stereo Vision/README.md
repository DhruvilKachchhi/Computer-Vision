# Uncalibrated Stereo Vision — Distance Estimation

A complete Python implementation of an **uncalibrated stereo vision** pipeline that estimates the depth of a selected object (square) in a classroom using two hand-held photographs captured with a Samsung Galaxy M34 smartphone.

---

## Assignment Overview

| Parameter | Value |
|---|---|
| Camera | Samsung Galaxy M34 |
| Resolution (native) | 4080 × 3060 px (12 MP) |
| Resolution (stored) | 1600 × 1200 px |
| Focal Length | 27 mm (35 mm equiv.) |
| Aperture / ISO | f/1.8 / ISO 150 |
| Stereo Baseline | 1 foot (30.48 cm) |
| Ground-Truth Distance | 7 feet (213.36 cm) |
| Target Object | Square target in classroom |

---

## Results

| Metric | Value |
|---|---|
| Focal length (pixels) | 1247.98 px |
| Disparity | 166.0 px |
| **Estimated Distance** | **229.15 cm = 7.52 ft** |
| Ground Truth | 213.36 cm = 7.00 ft |
| Absolute Error | 15.79 cm **(7.40%)** |
| rank(F) | 2 ✓ |
| E singular values | σ₁ = σ₂ = 10.13, σ₃ ≈ 0 ✓ |
| det(R) | 1.00000000 ✓ |
| Rotation angle θ | 7.44° |

---

## Project Structure

```
Stereo Vision/
├── left.jpeg                # Left stereo image
├── right.jpeg               # Right stereo image
├── stereo_vision.py         # Main implementation (assignment code only)
├── create_word_report.py    # Generates the Word derivation report
├── requirements.txt         # Python package dependencies
├── setup_and_run.bat        # One-click: creates venv, installs deps, runs all
├── output_stereo.png        # [generated] Annotated stereo pair plot
├── results.json             # [generated] Computed matrices (F, E, R, t)
├── stereo_report.docx       # [generated] Word report with OMML equations
└── venv/                    # [generated] Python virtual environment
```

---

## Quick Start

### Option 1 — One-click Batch File (Recommended)

```bat
.\setup_and_run.bat
```

This will:
1. Create a Python virtual environment (`venv/`)
2. Install all dependencies from `requirements.txt`
3. Run `stereo_vision.py` → produces `output_stereo.png` and `results.json`
4. Run `create_word_report.py` → produces `stereo_report.docx`

### Option 2 — Manual Run

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the stereo vision pipeline
python stereo_vision.py

# Generate the Word report
python create_word_report.py
```

---

## Pipeline Steps (`stereo_vision.py`)

```
Step 1 │ Load Images          left.jpeg, right.jpeg  (1600×1200 px)
Step 2 │ SIFT Matching        Brute-Force L2 + Lowe ratio test (0.75)
Step 3 │ Fundamental Matrix   Normalised 8-point algorithm + RANSAC
Step 4 │ Essential Matrix     E = KᵀFK  (rank-2 enforced via SVD)
Step 5 │ Rotation + Trans.    SVD(E) decomposition + cheirality check
Step 6 │ Square Detection     Canny → contours → 4-sided polygon filter
Step 7 │ Depth Estimation     Z = f·B / d  (stereo depth formula)
Step 8 │ Visualisation        Side-by-side annotated plot → output_stereo.png
Step 9 │ Save Results         results.json  (for report generation)
```

---

## Camera Intrinsics Derivation

The intrinsic matrix **K** is derived entirely from EXIF metadata — no formal calibration required (uncalibrated setup):

```
Sensor:           1/2.0"  →  width = 6.40 mm, height = 4.80 mm
Sensor diagonal:  √(6.40² + 4.80²) = 8.00 mm
35mm diagonal:    43.27 mm
Crop factor:      43.27 / 8.00 = 5.409
Actual FL:        27 mm / 5.409 = 4.99 mm
Pixel size:       6.40 mm / 4080 px = 1.569 μm/px
f (native px):    4.99 / 0.001569 = 3179 px
Scale:            1600 / 4080 = 0.3922
f (stored px):    3179 × 0.3922 = 1247.98 px

K = [[1247.98    0    800.0]
     [   0    1247.98  600.0]
     [   0       0      1.0]]
```

---

## Matrix Derivations (Summary)

### Fundamental Matrix F
- Computed from SIFT correspondences via the **Normalised 8-Point Algorithm**
- RANSAC outlier rejection (Sampson distance threshold = 1.0 px)
- Epipolar constraint: **x′ᵀ F x = 0**
- Denormalisation: **F = T′ᵀ F̃ T**
- Properties: rank(F) = 2, det(F) = 0, 7 DOF

### Essential Matrix E
- **E = Kᵀ F K** (calibrated version of F)
- Physical meaning: **E = [t]× R**
- Rank-2 enforcement: SVD → replace singular values with (σ̄, σ̄, 0)

### Rotation Matrix R
- SVD(E) = U Σ Vᵀ
- Auxiliary matrix: **W = [[0,−1,0],[1,0,0],[0,0,1]]**
- Four candidates: **R₁ = UWVᵀ**, **R₂ = UWᵀVᵀ**, **t̂ = ±U[:,2]**
- Correct solution selected via **cheirality check** (points in front of both cameras)
- Properties: det(R) = +1, RᵀR = I₃

### Distance Estimation
```
Z = (f × B) / d

f = 1247.98 px  (focal length)
B = 30.48 cm    (baseline = 1 foot)
d = 166.0 px    (horizontal disparity)

Z = (1247.98 × 30.48) / 166.0 = 229.15 cm = 7.52 ft
```

---

## Why Is This "Uncalibrated"?

| Calibrated Stereo | Uncalibrated Stereo (this project) |
|---|---|
| Fixed stereo rig, pre-measured R and t | Two free-hand photos, R/t unknown |
| Rectified images | No rectification |
| Checkerboard calibration for K | K approximated from EXIF specs |
| F derived from known geometry | **F computed from data only** |

The key property: **F is estimated purely from feature correspondences** — no prior knowledge of the camera pose is needed. This is the defining characteristic of uncalibrated stereo.

---

## Output Files

### `output_stereo.png`
Side-by-side annotated stereo pair showing:
- **Cyan `+` marks** — SIFT inlier feature correspondences
- **Green rectangle** — detected square bounding box
- **Red dot** — object centroid
- **Yellow annotation** — estimated vs ground-truth distance

### `stereo_report.docx`
Comprehensive Word document (7 sections) with:
- Camera specifications and intrinsic matrix K
- Full derivation of Fundamental Matrix F (with OMML equations)
- Full derivation of Essential Matrix E (with OMML equations)
- Full derivation of Rotation Matrix R (with OMML equations)
- Stereo depth formula and numerical computation
- Annotated stereo plot
- Conclusions

---

## Dependencies

```
opencv-python  >= 4.8.0
numpy          >= 1.24.0
matplotlib     >= 3.7.0
Pillow         >= 10.0.0
python-docx    >= 1.1.0
lxml           >= 4.9.0
reportlab      >= 4.0.0
```

---

## References

1. Hartley, R. (1997). *In defense of the eight-point algorithm*. IEEE TPAMI, 19(6), 580–593.
2. Hartley, R. & Zisserman, A. (2004). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
3. Lowe, D. G. (2004). *Distinctive image features from scale-invariant keypoints*. IJCV, 60(2), 91–110.
4. OpenCV documentation: `cv2.findFundamentalMat`, `cv2.triangulatePoints`.
