# Citations

All references cited in this project (Lucas-Kanade Optical Flow — Theory,
Visualisation & Validation).

---

## Academic Papers

### 1. Lucas & Kanade (1981) — Lucas-Kanade Optical Flow

> B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with
> an Application to Stereo Vision," in *Proceedings of the 7th International
> Joint Conference on Artificial Intelligence (IJCAI)*, Vancouver, BC, Canada,
> vol. 2, pp. 674–679, 1981.

**Used for:**
The foundational algorithm implemented throughout this project. Introduces the
brightness constancy assumption, the Optical Flow Constraint Equation (OFCE),
and the constant-flow neighbourhood assumption that transforms the
underdetermined single-pixel problem into an overdetermined least-squares
system solved via the 2×2 structure tensor (normal equations). Referenced
explicitly in Step 4 of the derivation (`lucas_kanade_tracker.py`, `THEORY`
block and README §4.1).

---

### 2. Bouguet (2001) — Pyramidal Lucas-Kanade

> J.-Y. Bouguet, "Pyramidal Implementation of the Lucas-Kanade Feature
> Tracker: Description of the Algorithm," *Intel Corporation, Microprocessor
> Research Labs*, Santa Clara, CA, 2001.

**Used for:**
The coarse-to-fine Gaussian image pyramid extension described in Step 6 of the
derivation. Allows the LK algorithm to handle large inter-frame displacements
that violate the small-displacement Taylor linearisation. The OpenCV function
`cv2.calcOpticalFlowPyrLK` is a direct implementation of this method and is
used for all sparse feature tracking in `lucas_kanade_tracker.py`.

---

### 3. Shi & Tomasi (1994) — Good Features to Track

> J. Shi and C. Tomasi, "Good Features to Track," in *Proceedings of the IEEE
> Conference on Computer Vision and Pattern Recognition (CVPR)*, Seattle, WA,
> USA, pp. 593–600, June 1994.

**Used for:**
The feature-selection criterion `min(λ₁, λ₂) > threshold` (Shi-Tomasi score),
where λ₁ and λ₂ are the eigenvalues of the structure tensor M. Points
satisfying this criterion are corners that are well-conditioned for LK
tracking. Implemented via `cv2.goodFeaturesToTrack` and visualised in the
eigenvalue scatter plot (`lk_eigenvalue_scatter_<tag>.png`). Referenced in
README §4.1 and throughout `lucas_kanade_tracker.py`.

---

### 4. Farnebäck (2003) — Dense Two-Frame Motion Estimation

> G. Farnebäck, "Two-Frame Motion Estimation Based on Polynomial Expansion,"
> in *Proceedings of the 13th Scandinavian Conference on Image Analysis
> (SCIA)*, Gothenburg, Sweden, *Lecture Notes in Computer Science*, vol. 2749,
> pp. 363–370, Springer, Berlin, Heidelberg, 2003.

**Used for:**
The dense per-pixel optical flow method used to generate the HSV colour-map
outputs (`lk_dense_hsv_<tag>.png` and `vid_dense_hsv_<tag>.mp4`). Each pixel's
flow vector is computed via polynomial expansion of the local neighbourhood.
Hue encodes direction and brightness encodes speed. Implemented via
`cv2.calcOpticalFlowFarneback` in `lucas_kanade_tracker.py`.

---

## Software Libraries

### 5. OpenCV

> G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*,
> 2000. Available: https://opencv.org

**Version used:** `opencv-python >= 4.8.0` (see `requirements.txt`)

**Functions used in this project:**

| Function | Purpose |
|---|---|
| `cv2.goodFeaturesToTrack` | Shi-Tomasi corner detection |
| `cv2.calcOpticalFlowPyrLK` | Pyramidal Lucas-Kanade sparse flow |
| `cv2.calcOpticalFlowFarneback` | Dense Farneback flow |
| `cv2.Sobel` | Spatial gradient computation (Iₓ, Iᵧ) |
| `cv2.GaussianBlur` / `cv2.getGaussianKernel` | Gaussian window weights |
| `cv2.VideoCapture` / `cv2.VideoWriter` | Video I/O |
| `cv2.arrowedLine`, `cv2.circle` | Annotation drawing |
| `cv2.cartToPolar` | Polar conversion for HSV encoding |
| `cv2.cvtColor` | Colour space conversions |

---

### 6. NumPy

> C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585,
> pp. 357–362, 2020. https://doi.org/10.1038/s41586-020-2649-2

**Version used:** `numpy >= 1.24.0` (see `requirements.txt`)

**Used for:** All array mathematics — gradient computation, structure tensor
construction, eigenvalue analysis (`np.linalg.eigvalsh`), bilinear weight
calculation, flow magnitude normalisation, and synthetic frame generation
(`np.random.default_rng`).

---

### 7. Matplotlib

> J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science
> & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.
> https://doi.org/10.1109/MCSE.2007.55

**Version used:** `matplotlib >= 3.7.0` (see `requirements.txt`)

**Used for:** All static figure generation — the theory derivation PNG
(`lk_theory.png`), tracking overlay (`lk_tracking_<tag>.png`), dense HSV
colour-map figure (`lk_dense_hsv_<tag>.png`), quiver arrow field
(`lk_quiver_<tag>.png`), eigenvalue scatter plot
(`lk_eigenvalue_scatter_<tag>.png`), bilinear neighbourhood detail
(`lk_bilinear_detail_<tag>.png`), and error bar charts (`lk_errors_<tag>.png`).

---

## Mathematical Concepts Referenced

### Brightness Constancy Equation

The assumption `I(x, y, t) = I(x+dx, y+dy, t+dt)` and its first-order Taylor
linearisation are standard results in computer vision. The formulation used
here follows Lucas & Kanade (1981) [citation 1 above].

### Bilinear Interpolation

The derivation from 1-D linear interpolation to the 2-D weighted formula
`f(x,y) = (1−a)(1−b)·Q₁₁ + a(1−b)·Q₂₁ + (1−a)b·Q₁₂ + ab·Q₂₂` is a
classical result in numerical analysis and image processing. The area/weight
interpretation (each weight equals the area of the opposite sub-rectangle in
the unit cell) is standard; see, e.g.:

> R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed.,
> Pearson, 2018, §2.4 (Sampling and Quantisation) and §16.1 (Geometric
> Transformations).

### Structure Tensor & Eigenvalue Analysis

The 2×2 structure tensor `M = AᵀWA` and its eigenvalue-based
corner/edge/flat classification follow the Harris corner detector literature
and the Shi-Tomasi refinement [citation 3 above]:

> C. Harris and M. Stephens, "A Combined Corner and Edge Detector," in
> *Proceedings of the 4th Alvey Vision Conference*, Manchester, UK,
> pp. 147–151, 1988.

---

## Citation Summary Table

| # | Authors | Title (short) | Venue / Publisher | Year |
|---|---|---|---|---|
| 1 | Lucas & Kanade | Iterative Image Registration | IJCAI | 1981 |
| 2 | Bouguet | Pyramidal LK Implementation | Intel Corporation | 2001 |
| 3 | Shi & Tomasi | Good Features to Track | CVPR | 1994 |
| 4 | Farnebäck | Two-Frame Motion Estimation | SCIA / Springer LNCS | 2003 |
| 5 | Bradski | The OpenCV Library | Dr. Dobb's Journal | 2000 |
| 6 | Harris et al. | Array Programming with NumPy | Nature | 2020 |
| 7 | Hunter | Matplotlib: A 2D Graphics Environment | Comp. in Science & Eng. | 2007 |
| 8 | Gonzalez & Woods | Digital Image Processing (4th ed.) | Pearson | 2018 |
| 9 | Harris & Stephens | Combined Corner and Edge Detector | Alvey Vision Conference | 1988 |
