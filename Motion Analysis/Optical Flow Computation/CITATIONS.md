# Citations — Optical Flow Computation Project

All sources cited in `optical_flow.py` and the project documentation.

---

## Academic Papers & Books

**[1]** Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. In *Proceedings of the 13th Scandinavian Conference on Image Analysis (SCIA)*, Lecture Notes in Computer Science, vol. 2749, pp. 363–370. Springer, Berlin, Heidelberg.
https://doi.org/10.1007/3-540-45103-X_50

> *Used for: dense optical flow computation (`cv2.calcOpticalFlowFarneback`). The Farnebäck algorithm estimates per-pixel motion vectors by fitting local polynomial expansions to image intensity and tracking their displacement between frames.*

---

**[2]** Lucas, B. D., & Kanade, T. (1981). An Iterative Image Registration Technique with an Application to Stereo Vision. In *Proceedings of the 7th International Joint Conference on Artificial Intelligence (IJCAI)*, pp. 674–679.

> *Used for: sparse optical flow tracking (`cv2.calcOpticalFlowPyrLK`). The Lucas–Kanade method estimates motion at selected feature points by assuming a locally constant flow within a small neighbourhood window.*

---

**[3]** Shi, J., & Tomasi, C. (1994). Good Features to Track. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 593–600.
https://doi.org/10.1109/CVPR.1994.323794

> *Used for: keypoint/corner detection before sparse tracking (`cv2.goodFeaturesToTrack`). The Shi–Tomasi criterion selects corners that are most suitable for reliable tracking across frames.*

---

**[4]** Horn, B. K. P., & Schunck, B. G. (1981). Determining Optical Flow. *Artificial Intelligence*, 17(1–3), pp. 185–203.
https://doi.org/10.1016/0004-3702(81)90024-2

> *Used for: foundational concepts of optical flow direction encoding and the HSV hue-direction mapping. The Horn–Schunck method introduced the global smoothness constraint and established the mathematical framework for dense optical flow estimation.*

---

**[5]** Gibson, J. J. (1950). *The Perception of the Visual World.* Houghton Mifflin.

> *Used for: time-to-collision (TTC) estimation via the Focus of Expansion (FOE) and looming / τ (tau) theory. Gibson's concept of optic flow underpins the relationship between radially diverging flow patterns and imminent collision:*
> τ ≈ distance\_from\_FOE / local\_flow\_magnitude

---

**[9]** Beauchemin, S. S., & Barron, J. L. (1995). The Computation of Optical Flow. *ACM Computing Surveys*, 27(3), pp. 433–466.
https://doi.org/10.1145/212094.212141

> *Used for: motion segmentation, independently moving object (IMO) detection, and magnitude-threshold-based binary masking. Provides a comprehensive survey of optical flow methods and evaluation criteria.*

---

**[10]** Fleet, D. J., & Weiss, Y. (2006). Optical Flow Estimation. In Paragios, N., Chen, Y., & Faugeras, O. (Eds.), *Handbook of Mathematical Models in Computer Vision.* Springer, pp. 237–257.
https://doi.org/10.1007/0-387-28831-7_15

> *Used for: distinguishing camera motion from independently moving objects (IMOs), and interpreting flow direction patterns (pan, tilt, zoom, looming).*

---

## Software & Libraries

**[6]** OpenCV Development Team. (2024). `cv2.calcOpticalFlowFarneback` — Video Analysis and Object Tracking. *OpenCV Documentation*, version 4.x.
https://docs.opencv.org/4.x/dc/d6b/group__video__track.html

> *Used for: dense Farnebäck optical flow computation and the HSV colour-encoding of flow vectors (hue = direction, value = magnitude).*

---

**[7]** OpenCV Development Team. (2024). `cv2.calcOpticalFlowPyrLK` — Video Analysis and Object Tracking. *OpenCV Documentation*, version 4.x.
https://docs.opencv.org/4.x/dc/d6b/group__video__track.html

> *Used for: pyramidal Lucas–Kanade sparse optical flow to track Shi–Tomasi feature points between consecutive frames.*

---

**[8]** OpenCV Development Team. (2024). `cv2.goodFeaturesToTrack` — Image Feature Detection. *OpenCV Documentation*, version 4.x.
https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html

> *Used for: Shi–Tomasi corner detection to select trackable keypoints prior to Lucas–Kanade optical flow.*

---

## Python Package Dependencies

The following open-source packages are used in this project. All are available on [PyPI](https://pypi.org).

| Package | Version Used | License | Purpose |
|---|---|---|---|
| `opencv-python` | 4.13.0.92 | Apache 2.0 | Video I/O, Farnebäck flow, Lucas-Kanade, corner detection, colormaps |
| `numpy` | 2.4.2 | BSD 3-Clause | Array mathematics, flow statistics, angle/magnitude accumulation |
| `matplotlib` | 3.10.8 | PSF / BSD-style | Time-series chart, polar rose diagram, accumulated energy heatmap |
| `scipy` | ≥ 1.11.0 | BSD 3-Clause | Gaussian smoothing (`scipy.ndimage.gaussian_filter`) of energy map |
| `pillow` | 12.1.1 | HPND (PIL) | Image I/O backend for Matplotlib |

### Full Pinned Dependency List (`requirements.txt`)

```
contourpy==1.3.3
cycler==0.12.1
fonttools==4.61.1
kiwisolver==1.4.9
matplotlib==3.10.8
numpy==2.4.2
opencv-python==4.13.0.92
packaging==26.0
pillow==12.1.1
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scipy>=1.11.0
six==1.17.0
```

---

## Citation Index

| # | Short Reference |
|---|---|
| \[1\] | Farnebäck (2003) — polynomial expansion dense flow |
| \[2\] | Lucas & Kanade (1981) — iterative sparse flow |
| \[3\] | Shi & Tomasi (1994) — good features to track |
| \[4\] | Horn & Schunck (1981) — determining optical flow |
| \[5\] | Gibson (1950) — perception of the visual world / looming / TTC |
| \[6\] | OpenCV Docs — `calcOpticalFlowFarneback` |
| \[7\] | OpenCV Docs — `calcOpticalFlowPyrLK` |
| \[8\] | OpenCV Docs — `goodFeaturesToTrack` |
| \[9\] | Beauchemin & Barron (1995) — computation of optical flow (survey) |
| \[10\] | Fleet & Weiss (2006) — optical flow estimation (handbook chapter) |
