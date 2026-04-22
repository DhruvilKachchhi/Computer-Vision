"""
================================================================================
  Facial Analysis Pipeline — Blink Rate & Facial Dimension Estimation
================================================================================
Author : Computer Vision Pipeline
Version: 2.0.0  (MediaPipe Tasks API — compatible with mediapipe >= 0.10)

Description:
    Processes a batch of video files and extracts two categories of metrics:
      A) Blink Rate   — Eye Aspect Ratio (EAR) threshold detection
      B) Facial Dimensions — Face, Eye, Nose, and Mouth bounding measurements

    Outputs:
      • Console log per video
      • Markdown summary report  (facial_analysis_report_<timestamp>.md)
      • CSV raw data file        (facial_analysis_data_<timestamp>.csv)

    On first run the script automatically downloads the MediaPipe
    FaceLandmarker model (~1 MB) into the project folder.

Usage:
    python facial_analysis.py                        # uses ./Videos folder
    python facial_analysis.py --video_dir ./Videos  # explicit folder
    python facial_analysis.py --video_dir C:\path\to\folder --report_dir C:\reports

Dependencies (install via requirements.txt):
    opencv-python, mediapipe>=0.10, numpy, scipy, tqdm
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import csv
import time
import urllib.request
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import distance as dist
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — GLOBAL CONFIGURATION  (adjust these constants as needed)
# ─────────────────────────────────────────────────────────────────────────────

# ── Blink Detection Thresholds ────────────────────────────────────────────────
# EAR below this value is considered a "closed" eye.
# Typical range: 0.18–0.25.  Lower = stricter (only deep blinks detected).
EAR_THRESHOLD: float = 0.20

# Number of consecutive frames the EAR must stay below EAR_THRESHOLD
# for the event to be counted as a single blink.
# Increase to ignore very brief flickers; decrease to catch fast blinks.
EAR_CONSEC_FRAMES: int = 2

# ── Video Processing ──────────────────────────────────────────────────────────
# How many initial frames to scan looking for a detectable face before skipping.
FACE_DETECTION_GRACE_FRAMES: int = 90   # ~3 s at 30 fps

# Process every Nth frame (1 = every frame; 2 = every other frame, etc.)
# Higher values speed up processing but reduce blink accuracy.
FRAME_SKIP: int = 1

# Supported video extensions (case-insensitive check applied automatically).
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# ── MediaPipe FaceLandmarker ──────────────────────────────────────────────────
# min_face_detection_confidence: 0–1 (higher = fewer false detections)
MP_DETECTION_CONFIDENCE: float = 0.5
# min_tracking_confidence: 0–1 (higher = more stable tracking but may lose face)
MP_TRACKING_CONFIDENCE: float = 0.5

# Path to store the downloaded MediaPipe model file.
# The model is downloaded automatically on first run (~1 MB).
MODEL_PATH: Path = Path(__file__).parent / "face_landmarker.task"
MODEL_URL: str = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MediaPipe LANDMARK INDICES
#   Reference: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
#   The FaceLandmarker uses the same 468-point topology as the legacy FaceMesh.
# ─────────────────────────────────────────────────────────────────────────────

# ── Eye landmark indices (6 points per eye for EAR calculation) ───────────────
# Ordered as: [P1=inner_corner, P2=upper_left, P3=upper_right,
#              P4=outer_corner, P5=lower_right, P6=lower_left]
LEFT_EYE_INDICES: List[int]  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES: List[int] = [33,  160, 158, 133, 153, 144]

# Extreme corners for eye width measurement (inner corner, outer corner)
LEFT_EYE_CORNERS: List[int]  = [362, 263]
RIGHT_EYE_CORNERS: List[int] = [33,  133]

# ── Face bounding landmarks ───────────────────────────────────────────────────
FACE_TOP_IDX: int    = 10    # forehead / top of skull
FACE_BOTTOM_IDX: int = 152   # chin tip
FACE_LEFT_IDX: int   = 234   # left ear / cheekbone
FACE_RIGHT_IDX: int  = 454   # right ear / cheekbone

# ── Nose landmarks ────────────────────────────────────────────────────────────
NOSE_TOP_IDX: int    = 6     # bridge top
NOSE_BOTTOM_IDX: int = 2     # nose tip / base
NOSE_LEFT_IDX: int   = 129   # left ala (nostril wing)
NOSE_RIGHT_IDX: int  = 358   # right ala (nostril wing)

# ── Mouth / lip landmarks ─────────────────────────────────────────────────────
MOUTH_TOP_IDX: int    = 13   # upper lip centre
MOUTH_BOTTOM_IDX: int = 14   # lower lip centre
MOUTH_LEFT_IDX: int   = 61   # left mouth corner
MOUTH_RIGHT_IDX: int  = 291  # right mouth corner

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MODEL MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def ensure_model(model_path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """
    Ensure the FaceLandmarker model file exists locally.

    If the model file is missing, it is downloaded automatically from the
    official MediaPipe model repository (~1.5 MB, float16 precision).

    Args:
        model_path : Local destination path for the .task model file.
        url        : Remote URL of the MediaPipe FaceLandmarker model bundle.

    Returns:
        The resolved Path to the model file.

    Raises:
        RuntimeError: If the download fails.
    """
    if model_path.exists():
        logger.info(f"  Model found  : {model_path.name}")
        return model_path

    logger.info(f"  Downloading FaceLandmarker model → {model_path.name} ...")
    logger.info(f"  Source: {url}")

    try:
        def _reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                # Print a simple inline progress indicator
                print(f"\r    Progress: {pct:5.1f}%  ({downloaded//1024} KB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(model_path), reporthook=_reporthook)
        print()  # newline after progress
        size_kb = model_path.stat().st_size // 1024
        logger.info(f"  Model saved  : {model_path.name} ({size_kb} KB)")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the MediaPipe FaceLandmarker model.\n"
            f"  URL : {url}\n"
            f"  Error: {exc}\n\n"
            f"You can manually download the file and place it at:\n"
            f"  {model_path.resolve()}"
        ) from exc

    return model_path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — HELPER / UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_distance(pt_a: np.ndarray, pt_b: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two 2-D points.

    Args:
        pt_a: (x, y) array-like for point A.
        pt_b: (x, y) array-like for point B.

    Returns:
        Scalar floating-point distance in the same units as the input coords.
    """
    return float(dist.euclidean(pt_a, pt_b))


def landmark_to_pixel(landmark, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert a normalised MediaPipe NormalizedLandmark (0–1 range) to pixel coords.

    Works with both the legacy FaceMesh API and the new Tasks API, since both
    expose `.x` and `.y` attributes on the landmark object.

    Args:
        landmark : A single MediaPipe NormalizedLandmark object (has .x and .y).
        img_w    : Frame width in pixels.
        img_h    : Frame height in pixels.

    Returns:
        numpy array [x_px, y_px].
    """
    return np.array([landmark.x * img_w, landmark.y * img_h])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BLINK / EAR CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ear(eye_landmarks: List[np.ndarray]) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    The EAR formula (Soukupová & Čech, 2016):
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    where:
        p1, p4 = horizontal (corner) landmarks
        p2, p6 = upper / lower vertical pair on the inner side
        p3, p5 = upper / lower vertical pair on the outer side

    A fully open eye has EAR ≈ 0.25–0.35; a closed eye → EAR ≈ 0.0.

    Args:
        eye_landmarks: List of 6 (x,y) pixel coordinate arrays in order
                       [p1, p2, p3, p4, p5, p6].

    Returns:
        EAR value as a float.  Returns 0.0 if computation fails.
    """
    try:
        # Vertical distances (two pairs)
        A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])  # p2–p6
        B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])  # p3–p5
        # Horizontal distance (one pair)
        C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])  # p1–p4

        if C < 1e-6:   # guard against division by zero
            return 0.0

        return float((A + B) / (2.0 * C))
    except Exception:
        return 0.0


def extract_eye_landmarks(
    face_lm_list: list,
    indices: List[int],
    img_w: int,
    img_h: int,
) -> List[np.ndarray]:
    """
    Pull the 6 EAR landmark pixel coordinates for one eye.

    In the MediaPipe Tasks API, `face_lm_list` is a plain Python list of
    NormalizedLandmark objects (one per face landmark), so indexing is direct:
    `face_lm_list[i]`.

    Args:
        face_lm_list : List of NormalizedLandmark objects for one detected face.
        indices      : List of 6 landmark index integers.
        img_w, img_h : Frame dimensions for denormalisation.

    Returns:
        List of 6 numpy (x, y) arrays in pixel coordinates.
    """
    return [
        landmark_to_pixel(face_lm_list[i], img_w, img_h)
        for i in indices
    ]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — FACIAL DIMENSION CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_facial_dimensions(
    face_lm_list: list,
    img_w: int,
    img_h: int,
) -> Dict[str, float]:
    """
    Measure pixel distances for face, eyes, nose, and mouth from a single frame.

    Measurements are Euclidean distances between specific landmark pairs rather
    than axis-aligned bounding boxes, making them robust to slight head tilts.
    The maximum value observed across all frames is retained per video.

    Args:
        face_lm_list : List of NormalizedLandmark objects (from Tasks API result).
        img_w, img_h : Frame width and height in pixels.

    Returns:
        Dictionary with keys:
            face_width, face_height,
            left_eye_width, left_eye_height,
            right_eye_width, right_eye_height,
            avg_eye_width, avg_eye_height,
            nose_width, nose_height,
            mouth_width, mouth_height
        All values are pixel distances (float), rounded to 2 decimal places.
    """
    def px(idx: int) -> np.ndarray:
        """Convert landmark at index idx to pixel coordinates."""
        return landmark_to_pixel(face_lm_list[idx], img_w, img_h)

    # ── Face ─────────────────────────────────────────────────────────────────
    face_height = euclidean_distance(px(FACE_TOP_IDX),    px(FACE_BOTTOM_IDX))
    face_width  = euclidean_distance(px(FACE_LEFT_IDX),   px(FACE_RIGHT_IDX))

    # ── Eyes — width: corner to corner; height: max vertical EAR pair ─────────
    left_eye_width  = euclidean_distance(px(LEFT_EYE_CORNERS[0]),  px(LEFT_EYE_CORNERS[1]))
    right_eye_width = euclidean_distance(px(RIGHT_EYE_CORNERS[0]), px(RIGHT_EYE_CORNERS[1]))

    left_eye_lms  = extract_eye_landmarks(face_lm_list, LEFT_EYE_INDICES,  img_w, img_h)
    right_eye_lms = extract_eye_landmarks(face_lm_list, RIGHT_EYE_INDICES, img_w, img_h)

    left_eye_height = max(
        euclidean_distance(left_eye_lms[1], left_eye_lms[5]),   # inner vertical pair
        euclidean_distance(left_eye_lms[2], left_eye_lms[4]),   # outer vertical pair
    )
    right_eye_height = max(
        euclidean_distance(right_eye_lms[1], right_eye_lms[5]),
        euclidean_distance(right_eye_lms[2], right_eye_lms[4]),
    )

    avg_eye_width  = (left_eye_width  + right_eye_width)  / 2.0
    avg_eye_height = (left_eye_height + right_eye_height) / 2.0

    # ── Nose ──────────────────────────────────────────────────────────────────
    nose_height = euclidean_distance(px(NOSE_TOP_IDX),   px(NOSE_BOTTOM_IDX))
    nose_width  = euclidean_distance(px(NOSE_LEFT_IDX),  px(NOSE_RIGHT_IDX))

    # ── Mouth ─────────────────────────────────────────────────────────────────
    mouth_height = euclidean_distance(px(MOUTH_TOP_IDX),    px(MOUTH_BOTTOM_IDX))
    mouth_width  = euclidean_distance(px(MOUTH_LEFT_IDX),   px(MOUTH_RIGHT_IDX))

    return {
        "face_width":       round(face_width,        2),
        "face_height":      round(face_height,       2),
        "left_eye_width":   round(left_eye_width,    2),
        "left_eye_height":  round(left_eye_height,   2),
        "right_eye_width":  round(right_eye_width,   2),
        "right_eye_height": round(right_eye_height,  2),
        "avg_eye_width":    round(avg_eye_width,     2),
        "avg_eye_height":   round(avg_eye_height,    2),
        "nose_width":       round(nose_width,         2),
        "nose_height":      round(nose_height,        2),
        "mouth_width":      round(mouth_width,        2),
        "mouth_height":     round(mouth_height,       2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PER-VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_video(video_path: Path, model_path: Path) -> Optional[Dict]:
    """
    Run the full analysis pipeline on a single video file.

    Processing steps:
        1. Open the video with OpenCV.
        2. Initialise MediaPipe FaceLandmarker in VIDEO running mode.
        3. Iterate over frames (respecting FRAME_SKIP).
        4. For each frame:
             a. Detect face landmarks using detect_for_video().
             b. Compute average EAR and track blink events via state machine.
             c. Accumulate maximum facial dimension measurements.
        5. Compute blink rate = total_blinks / video_duration_seconds.
        6. Return a structured result dictionary.

    Args:
        video_path  : pathlib.Path pointing to the video file.
        model_path  : pathlib.Path to the face_landmarker.task model file.

    Returns:
        Dictionary of results, or None if the video cannot be processed
        (e.g. file unreadable or no face detected within grace period).
    """
    filename = video_path.name
    logger.info(f"  ▶ Processing: {filename}")

    # ── Open Video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"    ✖ Cannot open video: {filename}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    img_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    logger.info(
        f"    Info → {img_w}×{img_h} px | {fps:.1f} fps | "
        f"{total_frames} frames | {duration_sec:.1f} s"
    )

    # ── MediaPipe FaceLandmarker (Tasks API, VIDEO mode) ──────────────────────
    # RunningMode.VIDEO enables temporal tracking between frames (faster than
    # RunningMode.IMAGE which re-detects every frame from scratch).
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    landmarker_options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,                                    # primary subject only
        min_face_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_face_presence_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE,
        output_face_blendshapes=False,                  # not needed, saves compute
        output_facial_transformation_matrixes=False,    # not needed
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_options)

    # ── State Trackers ────────────────────────────────────────────────────────
    blink_count      = 0       # cumulative blink events
    ear_below_thresh = 0       # consecutive sub-threshold frames
    eye_closed       = False   # True once a blink has been "entered"

    # Track the maximum dimension value seen across all frames
    max_dims: Dict[str, float] = {
        "face_width": 0, "face_height": 0,
        "left_eye_width": 0, "left_eye_height": 0,
        "right_eye_width": 0, "right_eye_height": 0,
        "avg_eye_width": 0, "avg_eye_height": 0,
        "nose_width": 0, "nose_height": 0,
        "mouth_width": 0, "mouth_height": 0,
    }

    frames_with_face   = 0      # frames where a face was successfully detected
    face_detected_ever = False   # for the early-exit grace-period check

    # ── Frame Loop ────────────────────────────────────────────────────────────
    frame_idx = 0
    pbar = tqdm(
        total=total_frames,
        desc=f"    {filename[:40]}",
        unit="frame",
        leave=False,
        ncols=80,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)

        # Grace-period: abort early if no face seen in first N frames
        if not face_detected_ever and frame_idx > FACE_DETECTION_GRACE_FRAMES:
            logger.warning(
                f"    ⚠ No face detected in first {FACE_DETECTION_GRACE_FRAMES} "
                f"frames — skipping {filename}"
            )
            pbar.close()
            cap.release()
            landmarker.close()
            return None

        # Honour FRAME_SKIP — skip processing but still read the frame
        if (frame_idx % FRAME_SKIP) != 0:
            continue

        # ── MediaPipe Inference ───────────────────────────────────────────────
        # Tasks API requires mp.Image in SRGB format; OpenCV delivers BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp in milliseconds — required for VIDEO running mode.
        # Use the capture position rather than a calculated value so it stays
        # in sync even if frames are dropped by the decoder.
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # face_landmarks is a list of faces; each face is a list of NormalizedLandmark
        if not detection_result.face_landmarks:
            # No face detected in this frame; EAR state is held (not reset)
            continue

        face_detected_ever = True
        frames_with_face  += 1
        # Primary face: a flat list of NormalizedLandmark objects indexed 0–477
        face_lm = detection_result.face_landmarks[0]

        # ── EAR Computation ───────────────────────────────────────────────────
        left_eye_pts  = extract_eye_landmarks(face_lm, LEFT_EYE_INDICES,  img_w, img_h)
        right_eye_pts = extract_eye_landmarks(face_lm, RIGHT_EYE_INDICES, img_w, img_h)

        ear_left  = compute_ear(left_eye_pts)
        ear_right = compute_ear(right_eye_pts)
        ear_avg   = (ear_left + ear_right) / 2.0

        # ── Blink State Machine ───────────────────────────────────────────────
        # Transition graph:
        #   OPEN → [EAR < threshold for ≥ EAR_CONSEC_FRAMES] → CLOSED (blink +1)
        #   CLOSED → [EAR ≥ threshold] → OPEN
        if ear_avg < EAR_THRESHOLD:
            ear_below_thresh += 1
            if ear_below_thresh >= EAR_CONSEC_FRAMES and not eye_closed:
                eye_closed   = True    # enter "closed" state once
                blink_count += 1       # register one blink event
        else:
            ear_below_thresh = 0
            eye_closed       = False   # eye re-opened → ready for next blink

        # ── Facial Dimension Tracking ─────────────────────────────────────────
        dims = compute_facial_dimensions(face_lm, img_w, img_h)
        for key, value in dims.items():
            if value > max_dims[key]:
                max_dims[key] = value

    # ── Cleanup ───────────────────────────────────────────────────────────────
    pbar.close()
    cap.release()
    landmarker.close()

    # ── Validity Check ────────────────────────────────────────────────────────
    if not face_detected_ever or frames_with_face == 0:
        logger.warning(f"    ⚠ No usable face data in {filename} — skipping.")
        return None

    # ── Blink Rate ────────────────────────────────────────────────────────────
    # Divide by total video duration (not just face-present duration) for
    # a consistent metric comparable across videos of different lengths.
    blink_rate = blink_count / duration_sec if duration_sec > 0 else 0.0

    result = {
        "filename":         filename,
        "duration_sec":     round(duration_sec,  2),
        "fps":              round(fps,            2),
        "resolution":       f"{img_w}x{img_h}",
        "total_frames":     total_frames,
        "frames_with_face": frames_with_face,
        "total_blinks":     blink_count,
        "blink_rate_per_s": round(blink_rate,    4),
        **{k: round(v, 2) for k, v in max_dims.items()},
    }

    # ── Per-Video Console Summary ─────────────────────────────────────────────
    logger.info(f"    ✔ Done  → Blinks: {blink_count}  |  Rate: {blink_rate:.4f} blinks/s  "
                f"({blink_rate * 60:.1f}/min)")
    logger.info(
        f"    Dims  → Face {max_dims['face_width']:.0f}×{max_dims['face_height']:.0f} px  |  "
        f"Eye avg {max_dims['avg_eye_width']:.0f}×{max_dims['avg_eye_height']:.0f} px  |  "
        f"Nose {max_dims['nose_width']:.0f}×{max_dims['nose_height']:.0f} px  |  "
        f"Mouth {max_dims['mouth_width']:.0f}×{max_dims['mouth_height']:.0f} px"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def collect_video_files(folder: Path) -> List[Path]:
    """
    Recursively collect all supported video files from a directory.

    Performs case-insensitive extension matching so that files like
    .MP4, .mp4, and .Mp4 are all captured on any OS.

    Args:
        folder: Root directory to search.

    Returns:
        Sorted, deduplicated list of Path objects for each video found.
    """
    videos = []
    for ext in SUPPORTED_EXTENSIONS:
        videos.extend(folder.rglob(f"*{ext}"))
        videos.extend(folder.rglob(f"*{ext.upper()}"))

    # Deduplicate by lower-cased absolute path (handles case-insensitive FSes)
    seen: set = set()
    unique: List[Path] = []
    for v in sorted(videos):
        key = str(v).lower()
        if key not in seen:
            seen.add(key)
            unique.append(v)

    return unique


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown_report(
    results: List[Dict],
    report_path: Path,
    video_dir: Path,
) -> None:
    """
    Write a structured Markdown report to disk.

    Sections:
        1. Header with run metadata and configuration
        2. Measurement Legend (how each metric is computed)
        3. Per-Video Results (info table, blink table, dimension table)
        4. Aggregate Summary (averages across all videos)
        5. All-Video Comparison Table (compact grid with AVG row)

    Args:
        results    : List of per-video result dicts.
        report_path: Destination .md file path.
        video_dir  : Source folder that was scanned (for the report header).
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n   = len(results)

    # Helper: average and total of a numeric key across all results
    def avg(key: str):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else "N/A"

    def total(key: str):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return round(sum(vals), 4) if vals else "N/A"

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# Facial Analysis Pipeline — Summary Report",
        "",
        f"> **Generated:** {now}  ",
        f"> **Source folder:** `{video_dir.resolve()}`  ",
        f"> **Videos analysed:** {n}  ",
        f"> **EAR threshold:** `{EAR_THRESHOLD}` | "
        f"**Consecutive frames:** `{EAR_CONSEC_FRAMES}` | "
        f"**Frame skip:** `{FRAME_SKIP}`",
        "",
        "---",
        "",
    ]

    # ── Legend ────────────────────────────────────────────────────────────────
    lines += [
        "## Measurement Legend",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| **Blink Rate** | Total blinks ÷ video duration (blinks/second) |",
        "| **Face W / H** | Ear-to-ear width / Forehead-to-chin height (pixels) |",
        "| **Eye W / H**  | Average of left+right eye width / height (pixels) |",
        "| **Nose W / H** | Nostril-to-nostril width / bridge-to-tip height (pixels) |",
        "| **Mouth W / H**| Corner-to-corner width / lip-gap height (pixels) |",
        "",
        "> **Unit Note:** All dimensions are in *pixels* as measured in the "
        "source video frame.  \n"
        "> To convert to real-world units use:  \n"
        "> `real_cm = pixel_distance × (known_reference_cm / known_reference_px)`  \n"
        "> A standard interpupillary distance (~6.3 cm) is a reliable reference "
        "when the subject faces the camera.",
        "",
        "---",
        "",
    ]

    # ── Per-Video Results ─────────────────────────────────────────────────────
    lines += ["## Per-Video Results", ""]

    for i, r in enumerate(results, 1):
        lines += [
            f"### {i}. `{r['filename']}`",
            "",
            "**Video Info**",
            "",
            "| Property | Value |",
            "|----------|-------|",
            f"| Duration | {r['duration_sec']} s |",
            f"| FPS | {r['fps']} |",
            f"| Resolution | {r['resolution']} |",
            f"| Total Frames | {r['total_frames']} |",
            f"| Frames with Face | {r['frames_with_face']} |",
            "",
            "**Blink Analysis**",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Blinks | {r['total_blinks']} |",
            f"| Blink Rate | {r['blink_rate_per_s']} blinks/s |",
            f"| Blink Rate | {round(r['blink_rate_per_s'] * 60, 2)} blinks/min |",
            "",
            "**Facial Dimensions** *(max across all frames, in pixels)*",
            "",
            "| Feature | Width (px) | Height (px) |",
            "|---------|:----------:|:-----------:|",
            f"| Face      | {r['face_width']}       | {r['face_height']} |",
            f"| Left Eye  | {r['left_eye_width']}   | {r['left_eye_height']} |",
            f"| Right Eye | {r['right_eye_width']}  | {r['right_eye_height']} |",
            f"| Avg Eye   | {r['avg_eye_width']}    | {r['avg_eye_height']} |",
            f"| Nose      | {r['nose_width']}       | {r['nose_height']} |",
            f"| Mouth     | {r['mouth_width']}      | {r['mouth_height']} |",
            "",
        ]

    lines += ["---", ""]

    # ── Aggregate Summary ─────────────────────────────────────────────────────
    avg_blink_rate = avg("blink_rate_per_s")
    avg_blink_min  = round(avg_blink_rate * 60, 2) if isinstance(avg_blink_rate, float) else "N/A"

    lines += [
        "## Aggregate Summary (Averages Across All Videos)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Videos Processed | {n} |",
        f"| Total Duration | {total('duration_sec')} s |",
        f"| Avg Video Duration | {avg('duration_sec')} s |",
        f"| **Avg Blink Rate** | **{avg_blink_rate} blinks/s** |",
        f"| Avg Blink Rate | {avg_blink_min} blinks/min |",
        f"| Total Blinks (all videos) | {total('total_blinks')} |",
        "",
        "**Average Facial Dimensions (pixels)**",
        "",
        "| Feature | Avg Width (px) | Avg Height (px) |",
        "|---------|:--------------:|:---------------:|",
        f"| Face      | {avg('face_width')}      | {avg('face_height')} |",
        f"| Left Eye  | {avg('left_eye_width')}  | {avg('left_eye_height')} |",
        f"| Right Eye | {avg('right_eye_width')} | {avg('right_eye_height')} |",
        f"| Avg Eye   | {avg('avg_eye_width')}   | {avg('avg_eye_height')} |",
        f"| Nose      | {avg('nose_width')}      | {avg('nose_height')} |",
        f"| Mouth     | {avg('mouth_width')}     | {avg('mouth_height')} |",
        "",
        "---",
        "",
    ]

    # ── All-Video Comparison Table ─────────────────────────────────────────────
    lines += [
        "## All-Video Data Table",
        "",
        "| # | Filename | Duration (s) | Blinks | Blink Rate (bl/s) | "
        "Face W | Face H | Eye W | Eye H | Nose W | Nose H | Mouth W | Mouth H |",
        "|:-:|----------|:------------:|:------:|:-----------------:|"
        ":------:|:------:|:-----:|:-----:|:------:|:------:|:-------:|:-------:|",
    ]

    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | `{r['filename']}` | {r['duration_sec']} | "
            f"{r['total_blinks']} | {r['blink_rate_per_s']} | "
            f"{r['face_width']} | {r['face_height']} | "
            f"{r['avg_eye_width']} | {r['avg_eye_height']} | "
            f"{r['nose_width']} | {r['nose_height']} | "
            f"{r['mouth_width']} | {r['mouth_height']} |"
        )

    # Average row — bold for the key metric
    lines.append(
        f"| **AVG** | *(all {n} videos)* | {avg('duration_sec')} | "
        f"{avg('total_blinks')} | **{avg_blink_rate}** | "
        f"{avg('face_width')} | {avg('face_height')} | "
        f"{avg('avg_eye_width')} | {avg('avg_eye_height')} | "
        f"{avg('nose_width')} | {avg('nose_height')} | "
        f"{avg('mouth_width')} | {avg('mouth_height')} |"
    )

    lines += [
        "",
        "---",
        "",
        "_Report generated by the Facial Analysis Pipeline — "
        "OpenCV + MediaPipe FaceLandmarker._",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  📄 Markdown report → {report_path}")


def generate_csv_report(results: List[Dict], csv_path: Path) -> None:
    """
    Write all per-video results to a CSV file for downstream analysis.

    Args:
        results  : List of per-video result dicts (same structure as returned
                   by process_video).
        csv_path : Destination .csv file path.
    """
    if not results:
        return

    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"  📊 CSV data file  → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with attributes:
            video_dir  : str — path to folder containing videos
            report_dir : str — path to write reports (default: same as video_dir)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Facial Analysis Pipeline: "
            "Blink Rate & Facial Dimension Estimation from video files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python facial_analysis.py
  python facial_analysis.py --video_dir ./Videos
  python facial_analysis.py --video_dir C:\\Experiments --report_dir C:\\Reports
        """,
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./Videos",
        help="Path to folder containing video files (default: ./Videos)",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="Path to write output reports (default: same as --video_dir)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate the full batch processing pipeline.

    Steps:
        1. Parse CLI arguments and validate directories.
        2. Ensure the FaceLandmarker model file exists (download if needed).
        3. Collect all video files in the target directory.
        4. Process each video → accumulate results.
        5. Generate Markdown and CSV reports.
        6. Print final aggregate summary to the console.
    """
    args       = parse_arguments()
    video_dir  = Path(args.video_dir)
    report_dir = Path(args.report_dir) if args.report_dir else video_dir

    # ── Validate Directories ──────────────────────────────────────────────────
    if not video_dir.exists():
        logger.error(f"Video directory not found: {video_dir.resolve()}")
        sys.exit(1)

    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Model Download ────────────────────────────────────────────────────────
    try:
        model_path = ensure_model()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    # ── Discover Videos ───────────────────────────────────────────────────────
    video_files = collect_video_files(video_dir)
    if not video_files:
        logger.error(
            f"No supported video files ({', '.join(SUPPORTED_EXTENSIONS)}) "
            f"found in: {video_dir.resolve()}"
        )
        sys.exit(1)

    # ── Pipeline Header ───────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  FACIAL ANALYSIS PIPELINE")
    logger.info("=" * 70)
    logger.info(f"  Source folder : {video_dir.resolve()}")
    logger.info(f"  Report folder : {report_dir.resolve()}")
    logger.info(f"  Model file    : {model_path.name}")
    logger.info(f"  Videos found  : {len(video_files)}")
    logger.info(f"  EAR threshold : {EAR_THRESHOLD}  |  Consec frames: {EAR_CONSEC_FRAMES}"
                f"  |  Frame skip: {FRAME_SKIP}")
    logger.info("=" * 70)

    # ── Batch Processing ──────────────────────────────────────────────────────
    results:   List[Dict] = []
    skipped:   List[str]  = []
    t_start = time.time()

    for idx, vpath in enumerate(video_files, 1):
        logger.info(f"\n[{idx}/{len(video_files)}] ─────────────────────────────────")
        result = process_video(vpath, model_path)
        if result:
            results.append(result)
        else:
            skipped.append(vpath.name)

    elapsed = time.time() - t_start

    # ── Reports ───────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if results:
        md_path  = report_dir / f"facial_analysis_report_{timestamp}.md"
        csv_path = report_dir / f"facial_analysis_data_{timestamp}.csv"

        logger.info("\n" + "=" * 70)
        logger.info("  GENERATING REPORTS")
        logger.info("=" * 70)

        generate_markdown_report(results, md_path,  video_dir)
        generate_csv_report(results, csv_path)
    else:
        logger.warning("No successful results to report — all videos were skipped.")

    # ── Final Summary ─────────────────────────────────────────────────────────
    def avg(key: str) -> float:
        vals = [r[key] for r in results]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    logger.info("\n" + "=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Videos processed : {len(results)}")
    logger.info(f"  Videos skipped   : {len(skipped)}")
    if skipped:
        for s in skipped:
            logger.info(f"    • {s}")
    logger.info(f"  Total runtime    : {elapsed:.1f} s")

    if results:
        br = avg("blink_rate_per_s")
        logger.info("")
        logger.info("  ── Aggregate Averages ─────────────────────────────────")
        logger.info(f"  Avg Blink Rate   : {br} blinks/s  ({round(br * 60, 2)} blinks/min)")
        logger.info(f"  Avg Face Size    : {avg('face_width')} × {avg('face_height')} px")
        logger.info(f"  Avg Eye Size     : {avg('avg_eye_width')} × {avg('avg_eye_height')} px")
        logger.info(f"  Avg Nose Size    : {avg('nose_width')} × {avg('nose_height')} px")
        logger.info(f"  Avg Mouth Size   : {avg('mouth_width')} × {avg('mouth_height')} px")
        logger.info("")
        logger.info(f"  Reports saved to : {report_dir.resolve()}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
