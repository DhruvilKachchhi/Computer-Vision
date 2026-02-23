"""
blink_detector.py – Eye Aspect Ratio (EAR) based blink detection
Uses MediaPipe Face Mesh for facial landmark detection.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import threading
from collections import deque


# MediaPipe Face Mesh landmark indices for left/right eye
# Based on MediaPipe's 468-point face mesh
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(landmarks, eye_indices, img_w, img_h):
    """
    Compute Eye Aspect Ratio for a set of 6 eye landmark indices.
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append((lm.x * img_w, lm.y * img_h))

    # Vertical distances
    A = _euclidean(pts[1], pts[5])
    B = _euclidean(pts[2], pts[4])
    # Horizontal distance
    C = _euclidean(pts[0], pts[3])

    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)


class BlinkDetector:
    """
    Runs webcam capture and blink detection in a background thread.
    Call start() to begin, stop() to end.
    """

    DEFAULT_EAR_THRESHOLD = 0.21
    CONSEC_FRAMES = 2          # frames EAR must be below threshold to count as blink
    BASELINE_DURATION = 5.0    # seconds for auto-calibration
    CALIBRATION_MARGIN = 0.10  # subtract from mean EAR for threshold

    def __init__(self, on_status=None):
        """
        on_status: optional callback(str) for status/warning messages
        """
        self.on_status = on_status or (lambda msg: None)

        self._cap = None
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        # State
        self.ear_threshold = self.DEFAULT_EAR_THRESHOLD
        self.total_blinks = 0
        self.blink_times = []          # timestamps of each blink
        self._consec_below = 0
        self._blink_in_progress = False
        self._start_time = None
        self._elapsed = 0.0

        # Calibration
        self._calibrating = False
        self._cal_ear_samples = []
        self._calibrated = False

        # Face Mesh
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = None

        # Lighting warning: track recent frames without a face
        self._no_face_streak = 0
        self._no_face_warned = False

    # ------------------------------------------------------------------ public

    def start(self):
        """Open webcam and start detection thread."""
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Please check your camera.")

        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.total_blinks = 0
        self.blink_times = []
        self._consec_below = 0
        self._blink_in_progress = False
        self._start_time = None
        self._elapsed = 0.0
        self._calibrating = True
        self._calibrated = False
        self._cal_ear_samples = []
        self._no_face_streak = 0
        self._no_face_warned = False
        self._running = True

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.on_status("Calibrating… keep eyes open and look at the camera.")

    def stop(self):
        """Stop detection and return results dict."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
        if self._face_mesh:
            self._face_mesh.close()

        elapsed = self._elapsed if self._elapsed > 0 else 1.0
        bps = self.total_blinks / elapsed
        bpm = bps * 60.0

        return {
            "total_blinks": self.total_blinks,
            "blinks_per_second": round(bps, 4),
            "blinks_per_minute": round(bpm, 2),
            "duration_seconds": round(elapsed, 1),
            "ear_threshold": round(self.ear_threshold, 3),
            "blink_times": list(self.blink_times),
        }

    @property
    def elapsed_seconds(self):
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # --------------------------------------------------------------- internals

    def _run_loop(self):
        cal_start = time.time()
        session_started = False

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                self._no_face_streak += 1
                if self._no_face_streak > 30 and not self._no_face_warned:
                    self.on_status(
                        "⚠️  No face detected — check lighting or camera angle."
                    )
                    self._no_face_warned = True
                continue

            self._no_face_streak = 0
            self._no_face_warned = False
            lms = results.multi_face_landmarks[0].landmark

            left_ear = compute_ear(lms, LEFT_EYE_INDICES, w, h)
            right_ear = compute_ear(lms, RIGHT_EYE_INDICES, w, h)
            ear = (left_ear + right_ear) / 2.0

            # --- Calibration phase ---
            if self._calibrating:
                self._cal_ear_samples.append(ear)
                elapsed_cal = time.time() - cal_start
                remaining = int(self.BASELINE_DURATION - elapsed_cal)
                if elapsed_cal < self.BASELINE_DURATION:
                    self.on_status(
                        f"Calibrating… {remaining}s remaining. Keep eyes open."
                    )
                else:
                    self._finalize_calibration()
                continue

            # --- Session phase ---
            if not session_started:
                session_started = True
                self._start_time = time.time()
                self.on_status("Detection active — session running.")

            self._elapsed = time.time() - self._start_time

            # Blink detection
            if ear < self.ear_threshold:
                self._consec_below += 1
            else:
                if self._consec_below >= self.CONSEC_FRAMES:
                    self.total_blinks += 1
                    self.blink_times.append(self._elapsed)
                self._consec_below = 0

        # Capture final elapsed
        if self._start_time:
            self._elapsed = time.time() - self._start_time

    def _finalize_calibration(self):
        if self._cal_ear_samples:
            mean_ear = np.mean(self._cal_ear_samples)
            self.ear_threshold = max(0.15, mean_ear - self.CALIBRATION_MARGIN)
        else:
            self.ear_threshold = self.DEFAULT_EAR_THRESHOLD

        self._calibrating = False
        self.on_status(
            f"Calibration complete. Threshold: {self.ear_threshold:.3f}. Session starting…"
        )
