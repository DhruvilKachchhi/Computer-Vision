"""
video_player.py – Plays a video file inside a Tkinter Toplevel window.
Primary: OpenCV frame-by-frame rendering with audio via pygame mixer.
Fallback: python-vlc if installed.
"""

import tkinter as tk
from tkinter import messagebox
import threading
import time
import os
import subprocess

import cv2
from PIL import Image, ImageTk

# Try to import pygame for audio playback
try:
    import pygame
    pygame.init()
    pygame.mixer.init()
    AUDIO_SUPPORTED = True
except ImportError:
    AUDIO_SUPPORTED = False


class VideoPlayer:
    """
    Embeds a video player inside a Tkinter Toplevel.
    Uses OpenCV for decoding and renders frames to a Canvas.
    """

    def __init__(self, parent, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Video file not found: {filepath}")

        self.filepath = filepath
        self._running = False
        self._thread = None
        self._photo = None
        self._audio_thread = None

        self.win = tk.Toplevel(parent)
        self.win.title(f"🎬 {os.path.basename(filepath)}")
        self.win.configure(bg="black")
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent close during session

        # Probe video dimensions
        cap_probe = cv2.VideoCapture(filepath)
        vw = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
        cap_probe.release()

        # Scale to max 900x600
        scale = min(900 / max(vw, 1), 600 / max(vh, 1), 1.0)
        self._dw = int(vw * scale)
        self._dh = int(vh * scale)

        self.canvas = tk.Canvas(self.win, width=self._dw, height=self._dh, bg="black",
                                highlightthickness=0)
        self.canvas.pack()

        # Controls
        ctrl = tk.Frame(self.win, bg="#111")
        ctrl.pack(fill=tk.X)
        self._pos_var = tk.StringVar(value="00:00")
        tk.Label(ctrl, textvariable=self._pos_var, font=("Helvetica", 10),
                 fg="#aaa", bg="#111").pack(side=tk.LEFT, padx=10, pady=4)
        tk.Label(ctrl, text="Playing for session…", font=("Helvetica", 10),
                 fg="#666", bg="#111").pack(side=tk.RIGHT, padx=10)

        self.win.geometry(f"{self._dw}x{self._dh + 30}")
        self.win.lift()

        # Start playback threads
        self._running = True
        self._thread = threading.Thread(target=self._play_loop, daemon=True)
        self._thread.start()
        
        # Start audio playback if supported
        if AUDIO_SUPPORTED:
            self._audio_thread = threading.Thread(target=self._play_audio, daemon=True)
            self._audio_thread.start()

    def _play_loop(self):
        cap = cv2.VideoCapture(self.filepath)
        frame_delay = 1.0 / self._fps
        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while self._running:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_num = 0
                continue

            frame_num += 1
            # Resize for display
            frame_resized = cv2.resize(frame, (self._dw, self._dh))
            # Convert BGR→RGB
            rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            # Schedule UI update in main thread
            self.win.after(0, self._update_frame, img, frame_num)

            time.sleep(frame_delay)

        cap.release()

    def _play_audio(self):
        """Play audio using pygame mixer."""
        try:
            # Load and play audio
            pygame.mixer.music.load(self.filepath)
            pygame.mixer.music.play()
            
            # Keep playing while video is running
            while self._running:
                if not pygame.mixer.music.get_busy():
                    # Audio finished, restart if video is still playing
                    pygame.mixer.music.rewind()
                    pygame.mixer.music.play()
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio playback error: {e}")

    def _update_frame(self, img, frame_num):
        if not self.win.winfo_exists():
            return
        self._photo = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        # Update position label
        secs = frame_num / self._fps
        m, s = divmod(int(secs), 60)
        self._pos_var.set(f"{m:02d}:{s:02d}")

    def close(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._audio_thread:
            self._audio_thread.join(timeout=2)
        if AUDIO_SUPPORTED:
            pygame.mixer.music.stop()
        if hasattr(self, 'win') and self.win.winfo_exists():
            self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
            self.win.destroy()
