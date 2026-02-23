"""
session_manager.py – Manages the lifecycle of movie and document sessions.
Handles file picking, countdown, threading, and result dispatch.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import time
import os

from blink_detector import BlinkDetector
from document_viewer import DocumentViewer
from video_player import VideoPlayer


SESSION_DURATION = 60  # seconds


class SessionManager:
    def __init__(self, app):
        self.app = app
        self._active = False

    def start_session(self, mode):
        if self._active:
            messagebox.showwarning("Session Active", "A session is already running. Please wait.")
            return

        if mode == "movie":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
            title = "Select a Video Clip"
        else:
            filetypes = [("Documents", "*.pdf *.docx"), ("All files", "*.*")]
            title = "Select a Document (PDF or DOCX)"

        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not filepath:
            return

        self._active = True
        t = threading.Thread(target=self._run_session, args=(mode, filepath), daemon=True)
        t.start()

    def _run_session(self, mode, filepath):
        detector = BlinkDetector(on_status=self.app.update_status)

        # Open content window in main thread
        content_win = [None]

        def open_content():
            try:
                if mode == "movie":
                    content_win[0] = VideoPlayer(self.app.root, filepath)
                else:
                    content_win[0] = DocumentViewer(self.app.root, filepath)
            except Exception as e:
                messagebox.showerror("Content Error", str(e))

        self.app.root.after(0, open_content)
        time.sleep(0.5)  # Give the window time to open

        # Start blink detection
        try:
            detector.start()
        except RuntimeError as e:
            messagebox.showerror("Webcam Error", str(e))
            self._active = False
            return

        # Show countdown window
        self.app.root.after(0, lambda: self._show_countdown(SESSION_DURATION, mode, detector, content_win))

    def _show_countdown(self, duration, mode, detector, content_win):
        """Show a floating countdown window and wait for session to complete."""
        win = tk.Toplevel(self.app.root)
        win.title("Session Running")
        win.geometry("320x200")
        win.resizable(False, False)
        win.configure(bg="#1a1a2e")
        win.attributes("-topmost", True)

        emoji = "🎬" if mode == "movie" else "📄"
        tk.Label(win, text=f"{emoji} {mode.capitalize()} Session",
                 font=("Helvetica", 14, "bold"), fg="#e94560", bg="#1a1a2e").pack(pady=(20, 5))

        tk.Label(win, text="Detecting blinks…", font=("Helvetica", 10),
                 fg="#a8a8b3", bg="#1a1a2e").pack()

        countdown_var = tk.StringVar(value=f"{duration}s remaining")
        tk.Label(win, textvariable=countdown_var, font=("Helvetica", 26, "bold"),
                 fg="#f5a623", bg="#1a1a2e").pack(pady=10)

        blink_var = tk.StringVar(value="Blinks: 0")
        tk.Label(win, textvariable=blink_var, font=("Helvetica", 12),
                 fg="#4caf50", bg="#1a1a2e").pack()

        stop_btn = tk.Button(
            win, text="Stop Early",
            font=("Helvetica", 10),
            bg="#555", fg="white", relief=tk.FLAT,
            padx=10, pady=4,
            command=lambda: setattr(self, '_stop_requested', True)
        )
        stop_btn.pack(pady=(10, 0))

        self._stop_requested = False
        start_time = time.time()

        def tick():
            if not win.winfo_exists():
                return
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            countdown_var.set(f"{int(remaining) + 1}s remaining")
            blink_var.set(f"Blinks: {detector.total_blinks}")

            if remaining <= 0 or self._stop_requested:
                win.destroy()
                self._finish_session(mode, detector, content_win)
            else:
                win.after(200, tick)

        tick()

    def _finish_session(self, mode, detector, content_win):
        results = detector.stop()

        # Close content window
        def close_content():
            cw = content_win[0]
            if cw and hasattr(cw, 'close'):
                try:
                    cw.close()
                except Exception:
                    pass

        self.app.root.after(0, close_content)

        self._active = False
        self.app.root.after(0, lambda: self._show_session_results(mode, results))

    def _show_session_results(self, mode, results):
        """Show a quick results popup, then notify the main app."""
        win = tk.Toplevel(self.app.root)
        win.title("Session Complete")
        win.geometry("380x280")
        win.resizable(False, False)
        win.configure(bg="#16213e")
        win.attributes("-topmost", True)

        emoji = "🎬" if mode == "movie" else "📄"
        tk.Label(win, text=f"{emoji} {mode.capitalize()} Session Complete",
                 font=("Helvetica", 15, "bold"), fg="#4caf50", bg="#16213e").pack(pady=(20, 10))

        stats = [
            ("Total Blinks", str(results['total_blinks'])),
            ("Blinks / Second", f"{results['blinks_per_second']:.3f}"),
            ("Blinks / Minute", f"{results['blinks_per_minute']:.1f}"),
            ("Duration", f"{results['duration_seconds']:.0f} s"),
            ("EAR Threshold", f"{results['ear_threshold']:.3f}"),
        ]

        for label, value in stats:
            row = tk.Frame(win, bg="#16213e")
            row.pack(fill=tk.X, padx=30, pady=2)
            tk.Label(row, text=label + ":", font=("Helvetica", 11),
                     fg="#a8a8b3", bg="#16213e", anchor="w", width=18).pack(side=tk.LEFT)
            tk.Label(row, text=value, font=("Helvetica", 11, "bold"),
                     fg="white", bg="#16213e", anchor="w").pack(side=tk.LEFT)

        tk.Button(
            win, text="OK",
            font=("Helvetica", 11, "bold"),
            bg="#e94560", fg="white", relief=tk.FLAT,
            padx=20, pady=6,
            command=win.destroy
        ).pack(pady=20)

        # Notify main app
        self.app.session_completed(mode, results)
