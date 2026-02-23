"""
Eye Blink Rate Detector – Movie vs. Document Reading
Main application entry point
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from session_manager import SessionManager
from results import ResultsWindow


class BlinkDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("👁 Eye Blink Rate Detector")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        self.root.configure(bg="#1a1a2e")

        self.session_manager = SessionManager(self)
        self.movie_results = None
        self.document_results = None

        self._build_ui()

    def _build_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#16213e", pady=20)
        header_frame.pack(fill=tk.X)

        tk.Label(
            header_frame,
            text="👁  Eye Blink Rate Detector",
            font=("Helvetica", 26, "bold"),
            fg="#e94560",
            bg="#16213e"
        ).pack()

        tk.Label(
            header_frame,
            text="Compare your blink rate while watching movies vs. reading documents",
            font=("Helvetica", 11),
            fg="#a8a8b3",
            bg="#16213e"
        ).pack(pady=(4, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready — select a session to begin")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            fg="#f5a623",
            bg="#0f3460",
            pady=6
        )
        status_bar.pack(fill=tk.X)

        # Main content
        content = tk.Frame(self.root, bg="#1a1a2e", pady=30)
        content.pack(fill=tk.BOTH, expand=True, padx=40)

        # Session cards row
        cards_frame = tk.Frame(content, bg="#1a1a2e")
        cards_frame.pack(fill=tk.X, pady=10)

        self._build_session_card(
            cards_frame,
            "🎬 Movie Session",
            "Watch a video clip (~60 sec)\nwhile the app tracks your blinks",
            "#e94560",
            "movie",
            side=tk.LEFT
        )

        tk.Frame(cards_frame, bg="#1a1a2e", width=20).pack(side=tk.LEFT)

        self._build_session_card(
            cards_frame,
            "📄 Document Session",
            "Read a PDF or Word document\nwhile the app tracks your blinks",
            "#0f3460",
            "document",
            side=tk.LEFT
        )

        # Results summary row
        self.results_frame = tk.Frame(content, bg="#1a1a2e")
        self.results_frame.pack(fill=tk.X, pady=20)

        self.movie_badge = self._result_badge(self.results_frame, "🎬 Movie", "#e94560")
        self.movie_badge.pack(side=tk.LEFT, padx=10)

        self.doc_badge = self._result_badge(self.results_frame, "📄 Document", "#4a90d9")
        self.doc_badge.pack(side=tk.LEFT, padx=10)

        # Compare button
        self.compare_btn = tk.Button(
            content,
            text="📊  View Comparison Chart",
            font=("Helvetica", 13, "bold"),
            bg="#533483",
            fg="white",
            activebackground="#6a44a0",
            activeforeground="white",
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self._show_comparison
        )
        self.compare_btn.pack(pady=5)

    def _build_session_card(self, parent, title, desc, color, mode, side):
        card = tk.Frame(parent, bg="#16213e", padx=20, pady=20, relief=tk.FLAT, bd=2)
        card.pack(side=side, fill=tk.BOTH, expand=True)

        tk.Label(card, text=title, font=("Helvetica", 15, "bold"),
                 fg=color, bg="#16213e").pack(anchor=tk.W)

        tk.Label(card, text=desc, font=("Helvetica", 10),
                 fg="#a8a8b3", bg="#16213e", justify=tk.LEFT).pack(anchor=tk.W, pady=(6, 14))

        # Status indicator for this session
        status_label = tk.Label(card, text="● Not started", font=("Helvetica", 10),
                                fg="#555577", bg="#16213e")
        status_label.pack(anchor=tk.W, pady=(0, 10))

        btn = tk.Button(
            card,
            text=f"Start {title.split()[1]} Session",
            font=("Helvetica", 11, "bold"),
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            relief=tk.FLAT,
            padx=16,
            pady=8,
            cursor="hand2",
            command=lambda m=mode: self._start_session(m)
        )
        btn.pack(anchor=tk.W)

        if mode == "movie":
            self.movie_btn = btn
            self.movie_status_label = status_label
        else:
            self.doc_btn = btn
            self.doc_status_label = status_label

    def _result_badge(self, parent, label, color):
        frame = tk.Frame(parent, bg="#16213e", padx=14, pady=10)
        tk.Label(frame, text=label, font=("Helvetica", 11, "bold"),
                 fg=color, bg="#16213e").pack()
        result_var = tk.StringVar(value="—")
        tk.Label(frame, textvariable=result_var, font=("Helvetica", 12),
                 fg="white", bg="#16213e").pack()

        if "Movie" in label:
            self.movie_result_var = result_var
        else:
            self.doc_result_var = result_var

        return frame

    def _start_session(self, mode):
        self.session_manager.start_session(mode)

    def update_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    def session_completed(self, mode, results):
        """Called by SessionManager when a session ends."""
        bps = results['blinks_per_second']
        bpm = results['blinks_per_minute']
        total = results['total_blinks']

        if mode == "movie":
            self.movie_results = results
            self.movie_result_var.set(f"{bpm:.1f} blinks/min\n({total} total)")
            self.movie_status_label.config(text="✔ Completed", fg="#4caf50")
            self.movie_btn.config(text="Re-run Movie Session")
        else:
            self.document_results = results
            self.doc_result_var.set(f"{bpm:.1f} blinks/min\n({total} total)")
            self.doc_status_label.config(text="✔ Completed", fg="#4caf50")
            self.doc_btn.config(text="Re-run Document Session")

        if self.movie_results and self.document_results:
            self.compare_btn.config(state=tk.NORMAL)

        self.update_status(f"{mode.capitalize()} session complete — {total} blinks detected")

    def _show_comparison(self):
        if self.movie_results and self.document_results:
            ResultsWindow(self.root, self.movie_results, self.document_results)

    def run(self):
        self.root.mainloop()


def main():
    root = tk.Tk()
    app = BlinkDetectorApp(root)
    app.run()


if __name__ == "__main__":
    main()
