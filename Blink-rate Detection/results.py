"""
results.py – Displays session stats and a side-by-side comparison bar chart.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


MOVIE_COLOR = "#e94560"
DOC_COLOR = "#4a90d9"


class ResultsWindow:
    """Full comparison window with stats table and bar chart."""

    def __init__(self, parent, movie_results, doc_results):
        self.win = tk.Toplevel(parent)
        self.win.title("📊 Blink Rate Comparison")
        self.win.geometry("900x620")
        self.win.configure(bg="#1a1a2e")
        self.win.lift()

        self._build(movie_results, doc_results)

    def _build(self, movie, doc):
        win = self.win

        # Title
        tk.Label(win, text="📊  Blink Rate Comparison",
                 font=("Helvetica", 18, "bold"), fg="white", bg="#1a1a2e").pack(pady=(18, 2))
        tk.Label(win,
                 text="Higher blink rates during document reading are normal — the eye works harder.",
                 font=("Helvetica", 10), fg="#a8a8b3", bg="#1a1a2e").pack()

        # Main area: chart left, table right
        main = tk.Frame(win, bg="#1a1a2e")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # --- Chart ---
        chart_frame = tk.Frame(main, bg="#1a1a2e")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_chart(chart_frame, movie, doc)

        # --- Stats table ---
        table_frame = tk.Frame(main, bg="#16213e", padx=16, pady=16, relief=tk.FLAT)
        table_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        self._build_table(table_frame, movie, doc)

        # --- Insight ---
        insight = self._generate_insight(movie, doc)
        tk.Label(win, text=insight, font=("Helvetica", 10, "italic"),
                 fg="#f5a623", bg="#1a1a2e", wraplength=860, justify=tk.CENTER).pack(pady=(0, 6))

        tk.Button(win, text="Close", font=("Helvetica", 11, "bold"),
                  bg="#e94560", fg="white", relief=tk.FLAT, padx=20, pady=6,
                  command=win.destroy).pack(pady=8)

    def _build_chart(self, parent, movie, doc):
        fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.8))
        fig.patch.set_facecolor("#1a1a2e")

        metrics = {
            "Blinks/min": (movie["blinks_per_minute"], doc["blinks_per_minute"]),
            "Blinks/sec": (movie["blinks_per_second"], doc["blinks_per_second"]),
        }

        for ax, (metric, (m_val, d_val)) in zip(axes, metrics.items()):
            ax.set_facecolor("#16213e")
            bars = ax.bar(
                ["🎬 Movie", "📄 Document"],
                [m_val, d_val],
                color=[MOVIE_COLOR, DOC_COLOR],
                width=0.5,
                edgecolor="none"
            )
            ax.set_title(metric, color="white", fontsize=11, pad=8)
            ax.set_ylabel(metric, color="#a8a8b3", fontsize=9)
            ax.tick_params(colors="white", labelsize=9)
            ax.spines[:].set_color("#333355")
            for spine in ax.spines.values():
                spine.set_alpha(0.4)
            ax.yaxis.label.set_color("#a8a8b3")
            ax.set_ylim(0, max(m_val, d_val) * 1.3 + 0.5)

            # Value labels on bars
            for bar, val in zip(bars, [m_val, d_val]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(m_val, d_val) * 0.03,
                    f"{val:.2f}",
                    ha="center", va="bottom", color="white", fontsize=10, fontweight="bold"
                )

        plt.tight_layout(pad=2.0)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().configure(bg="#1a1a2e", highlightthickness=0)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_table(self, parent, movie, doc):
        tk.Label(parent, text="Session Details", font=("Helvetica", 12, "bold"),
                 fg="white", bg="#16213e").pack(anchor=tk.W, pady=(0, 10))

        rows = [
            ("Metric", "🎬 Movie", "📄 Document"),
            ("Total Blinks", str(movie["total_blinks"]), str(doc["total_blinks"])),
            ("Blinks / Minute", f"{movie['blinks_per_minute']:.1f}", f"{doc['blinks_per_minute']:.1f}"),
            ("Blinks / Second", f"{movie['blinks_per_second']:.3f}", f"{doc['blinks_per_second']:.3f}"),
            ("Duration (s)", f"{movie['duration_seconds']:.0f}", f"{doc['duration_seconds']:.0f}"),
            ("EAR Threshold", f"{movie['ear_threshold']:.3f}", f"{doc['ear_threshold']:.3f}"),
        ]

        for i, (label, m_val, d_val) in enumerate(rows):
            row_bg = "#16213e" if i % 2 == 0 else "#1e2a4a"
            row = tk.Frame(parent, bg=row_bg)
            row.pack(fill=tk.X, pady=1)

            is_header = i == 0
            font = ("Helvetica", 10, "bold") if is_header else ("Helvetica", 10)
            fg_label = "#a8a8b3" if is_header else "#cccccc"

            tk.Label(row, text=label, font=font, fg=fg_label, bg=row_bg,
                     width=16, anchor="w", padx=6, pady=4).pack(side=tk.LEFT)
            tk.Label(row, text=m_val, font=font, fg=MOVIE_COLOR if not is_header else "#e94560",
                     bg=row_bg, width=10, anchor="center").pack(side=tk.LEFT)
            tk.Label(row, text=d_val, font=font, fg=DOC_COLOR if not is_header else "#4a90d9",
                     bg=row_bg, width=10, anchor="center").pack(side=tk.LEFT)

        # Delta row
        if movie["blinks_per_minute"] > 0 and doc["blinks_per_minute"] > 0:
            delta = doc["blinks_per_minute"] - movie["blinks_per_minute"]
            pct = (delta / movie["blinks_per_minute"]) * 100
            sign = "+" if delta >= 0 else ""
            color = "#4caf50" if delta >= 0 else "#e94560"

            delta_frame = tk.Frame(parent, bg="#16213e")
            delta_frame.pack(fill=tk.X, pady=(10, 0))
            tk.Label(delta_frame, text="Δ Doc vs Movie:",
                     font=("Helvetica", 10, "bold"), fg="#a8a8b3",
                     bg="#16213e", anchor="w", padx=6).pack(side=tk.LEFT)
            tk.Label(delta_frame, text=f"{sign}{delta:.1f} bpm ({sign}{pct:.0f}%)",
                     font=("Helvetica", 10, "bold"), fg=color, bg="#16213e").pack(side=tk.LEFT)

    def _generate_insight(self, movie, doc):
        m_bpm = movie["blinks_per_minute"]
        d_bpm = doc["blinks_per_minute"]
        normal_range = (12, 20)

        insights = []

        if d_bpm > m_bpm:
            diff = d_bpm - m_bpm
            insights.append(
                f"Your blink rate was {diff:.1f} bpm higher during document reading — "
                f"consistent with research showing increased cognitive focus suppresses blinking."
            )
        elif m_bpm > d_bpm:
            diff = m_bpm - d_bpm
            insights.append(
                f"Interestingly, your movie blink rate was {diff:.1f} bpm higher — "
                f"you may have been especially focused while reading."
            )
        else:
            insights.append("Your blink rate was similar in both conditions.")

        if d_bpm < normal_range[0]:
            insights.append(
                f"⚠️  Your document blink rate ({d_bpm:.1f} bpm) is below the normal range "
                f"({normal_range[0]}–{normal_range[1]} bpm) — consider taking regular eye breaks."
            )

        return "  |  ".join(insights)
