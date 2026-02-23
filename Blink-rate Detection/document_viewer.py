"""
document_viewer.py – Renders the first page of a PDF or DOCX inside a Tkinter window.
PDF: uses PyMuPDF (fitz)
DOCX: converts to PDF via LibreOffice headless, then renders with PyMuPDF
"""

import tkinter as tk
from tkinter import messagebox
import subprocess
import tempfile
import os
import sys
from PIL import Image, ImageTk
import fitz  # PyMuPDF


class DocumentViewer:
    """Opens a Toplevel window displaying page 1 of a PDF or DOCX file."""

    def __init__(self, parent, filepath):
        self.parent = parent
        self.filepath = filepath
        self._tmp_pdf = None  # temp file for DOCX conversion

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".docx":
            pdf_path = self._convert_docx_to_pdf(filepath)
        elif ext == ".pdf":
            pdf_path = filepath
        else:
            raise ValueError(f"Unsupported document type: {ext}")

        img = self._render_page(pdf_path)

        self.win = tk.Toplevel(parent)
        self.win.title(f"📄 Document — {os.path.basename(filepath)}")
        self.win.configure(bg="#f5f5f0")
        self.win.protocol("WM_DELETE_WINDOW", lambda: None)  # prevent close during session

        # Scrollable canvas
        frame = tk.Frame(self.win, bg="#f5f5f0")
        frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(frame, bg="#f5f5f0", highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a frame inside canvas for the image
        self._img_frame = tk.Frame(canvas, bg="#f5f5f0")
        self._img_frame_id = canvas.create_window((0, 0), window=self._img_frame, anchor=tk.NW)

        # Display image
        self._photo = ImageTk.PhotoImage(img)
        self._img_label = tk.Label(self._img_frame, image=self._photo, bg="#f5f5f0")
        self._img_label.pack()

        # Configure scroll region
        self._img_frame.update_idletasks()  # Update geometry
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Make canvas scrollable with mouse wheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # Update scroll region when window is resized
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self._img_frame.bind("<Configure>", _on_frame_configure)

        # Size window to fit page (max 900x750)
        w = min(img.width + 20, 900)
        h = min(img.height + 20, 750)
        self.win.geometry(f"{w}x{h}")
        self.win.lift()

    def _convert_docx_to_pdf(self, docx_path):
        """Convert DOCX to PDF using LibreOffice headless."""
        tmp_dir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [
                    "libreoffice", "--headless",
                    "--convert-to", "pdf",
                    "--outdir", tmp_dir,
                    docx_path
                ],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"LibreOffice conversion failed:\n{result.stderr}"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "LibreOffice not found. Install it to support DOCX files.\n"
                "(sudo apt install libreoffice  or  brew install libreoffice)"
            )

        base = os.path.splitext(os.path.basename(docx_path))[0]
        pdf_path = os.path.join(tmp_dir, base + ".pdf")
        if not os.path.exists(pdf_path):
            raise RuntimeError("LibreOffice conversion produced no output file.")

        self._tmp_pdf = pdf_path
        return pdf_path

    def _render_page(self, pdf_path, page_num=0, zoom=1.5):
        """Render a PDF page to a PIL Image."""
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            page_num = 0
        page = doc[page_num]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()

        # Convert pixmap to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    def close(self):
        if self._tmp_pdf and os.path.exists(self._tmp_pdf):
            try:
                os.remove(self._tmp_pdf)
            except Exception:
                pass
        if hasattr(self, 'win') and self.win.winfo_exists():
            self.win.protocol("WM_DELETE_WINDOW", self.win.destroy)
            self.win.destroy()
