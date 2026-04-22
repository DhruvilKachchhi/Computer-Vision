"""
================================================================================
  Pixel → Millimetre Conversion Script
  Facial Analysis Pipeline — Biometric Spatial Calibration
================================================================================
Author : Data Analysis Utility
Version: 1.0.0

Description:
    Reads the facial-analysis CSV produced by facial_analysis.py and converts
    all spatial pixel measurements into real-world millimetres (mm) using a
    per-row biometric scale factor derived from the known iris diameter.

Calibration Method (Iris-Based Biometric Constant):
    • Reference constant : Iris Diameter = 11.8 mm  (human population average)
    • Pixel proxy        : avg_eye_height column
      (the vertical opening of the eye — upper-to-lower lid — closely
       approximates the visible iris diameter when the eye is fully open)
    • Per-row Scale Factor:
          Scale_Factor = IRIS_DIAMETER_MM / avg_eye_height_px
                       = 11.8 / avg_eye_height
    • Conversion:
          <feature>_mm = <feature>_px × Scale_Factor

Columns converted (12 spatial features):
    face_width, face_height,
    left_eye_width, left_eye_height,
    right_eye_width, right_eye_height,
    avg_eye_width, avg_eye_height,
    nose_width, nose_height,
    mouth_width, mouth_height

Output (two CSV files are written):
    1. Full CSV  — all original columns PLUS 12 new *_mm columns appended
                   facial_analysis_data_mm_<timestamp>.csv
    2. mm-Only CSV — metadata columns + ONLY the 12 *_mm columns (no pixel columns)
                   facial_analysis_data_mm_only_<timestamp>.csv

       Metadata columns kept in the mm-only CSV:
           filename, duration_sec, fps, resolution, total_frames,
           frames_with_face, total_blinks, blink_rate_per_s

Usage:
    python convert_to_mm.py                              # auto-detects latest CSV
    python convert_to_mm.py --input Videos/my_data.csv  # explicit input file
    python convert_to_mm.py --input Videos/my_data.csv --output Videos/out.csv
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import argparse
import datetime
import sys
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas is not installed.  Run:  pip install pandas")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Biometric constant: average human iris diameter in millimetres.
# Source: Bekerman I. et al., "Sizes of Ocular Structures", IOVS 2014.
IRIS_DIAMETER_MM: float = 11.8

# Column used as the pixel proxy for the iris diameter.
# avg_eye_height ≈ visible vertical aperture of the eye ≈ iris diameter
# when the subject's eyes are fully open.
IRIS_PIXEL_COLUMN: str = "avg_eye_height"

# Spatial feature columns to convert (pixels → millimetres).
SPATIAL_COLUMNS = [
    "face_width",
    "face_height",
    "left_eye_width",
    "left_eye_height",
    "right_eye_width",
    "right_eye_height",
    "avg_eye_width",
    "avg_eye_height",
    "nose_width",
    "nose_height",
    "mouth_width",
    "mouth_height",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — AUTO-DETECT THE MOST RECENT CSV IN ./Videos
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_csv(search_dir: Path) -> Path:
    """
    Return the most recently modified raw facial_analysis_data_*.csv in search_dir.

    Excludes already-converted files whose names contain '_mm' so that
    re-running the script always picks up the original pixel-unit data.

    Args:
        search_dir: Directory to scan.

    Returns:
        Path to the newest matching CSV.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    candidates = sorted(
        (
            p for p in search_dir.glob("facial_analysis_data_*.csv")
            if "_mm" not in p.stem          # skip already-converted files
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No raw 'facial_analysis_data_*.csv' file found in: {search_dir.resolve()}\n"
            f"(Files whose names contain '_mm' are skipped — they are already converted.)"
        )
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# CORE CONVERSION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def convert_pixels_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append millimetre-converted columns to the facial analysis DataFrame.

    Conversion logic (per row):
        1.  scale_factor = IRIS_DIAMETER_MM / avg_eye_height
            → maps 1 pixel to its real-world mm equivalent for this subject/frame
        2.  <col>_mm = <col>_px * scale_factor   for every spatial column

    Args:
        df: Raw DataFrame loaded from the facial_analysis CSV.

    Returns:
        The same DataFrame with 12 new *_mm columns appended.

    Raises:
        ValueError : If IRIS_PIXEL_COLUMN is missing or contains zero/NaN values.
    """
    # ── Validate iris proxy column ────────────────────────────────────────────
    if IRIS_PIXEL_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{IRIS_PIXEL_COLUMN}' not found in the dataset.\n"
            f"Available columns: {list(df.columns)}"
        )

    if df[IRIS_PIXEL_COLUMN].isnull().any():
        bad_rows = df[df[IRIS_PIXEL_COLUMN].isnull()].index.tolist()
        raise ValueError(
            f"NaN values detected in '{IRIS_PIXEL_COLUMN}' at rows: {bad_rows}.\n"
            "Cannot compute a scale factor for these rows."
        )

    zero_mask = df[IRIS_PIXEL_COLUMN] == 0
    if zero_mask.any():
        bad_rows = df[zero_mask].index.tolist()
        raise ValueError(
            f"Zero values detected in '{IRIS_PIXEL_COLUMN}' at rows: {bad_rows}.\n"
            "Division by zero would produce an invalid scale factor."
        )

    # ── Validate spatial columns ──────────────────────────────────────────────
    missing_cols = [c for c in SPATIAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following spatial columns are missing from the dataset: {missing_cols}"
        )

    # ── Per-row Scale Factor ──────────────────────────────────────────────────
    # scale_factor[i] = 11.8 / avg_eye_height[i]
    # This is a pandas Series with one scalar per row — vectorised, no loops.
    scale_factor: pd.Series = IRIS_DIAMETER_MM / df[IRIS_PIXEL_COLUMN]

    print(f"\n{'─' * 60}")
    print(f"  Biometric Calibration — Iris Reference Constant")
    print(f"{'─' * 60}")
    print(f"  IRIS_DIAMETER_MM    : {IRIS_DIAMETER_MM} mm")
    print(f"  Iris pixel proxy    : '{IRIS_PIXEL_COLUMN}'")
    print(f"  Scale factor range  : "
          f"{scale_factor.min():.6f} – {scale_factor.max():.6f} mm/px")
    print(f"  Scale factor mean   : {scale_factor.mean():.6f} mm/px")
    print(f"{'─' * 60}\n")

    # ── Apply conversion ──────────────────────────────────────────────────────
    result_df = df.copy()

    for col in SPATIAL_COLUMNS:
        mm_col = f"{col}_mm"
        result_df[mm_col] = (df[col] * scale_factor).round(4)
        print(f"  ✔  {col:22s}  →  {mm_col}")

    print()
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_conversion_summary(original_df: pd.DataFrame, converted_df: pd.DataFrame) -> None:
    """
    Print a side-by-side comparison table showing pixel vs mm values per video.

    Args:
        original_df  : DataFrame before conversion (pixel values).
        converted_df : DataFrame after conversion (includes _mm columns).
    """
    print(f"\n{'═' * 90}")
    print("  CONVERSION SUMMARY — Pixel vs Millimetre Values")
    print(f"{'═' * 90}")

    # Header row
    header = f"  {'Filename':<20} {'Scale Factor':>13}  "
    header += f"{'Face W':>8} {'Face W mm':>10}  "
    header += f"{'Eye H':>7} {'Eye H mm':>9}  "
    header += f"{'Nose W':>7} {'Nose W mm':>10}  "
    header += f"{'Mouth W':>7} {'Mouth W mm':>10}"
    print(header)
    print(f"  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*6}  {'-'*9}")

    scale_series = IRIS_DIAMETER_MM / original_df[IRIS_PIXEL_COLUMN]

    for i, row in converted_df.iterrows():
        sf = scale_series.iloc[i] if isinstance(i, int) else scale_series[i]
        line = f"  {str(row['filename']):<20} {sf:>13.6f}  "
        line += f"{row['face_width']:>8.2f} {row['face_width_mm']:>10.4f}  "
        line += f"{row['avg_eye_height']:>7.2f} {row['avg_eye_height_mm']:>9.4f}  "
        line += f"{row['nose_width']:>7.2f} {row['nose_width_mm']:>10.4f}  "
        line += f"{row['mouth_width']:>7.2f} {row['mouth_width_mm']:>10.4f}"
        print(line)

    print(f"{'─' * 90}")

    # Aggregate stats for the mm columns
    mm_cols = [f"{c}_mm" for c in SPATIAL_COLUMNS]
    print("\n  Aggregate statistics for converted (mm) columns:\n")
    print(converted_df[mm_cols].describe().round(4).to_string(
        index=True,
        justify="right",
    ))
    print(f"\n{'═' * 90}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace with:
            input  : str | None — path to input CSV (auto-detected if omitted)
            output : str | None — path for output CSV (auto-named if omitted)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Convert pixel measurements in a facial-analysis CSV to millimetres "
            "using an iris-diameter biometric scale factor (11.8 mm)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_to_mm.py
  python convert_to_mm.py --input Videos/facial_analysis_data_20260422_012047.csv
  python convert_to_mm.py --input Videos/data.csv --output Videos/data_mm.csv
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help=(
            "Path to the input CSV file.  "
            "If omitted, the script auto-detects the most recent "
            "'facial_analysis_data_*.csv' in ./Videos."
        ),
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=(
            "Path for the output CSV file.  "
            "Defaults to <input_dir>/facial_analysis_data_mm_<timestamp>.csv."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    End-to-end orchestration:
        1. Resolve input CSV path (explicit or auto-detected).
        2. Load CSV into a pandas DataFrame.
        3. Compute per-row scale factors and append *_mm columns.
        4. Print conversion summary to console.
        5. Export full CSV  : original columns + 12 new *_mm columns.
        6. Export mm-only CSV : metadata columns + 12 *_mm columns only
                                (no pixel spatial columns).
    """
    args = parse_arguments()

    # ── Resolve input path ────────────────────────────────────────────────────
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] Input file not found: {input_path.resolve()}")
            sys.exit(1)
    else:
        videos_dir = Path(__file__).parent / "Videos"
        try:
            input_path = find_latest_csv(videos_dir)
            print(f"[INFO]  Auto-detected input CSV: {input_path}")
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            sys.exit(1)

    # ── Resolve output paths ──────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.output:
        full_output_path = Path(args.output)
    else:
        full_output_path = input_path.parent / f"facial_analysis_data_mm_{timestamp}.csv"

    # mm-only path: always auto-named next to the input file
    mm_only_output_path = input_path.parent / f"facial_analysis_data_mm_only_{timestamp}.csv"

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"\n[INFO]  Loading:  {input_path.resolve()}")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"[ERROR] Failed to read CSV: {exc}")
        sys.exit(1)

    print(f"[INFO]  Shape   : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"[INFO]  Columns : {list(df.columns)}")

    # ── Convert pixels → mm ───────────────────────────────────────────────────
    try:
        converted_df = convert_pixels_to_mm(df)
    except ValueError as exc:
        print(f"[ERROR] Conversion failed: {exc}")
        sys.exit(1)

    # ── Display summary ───────────────────────────────────────────────────────
    print_conversion_summary(df, converted_df)

    # ── Non-spatial metadata columns (kept in both output files) ──────────────
    # These are every column in the original CSV that is NOT a spatial pixel feature.
    metadata_cols = [c for c in df.columns if c not in SPATIAL_COLUMNS]

    # ── mm-only column list: metadata + the 12 *_mm columns ──────────────────
    mm_cols        = [f"{c}_mm" for c in SPATIAL_COLUMNS]
    mm_only_cols   = metadata_cols + mm_cols

    # ── Export 1: Full CSV (original + appended *_mm columns) ─────────────────
    try:
        converted_df.to_csv(full_output_path, index=False)
        print(f"[INFO]  ✔  Full CSV saved     : {full_output_path.resolve()}")
        print(f"[INFO]     Columns : {len(converted_df.columns)}  "
              f"({len(df.columns)} original  +  "
              f"{len(converted_df.columns) - len(df.columns)} *_mm columns)")
    except Exception as exc:
        print(f"[ERROR] Failed to write full CSV: {exc}")
        sys.exit(1)

    # ── Export 2: mm-only CSV (metadata + *_mm columns, no pixel columns) ─────
    try:
        mm_only_df = converted_df[mm_only_cols]
        mm_only_df.to_csv(mm_only_output_path, index=False)
        print(f"\n[INFO]  ✔  mm-only CSV saved  : {mm_only_output_path.resolve()}")
        print(f"[INFO]     Columns : {len(mm_only_df.columns)}  "
              f"({len(metadata_cols)} metadata  +  {len(mm_cols)} *_mm columns)")
        print(f"[INFO]     Columns included: {list(mm_only_df.columns)}\n")
    except Exception as exc:
        print(f"[ERROR] Failed to write mm-only CSV: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
