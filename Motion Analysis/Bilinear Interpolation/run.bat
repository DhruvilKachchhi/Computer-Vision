@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM  run.bat  —  Activate venv and launch the tracker
REM
REM  Usage:
REM    run.bat                                  (auto-detect video or synthetic)
REM    run.bat --video path\to\clip.mp4         (single video)
REM    run.bat --video a.mp4 --video b.mp4      (multiple videos)
REM    run.bat --video clip.mp4 --frame1 5 --frame2 6
REM    run.bat --no-theory                      (skip theory print)
REM    run.bat --outdir results                 (custom output folder)
REM
REM  Outputs saved to output\ :
REM    Static PNGs : lk_theory  lk_tracking  lk_bilinear_detail  lk_errors
REM                  lk_dense_hsv  lk_quiver  lk_eigenvalue_scatter
REM    Video  MP4s : vid_dense_hsv  vid_trails  vid_quiver
REM ============================================================

if not exist venv\Scripts\activate.bat (
    echo.
    echo  [ERROR] Virtual environment not found.
    echo          Please run setup.bat first.
    echo.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

REM ── Force UTF-8 so theory Unicode renders correctly in any terminal ───────
set PYTHONIOENCODING=utf-8
chcp 65001 >nul 2>&1

REM ── If the user supplied any arguments, pass them straight through ────────
if not "%~1"=="" (
    echo  Running: python lucas_kanade_tracker.py %*
    echo.
    python lucas_kanade_tracker.py %*
    goto :done
)

REM ── No arguments: scan current directory for video files ─────────────────
echo  No arguments supplied.  Scanning for video files ...
echo.

set VIDEO_ARGS=
set VIDEO_COUNT=0

for %%F in (*.mp4 *.avi *.mov *.mkv) do (
    set /a VIDEO_COUNT+=1
    set VIDEO_ARGS=!VIDEO_ARGS! --video "%%F"
    echo    Detected: %%F
)

if %VIDEO_COUNT%==0 (
    echo    No video files found in current directory.
    echo    Falling back to synthetic frames ^(no .mp4/.avi/.mov/.mkv present^).
    echo    Tip: copy your video here and re-run, or use:
    echo         run.bat --video path\to\your_clip.mp4
    echo.
    python lucas_kanade_tracker.py
    goto :done
)

echo.
echo  Found %VIDEO_COUNT% video(s).  Starting ...
echo  Command: python lucas_kanade_tracker.py !VIDEO_ARGS!
echo.
python lucas_kanade_tracker.py !VIDEO_ARGS!

:done
if errorlevel 1 (
    echo.
    echo  [ERROR] Script exited with an error ^(code %errorlevel%^).
    pause
)

endlocal
