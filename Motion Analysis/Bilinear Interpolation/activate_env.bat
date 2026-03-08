@echo off
REM ============================================================
REM  activate_env.bat  —  Open a shell with the venv active
REM
REM  Use this to run custom commands, e.g.:
REM    python lucas_kanade_tracker.py --video clip.mp4 --outdir my_out
REM    python lucas_kanade_tracker.py --video a.mp4 --video b.mp4
REM    python lucas_kanade_tracker.py --video clip.mp4 --frame1 25 --frame2 26
REM ============================================================
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
echo.
echo  Virtual environment activated.
echo  Type "deactivate" to exit.
echo.
echo  Quick examples:
echo    python lucas_kanade_tracker.py --video "Video 1.mp4"
echo    python lucas_kanade_tracker.py --video clip.mp4 --outdir results
echo    python lucas_kanade_tracker.py --video clip.mp4 --no-theory
echo.
cmd /k
