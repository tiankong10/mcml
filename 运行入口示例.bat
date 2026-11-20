@echo off
chcp 65001 >nul
cd /d "%~dp0src"
python run_final_paper.py %*
if errorlevel 1 (
  echo.
  echo [ERROR] Check the Python error printed above.
)
pause
