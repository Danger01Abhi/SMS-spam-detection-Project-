
@echo off
title SMS Spam Detection Project

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project virtual environment was not found.
    echo Check that the .venv folder exists.
    pause
    exit /b 1
)

if not exist "app.py" (
    echo ERROR: app.py was not found in the project folder.
    pause
    exit /b 1
)

echo Starting SMS Spam Detection...
echo Your browser will open when Streamlit starts.
echo Press Ctrl+C in this window to stop the application.
echo.

".venv\Scripts\python.exe" -m streamlit run "app.py"

pause
