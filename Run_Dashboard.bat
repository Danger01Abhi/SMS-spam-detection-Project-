@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m streamlit run "dashboard_ui_modern.py"
) else (
    python -m streamlit run "dashboard_ui_modern.py"
)

pause