@echo off
cd /d "%~dp0"
echo Starting MiniFold Web Portal...
if exist web_ui\server.py (
    python web_ui\server.py
) else (
    echo Error: web_ui\server.py not found!
    pause
)
pause
