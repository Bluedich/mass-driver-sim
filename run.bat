@echo off
cd /d "%~dp0"

if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing / updating dependencies...
pip install -q -r requirements.txt

echo Starting Lunar Mass Driver Sim at http://localhost:8050
python app.py
