@echo off
cd /d "%~dp0"
echo ============================================================
echo  Starting MediFed-DP Healthcare Dashboard (Green, White, Blue)
echo ============================================================
if not exist .venv (
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call .venv\Scripts\activate
)
python app.py
