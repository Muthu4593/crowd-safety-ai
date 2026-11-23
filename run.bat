@echo off
cd /d "%~dp0"
echo Installing requirements, please wait...
python -m pip install -r requirements.txt
echo.
echo Starting Crowd Safety AI. A browser tab will open.
python -m streamlit run app.py
pause
