@echo off
cd /d "%~dp0"
python -m streamlit run app/sismo_app_F/sismo_app/app.py
pause
