@echo off
cd /d %~dp0
python -m streamlit run app/sismo_app_F/sismo_app/app.py --server.headless true --server.port 8501
