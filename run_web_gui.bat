@echo off
title Enterprise AML Compliance - Streamlit Web Dashboard
echo ========================================================
echo Starting Enterprise AML AI SAR Narrative Generator (Web Dashboard)...
echo ========================================================
cd /d "%~dp0"
python -m streamlit run src/ui/web_app.py
pause
