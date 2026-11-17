@echo off
title Enterprise AML Compliance - Train Models
echo ========================================================
echo Starting AML Detection Model Training Pipeline...
echo ========================================================
cd /d "%~dp0"
python train_model.py
echo.
echo ========================================================
echo Training Complete! Models saved to data/trained_aml_models.pkl
echo ========================================================
pause
