@echo off
title Enterprise AML Compliance - 6 Million Records Training Pipeline
echo =========================================================================
echo Starting 6,000,000 (6 Million) Data Generation, Preprocessing and Multi-Model Training...
echo FinCEN / FCA Regulatory Standards - Noise Sanitization and Calibration
echo =========================================================================
cd /d "%~dp0"
python train_6_million.py --target-total 6000000 --chunk-size 500000
echo.
echo Training complete. Press any key to exit.
pause
