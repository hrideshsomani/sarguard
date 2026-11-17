@echo off
title Enterprise AML - Train 3 Lakh (300,000) Data
echo ========================================================
echo Starting AML Detection Training for 3 Lakh (300,000) Data...
echo Generates fresh transactions until 300,000 records are reached!
echo ========================================================
cd /d "%~dp0"
python continuous_training.py --target-total 300000 --batch-size 50000 --interval 1
echo.
echo ========================================================
echo 3 Lakh Data Training Complete! Models saved to data/trained_aml_models.pkl
echo ========================================================
pause
