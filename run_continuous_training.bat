@echo off
title Enterprise AML - Continuous Data Generation ^& Training
echo ========================================================
echo Starting Continuous Data Generation ^& AML Training...
echo Generates fresh transactions, appends to dataset, and retrains!
echo ========================================================
cd /d "%~dp0"
python continuous_training.py --cycles 5 --batch-size 6000 --interval 2
echo.
pause
