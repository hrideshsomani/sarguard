@echo off
TITLE Enterprise AML Compliance - 60 Million Transactions SAR Training & Verification Engine
echo ====================================================================================================
echo  ENTERPRISE AML COMPLIANCE | 60 MILLION TRANSACTION VERIFICATION & CONTINUOUS TRAINING PIPELINE
echo  Multi-Source Ingestion: IBM Transactions (AMLWorld), PaySim Mobile Money, Synthetic Typologies
echo ====================================================================================================

cd /d "%~dp0"

echo [+] Verifying Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

echo [+] Launching 60 Million Transactions Pipeline...
python train_60_million.py --target-total 60000000 --chunk-size 500000 --resume

echo.
echo [OK] Execution finished.
pause
