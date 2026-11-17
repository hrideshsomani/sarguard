"""
IBM AML & Multi-Dataset Downloader & Local Ingestion Helper
URL: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

This utility helps you manage local offline copies of:
1. IBM Transactions for Anti-Money Laundering (AMLWorld)
2. PaySim Financial Mobile Money Fraud/AML Dataset
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOCAL_DIR = r"D:\kaggle_ibm" if os.path.exists("D:\\") else os.path.join(PROJECT_ROOT, "data", "kaggle_ibm")

def print_dataset_instructions():
    print("=" * 80)
    print("  AML DATASETS CONFIGURATION & INGESTION GUIDE")
    print("=" * 80)
    print(f"Target Offline Folder: {DEFAULT_LOCAL_DIR}")
    print("\n[Option A] Automatic Online Streaming (Default - Already Active):")
    print("  * The pipeline automatically streams IBM AML and PaySim datasets from")
    print("    verified repositories directly into the 60,000,000 transaction engine.")
    print("  * No manual downloading or API keys are required.")
    print("\n[Option B] Local Kaggle Download (For Maximum Offline Speed):")
    print("  1. Download the IBM AML dataset from:")
    print("     https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml")
    print(f"  2. Extract any of the CSV files (e.g. HI-Small_Trans.csv, LI-Small_Trans.csv)")
    print(f"     into: {DEFAULT_LOCAL_DIR}")
    print("  3. Run: python train_60_million.py --local-kaggle-dir " + DEFAULT_LOCAL_DIR)
    print("=" * 80)

if __name__ == '__main__':
    print_dataset_instructions()
