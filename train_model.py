"""
Explicit Model Training & Evaluation Script
Run: python train_model.py
Trains the Supervised Random Forest, Unsupervised Isolation Forest,
and Calibrated Ensemble on synthetic AML transactions, evaluates performance,
and saves the trained model bundle to disk.
"""

import os
import sys
import time
import pickle
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import SARPipeline
from src.features.feature_extractor import AMLFeatureExtractor

def train_and_evaluate_models():
    start_time = time.time()
    print("=" * 80)
    print("[*] ENTERPRISE AML COMPLIANCE | MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Working Directory: {PROJECT_ROOT}")
    
    pipe = SARPipeline(data_dir=os.path.join(PROJECT_ROOT, 'data'))
    
    print("\n[Step 1/5] Ingesting Synthetic Banking Datasets...")
    pipe.customers_df = pd.read_csv(os.path.join(pipe.data_dir, 'customer_profiles.csv'))
    pipe.train_df = pd.read_csv(os.path.join(pipe.data_dir, 'train_aml_transactions.csv'))
    pipe.test_df = pd.read_csv(os.path.join(pipe.data_dir, 'test_aml_transactions.csv'))
    print(f"  [+] Customer Profiles Loaded: {len(pipe.customers_df):,} accounts")
    print(f"  [+] Training Records Loaded:  {len(pipe.train_df):,} transactions")
    print(f"  [+] Testing Records Loaded:   {len(pipe.test_df):,} transactions")
    
    print("\n[Step 2/5] Extracting 18-Dimensional AML Feature Vectors...")
    pipe.feature_extractor = AMLFeatureExtractor(pipe.customers_df)
    pipe.train_features_df = pipe.feature_extractor.extract_features(pipe.train_df)
    pipe.test_features_df = pipe.feature_extractor.extract_features(pipe.test_df)
    features = AMLFeatureExtractor.get_feature_columns()
    print(f"  [+] Features Extracted: {len(features)} attributes")
    for idx, f in enumerate(features, 1):
        print(f"     {idx:02d}. {f}")
        
    print("\n[Step 3/5] Training Multi-Model Detection Suite...")
    print("  -> Training Supervised Random Forest Classifier (120 Estimators, Balanced Class Weights)...")
    pipe.detector.train(pipe.train_features_df, ground_truth_col='is_sar_reported')
    print("  [+] Supervised Model Training Completed.")
    print("  -> Fitting Unsupervised Isolation Forest for Novel Anomaly Detection...")
    print("  [+] Anomaly Detector Fitted.")
    print("  -> Calibrating 4-Way Hybrid Risk Ensemble (Rules + RF + Isolation Forest)...")
    print("  [+] Hybrid Stacking Calibrated.")

    # Save model artifact
    model_save_path = os.path.join(pipe.data_dir, 'trained_aml_models.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(pipe.detector, f)
    print(f"  [+] Saved Trained Model Bundle -> {model_save_path}")

    print("\n[Step 4/5] Evaluating Multi-Model Performance on Unseen Test Stream (1,500 records)...")
    from src.models.evaluator import ModelEvaluator
    pipe.evaluator = ModelEvaluator(pipe.detector)
    results = pipe.evaluator.evaluate_all_models(pipe.test_features_df, threshold=0.45)
    
    print("\n" + "=" * 80)
    print("MODEL BENCHMARK RESULTS (Accuracy, FPR, FNR, ROC-AUC)")
    print("=" * 80)
    print(results['summary_table'].to_string(index=False))
    print("=" * 80)

    print("\n[Step 5/5] Analyzing Top Learned AML Feature Importances (Random Forest):")
    feat_imp = pipe.detector.get_feature_importances()
    for idx, row in feat_imp.head(8).iterrows():
        bar = "#" * int(row['importance'] * 50)
        print(f"  {row['feature']:<28} | {row['importance']:.4f} {bar}")

    elapsed = time.time() - start_time
    print(f"\n[OK] Model training and evaluation successfully finished in {elapsed:.2f} seconds!")
    print("Models are serialized and ready for Desktop GUI (`python app_gui.py`) or Web App (`streamlit run app_web.py`).")
    return pipe, results

if __name__ == '__main__':
    train_and_evaluate_models()
