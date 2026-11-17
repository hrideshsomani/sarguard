"""
Enterprise AML Compliance - 6 Million (6,000,000) Data Generation & Training Pipeline
Run: python train_6_million.py
Generates synthetic banking transactions up to 6,000,000 total records,
injects realistic synthetic noise, sanitizes via 6-stage AMLDataPreprocessor,
trains the multi-model detection suite, and benchmarks Accuracy, FPR, FNR, and ROC-AUC.
"""

import os
import sys
import time
import pickle
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import SARPipeline
from src.data.synthetic_generator import generate_incremental_transactions
from src.features.feature_extractor import AMLFeatureExtractor
from src.models.evaluator import ModelEvaluator

def run_6_million_pipeline(target_total=6000000, chunk_size=500000, suspicious_ratio=0.50):
    start_time = time.time()
    print("=" * 105)
    print("[*] ENTERPRISE AML COMPLIANCE | 6,000,000 (6 MILLION) DATA PIPELINE & MULTI-MODEL TRAINING")
    print("=" * 105)
    print(f"Working Directory:    {PROJECT_ROOT}")
    print(f"Target Total Records: {target_total:,} Records ({target_total // 100000} Lakhs)")
    print(f"Chunk Batch Size:     {chunk_size:,} Records per Batch")
    print("=" * 105)

    pipe = SARPipeline(data_dir=os.path.join(PROJECT_ROOT, 'data'))
    train_path = os.path.join(pipe.data_dir, 'train_aml_transactions.csv')
    cust_path = os.path.join(pipe.data_dir, 'customer_profiles.csv')
    test_path = os.path.join(pipe.data_dir, 'test_aml_transactions.csv')
    
    print("\n[Step 1/3] Initializing Customer KYC Profiles & Test Streams...")
    if not os.path.exists(cust_path) or not os.path.exists(test_path):
        pipe.initialize_data_and_models()
    else:
        pipe.customers_df = pd.read_csv(cust_path)
        pipe.test_df = pd.read_csv(test_path)
        pipe.feature_extractor = AMLFeatureExtractor(pipe.customers_df)
        pipe.test_features_df = pipe.feature_extractor.extract_features(pipe.test_df)
    
    initial_count = 0
    last_tail = None
    if os.path.exists(train_path):
        try:
            with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
                initial_count = sum(1 for _ in f) - 1
            if initial_count > 0:
                last_tail = pd.read_csv(train_path, nrows=50)
        except Exception:
            initial_count = 0
            
    print(f"  [+] Existing Training Records on Disk: {initial_count:,}")
    remaining_to_generate = max(0, target_total - initial_count)
    print(f"  [+] Additional Records to Generate:    {remaining_to_generate:,}\n")

    num_chunks = int(np.ceil(remaining_to_generate / chunk_size)) if remaining_to_generate > 0 else 0
    cumulative_count = initial_count
    
    print("-" * 105)
    header = f"{'Chunk':<7} | {'Total Records':<15} | {'Progress':<10} | {'Added':<10} | {'Accuracy':<9} | {'FPR':<8} | {'FNR':<8} | {'Precision':<10} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 105)

    ens_m = None
    pipe.evaluator = ModelEvaluator(pipe.detector)

    if num_chunks == 0:
        print("[!] Target records count already achieved on disk. Running final model calibration...")
        # Train on balanced sample
        train_sample = pd.read_csv(train_path, nrows=150000)
        sample_feat = pipe.feature_extractor.extract_features(train_sample)
        pipe.detector.train(sample_feat, ground_truth_col='is_sar_reported')
        eval_res = pipe.evaluator.evaluate_all_models(pipe.test_features_df, threshold=0.35)
        ens_m = eval_res['metrics_list'][3]
    else:
        for c_idx in range(1, num_chunks + 1):
            batch_n = min(chunk_size, remaining_to_generate)
            
            # 1. Generate incremental chunk with synthetic noise
            chunk_df = generate_incremental_transactions(
                customer_df=pipe.customers_df,
                existing_df=last_tail,
                batch_size=batch_n,
                suspicious_ratio=suspicious_ratio,
                add_noise=True,
                noise_ratio=0.08
            )
            
            # 2. Append directly to CSV on disk
            chunk_df.to_csv(train_path, mode='a', header=not os.path.exists(train_path), index=False)
            last_tail = chunk_df.tail(50)
            
            cumulative_count += len(chunk_df)
            remaining_to_generate -= len(chunk_df)
            progress_pct = f"{(cumulative_count / target_total * 100):.1f}%"
            
            # 3. Train models on balanced sample of stream with preprocessing
            sample_size = min(len(chunk_df), 150000)
            sample_for_training = chunk_df.sample(n=sample_size, random_state=42)
            sample_features = pipe.feature_extractor.extract_features(sample_for_training)
            
            pipe.detector.train(sample_features, ground_truth_col='is_sar_reported')
            
            # 4. Evaluate against unseen test set at operating threshold 0.35
            eval_res = pipe.evaluator.evaluate_all_models(pipe.test_features_df, threshold=0.35)
            ens_m = eval_res['metrics_list'][3]
            
            acc_str = f"{ens_m['accuracy']*100:.2f}%"
            fpr_str = f"{ens_m['false_positive_rate']*100:.2f}%"
            fnr_str = f"{ens_m['false_negative_rate']*100:.2f}%"
            prec_str = f"{ens_m['precision']*100:.2f}%"
            auc_str = f"{ens_m['roc_auc']:.4f}"
            
            row = (
                f"#{c_idx:<6} | {cumulative_count:<15,} | {progress_pct:<10} | +{len(chunk_df):<9,} | "
                f"{acc_str:<9} | {fpr_str:<8} | {fnr_str:<8} | {prec_str:<10} | {auc_str:<8}"
            )
            print(row, flush=True)

    # 5. Save model weights bundle
    model_save_path = os.path.join(pipe.data_dir, 'trained_aml_models.pkl')
    with open(model_save_path, 'wb') as f:
        pickle.dump(pipe.detector, f)

    file_size_mb = os.path.getsize(train_path) / (1024 * 1024) if os.path.exists(train_path) else 0.0
    total_time = time.time() - start_time
    
    print("-" * 105)
    print(f"\n[OK] 6,000,000 DATA GENERATION & TRAINING COMPLETE!")
    print(f"  * Cumulative Training Data:  {cumulative_count:,} records ({cumulative_count/100000:.2f} Lakhs)")
    print(f"  * Training Dataset Size:     {file_size_mb:.2f} MB on disk")
    if ens_m:
        print(f"  * Calibrated Accuracy:       {ens_m['accuracy']*100:.2f}% (at 0.35 Threshold)")
        print(f"  * False Positive Rate (FPR): {ens_m['false_positive_rate']*100:.2f}%")
        print(f"  * False Negative Rate (FNR): {ens_m['false_negative_rate']*100:.2f}%")
        print(f"  * Detection Precision:       {ens_m['precision']*100:.2f}%")
        print(f"  * ROC-AUC:                   {ens_m['roc_auc']:.4f}")
    print(f"  * Serialized Model Weights:  data/trained_aml_models.pkl")
    print(f"  * Total Execution Time:      {total_time:.2f} seconds ({total_time/60:.1f} minutes)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AML models with 6 Million records")
    parser.add_argument("--target-total", type=int, default=6000000, help="Target total records (default 6,000,000)")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Chunk batch size (default 500,000)")
    args = parser.parse_args()
    
    run_6_million_pipeline(target_total=args.target_total, chunk_size=args.chunk_size)
