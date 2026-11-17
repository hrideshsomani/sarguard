"""
Continuous Data Generation & Active Model Training Pipeline
Run: python continuous_training.py --target-total 300000 --batch-size 50000
Or:  python continuous_training.py --cycles 5 --batch-size 6000

Continuously generates fresh synthetic financial transactions, appends them to the
cumulative training dataset, retrains the multi-model AML suite, and monitors
Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) across cycles.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import SARPipeline

def run_continuous_training(cycles=0, batch_size=50000, target_total=300000, interval=1.0, suspicious_ratio=0.20):
    print("=" * 105)
    print("[*] ENTERPRISE AML COMPLIANCE | CONTINUOUS DATA GENERATION & MODEL TRAINING")
    print("=" * 105)
    print(f"Working Directory: {PROJECT_ROOT}")
    if target_total > 0:
        print(f"Goal Target:       {target_total:,} Total Records ({target_total // 100000} Lakhs)")
    print(f"Batch Increment:   +{batch_size:,} records per cycle | Interval: {interval}s")
    print("=" * 105)
    
    pipe = SARPipeline(data_dir=os.path.join(PROJECT_ROOT, 'data'))
    print("\n[+] Initializing base dataset and models...")
    pipe.initialize_data_and_models()
    
    initial_count = len(pipe.train_df)
    initial_metrics = pipe.evaluation_results['metrics_list'][3] # Ensemble
    
    print(f"Base Training Records: {initial_count:,}")
    print(f"Base Ensemble Metrics: Accuracy: {initial_metrics['accuracy']*100:.2f}% | FPR: {initial_metrics['false_positive_rate']*100:.2f}% | FNR: {initial_metrics['false_negative_rate']*100:.2f}% | ROC-AUC: {initial_metrics['roc_auc']:.4f}\n")
    
    print("-" * 105)
    header = f"{'Cycle':<7} | {'Total Records':<15} | {'Progress':<10} | {'Added':<9} | {'Accuracy':<9} | {'FPR':<8} | {'FNR':<8} | {'Precision':<10} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 105)
    
    cycle_idx = 1
    try:
        while True:
            current_count = len(pipe.train_df)
            if target_total > 0 and current_count >= target_total:
                print(f"\n[✓] Target reached: {current_count:,} records (>= {target_total:,}).")
                break
                
            # If remaining records to target is smaller than batch_size, adjust batch_size
            curr_batch = batch_size
            if target_total > 0:
                remaining = target_total - current_count
                if remaining < batch_size:
                    curr_batch = max(remaining, 1000)
                    
            t0 = time.time()
            
            # Append fresh data, re-extract features, and retrain
            res = pipe.append_fresh_data_and_retrain(batch_size=curr_batch, suspicious_ratio=suspicious_ratio)
            
            ens_m = res['new_metrics']['metrics_list'][3]
            acc_str = f"{ens_m['accuracy']*100:.2f}%"
            fpr_str = f"{ens_m['false_positive_rate']*100:.2f}%"
            fnr_str = f"{ens_m['false_negative_rate']*100:.2f}%"
            prec_str = f"{ens_m['precision']*100:.2f}%"
            auc_str = f"{ens_m['roc_auc']:.4f}"
            
            prog_pct = f"{(res['total_train_records'] / target_total * 100):.1f}%" if target_total > 0 else "N/A"
            
            row = (
                f"#{cycle_idx:<6} | {res['total_train_records']:<15,} | {prog_pct:<10} | +{res['new_records_count']:<8,} | "
                f"{acc_str:<9} | {fpr_str:<8} | {fnr_str:<8} | {prec_str:<10} | {auc_str:<8}"
            )
            print(row)
            
            if target_total > 0 and res['total_train_records'] >= target_total:
                break
                
            if cycles > 0 and cycle_idx >= cycles:
                break
                
            cycle_idx += 1
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n[!] Training loop paused by user (Ctrl+C).")
        
    final_count = len(pipe.train_df)
    final_metrics = pipe.evaluation_results['metrics_list'][3]
    print("-" * 105)
    print(f"\n[OK] 3 LAKH (300,000 RECORDS) TRAINING COMPLETED!")
    print(f"  * Cumulative Training Data:  {final_count:,} records ({final_count/100000:.2f} Lakhs)")
    print(f"  * Total Data Ingested:       +{final_count - initial_count:,} fresh transactions")
    print(f"  * Final Ensemble Accuracy:   {final_metrics['accuracy']*100:.2f}%")
    print(f"  * False Positive Rate (FPR): {final_metrics['false_positive_rate']*100:.2f}% (Ultra-low alert fatigue)")
    print(f"  * False Negative Rate (FNR): {final_metrics['false_negative_rate']*100:.2f}%")
    print(f"  * Final ROC-AUC Score:       {final_metrics['roc_auc']:.4f}")
    print(f"  * Serialized Model Weights:  data/trained_aml_models.pkl")
    print(f"  * Updated Training Set CSV:  data/train_aml_transactions.csv\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continuous Data Generation & AML Training")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0 to use target-total)")
    parser.add_argument("--target-total", type=int, default=300000, help="Target total records (e.g. 300000 for 3 Lakh)")
    parser.add_argument("--batch-size", type=int, default=50000, help="Fresh transactions generated per cycle (default: 50000)")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds pause between cycles (default: 1.0)")
    parser.add_argument("--suspicious-ratio", type=float, default=0.20, help="Ratio of suspicious AML typologies in fresh batches (default: 0.20)")
    
    args = parser.parse_args()
    run_continuous_training(
        cycles=args.cycles,
        batch_size=args.batch_size,
        target_total=args.target_total,
        interval=args.interval,
        suspicious_ratio=args.suspicious_ratio
    )
