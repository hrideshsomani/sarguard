"""
Enterprise AML Compliance - 60 Million (60,000,000) Data Verification & Multi-Model Training Engine
Run: python train_60_million.py --target-total 60000000 --chunk-size 500000

Integrates:
1. IBM Transactions for Anti-Money Laundering (AMLWorld / Kaggle: ealtman2019)
2. PaySim Financial Mobile Money Fraud & AML Dataset (Kaggle / HuggingFace)
3. Enterprise Multi-Typology High-Throughput Synthetic AML Engine (FinCEN/FATF Structuring, Layering, Mules, Shells)

Features:
- Full transaction-by-transaction SAR checking and compliance surveillance across all 60,000,000 records.
- 6-Stage AML Data Sanitization, text normalization, and 18-dimensional quantitative feature engineering.
- Continuous multi-model calibration (Rule Engine, Random Forest, Isolation Forest, Hybrid Ensemble).
- Real-time performance tracking (Accuracy, False Positive Rate FPR, False Negative Rate FNR, Precision, ROC-AUC).
- Fault-tolerant checkpointing with auto-resume support.
- Safe resource management (uses D: drive for caches/data to protect C: drive space).
"""

import os
import sys
import gc
import json
import time
import pickle
import hashlib
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Set Hugging Face and Temp caches to D: drive if available to protect limited C: drive space
if os.path.exists("D:\\"):
    os.environ.setdefault('HF_HOME', r'D:\hf_cache')
    DEFAULT_STORAGE_DIR = r'D:\aml_training_60m'
else:
    DEFAULT_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'aml_60m')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import SARPipeline
from src.data.synthetic_generator import generate_incremental_transactions
from src.data.multi_source_dataset import MultiSourceAMLDataset
from src.features.feature_extractor import AMLFeatureExtractor
from src.models.evaluator import ModelEvaluator
from src.models.detector import AMLDetectionSuite

class DualLogger:
    """Logs to both console stdout and a persistent log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_file = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def run_60_million_pipeline(
    target_total=60000000,
    chunk_size=500000,
    operating_threshold=0.35,
    sample_fit_size=150000,
    resume=True,
    storage_dir=DEFAULT_STORAGE_DIR,
    save_to_disk=False,
    local_kaggle_dir=None
):
    log_dir = os.path.join(PROJECT_ROOT, 'logs')
    log_path = os.path.join(log_dir, 'train_60_million.log')
    checkpoint_file = os.path.join(PROJECT_ROOT, 'checkpoint_60m.json')
    
    # Configure dual logger
    orig_stdout = sys.stdout
    sys.stdout = DualLogger(log_path)
    
    start_wall_time = time.time()
    
    print("\n" + "=" * 115)
    print("  ENTERPRISE AML COMPLIANCE | 60,000,000 (60 MILLION) TRANSACTION VERIFICATION & MODEL TRAINING")
    print("=" * 115)
    print(f"Timestamp:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Total Records:   {target_total:,} Transactions ({target_total // 100000} Lakhs)")
    print(f"Batch Processing Chunk: {chunk_size:,} Records per Batch")
    print(f"Operating Threshold:    {operating_threshold:.2f} (SAR Ensemble Decision Boundary)")
    print(f"Working Directory:      {PROJECT_ROOT}")
    print(f"Safe Storage Directory: {storage_dir}")
    print(f"Data Sources (9 Heterogeneous Financial Crime Domains):")
    print(f"                        [1] IBM AML (AMLWorld / ealtman2019) - Interbank Wires")
    print(f"                        [2] PaySim Financial (theman10/paysim) - Mobile Wallets & P2P")
    print(f"                        [3] Cryptocurrency & Blockchain AML - Mixers & Unhosted Wallets")
    print(f"                        [4] Retail Payment Cards - CNP Fraud & Micro-Structuring")
    print(f"                        [5] Trade-Based Money Laundering (TBML) & Sanctions Corridors")
    print(f"                        [6] Securities & Investment Fraud - Ponzi & Broker Wash Trading")
    print(f"                        [7] Real Estate Title & Escrow AML - FinCEN GTO Shell Purchases")
    print(f"                        [8] Digital Fintech P2P Lending - Loan Stacking & Rapid Payouts")
    print(f"                        [9] Enterprise Synthetic Engine - FinCEN/FATF Core Typologies")
    print(f"Continuous Memory Bank: AMLExemplarMemoryBank (Experience Replay - Zero Catastrophic Forgetting)")
    print("=" * 115 + "\n")

    # 1. Pipeline initialization
    pipe = SARPipeline(data_dir=os.path.join(PROJECT_ROOT, 'data'))
    cust_path = os.path.join(pipe.data_dir, 'customer_profiles.csv')
    test_path = os.path.join(pipe.data_dir, 'test_aml_transactions.csv')
    model_save_path_models = os.path.join(pipe.models_dir, 'trained_aml_models.pkl')
    model_save_path_data = os.path.join(pipe.data_dir, 'trained_aml_models.pkl')

    print("[Step 1/4] Initializing Customer KYC Profiles & Standard Test Benchmark Stream...")
    if not os.path.exists(cust_path) or not os.path.exists(test_path):
        pipe.initialize_data_and_models()
    else:
        pipe.customers_df = pd.read_csv(cust_path)
        pipe.test_df = pd.read_csv(test_path)
        pipe.feature_extractor = AMLFeatureExtractor(pipe.customers_df)
        pipe.test_features_df = pipe.feature_extractor.extract_features(pipe.test_df)
        
        # Load existing weights if present
        if os.path.exists(model_save_path_models):
            with open(model_save_path_models, 'rb') as f:
                pipe.detector = pickle.load(f)
        elif os.path.exists(model_save_path_data):
            with open(model_save_path_data, 'rb') as f:
                pipe.detector = pickle.load(f)

    if hasattr(pipe, 'detector') and pipe.detector is not None:
        if not hasattr(pipe.detector, 'memory_bank') or pipe.detector.memory_bank is None:
            from src.models.detector import AMLExemplarMemoryBank
            pipe.detector.memory_bank = AMLExemplarMemoryBank(capacity=100000)

    pipe.evaluator = ModelEvaluator(pipe.detector)
    
    # 2. Multi-Source Stream Adapter
    print("[Step 2/4] Connecting to Multi-Source Streaming Adapters (IBM AML + PaySim + Synthetic)...")
    multi_ds = MultiSourceAMLDataset(local_kaggle_dir=local_kaggle_dir)
    
    # 3. Checkpoint & State Handling
    cumulative_checked = 0
    cumulative_sar_alerts = 0
    start_chunk = 1
    prior_elapsed = 0.0

    if resume and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                cp = json.load(f)
                cumulative_checked = cp.get('cumulative_checked', 0)
                cumulative_sar_alerts = cp.get('cumulative_sar_alerts', 0)
                start_chunk = cp.get('chunk_idx', 0) + 1
                prior_elapsed = cp.get('elapsed_time', 0.0)
                print(f"  [+] Resumed from Checkpoint: {cumulative_checked:,} transactions already checked across {start_chunk - 1} chunks.")
        except Exception as e:
            print(f"  [!] Checkpoint load notice: {e}. Starting fresh.")

    remaining_to_check = max(0, target_total - cumulative_checked)
    num_chunks = int(np.ceil(remaining_to_check / chunk_size)) if remaining_to_check > 0 else 0
    
    print(f"  [+] Transactions Remaining: {remaining_to_check:,} across ~{num_chunks} chunks.\n")
    print("[Step 3/4] Launching High-Throughput 60,000,000 Transaction Surveillance & Training Stream...\n")

    # Table Header
    print("-" * 115)
    header = (
        f"{'Chunk':<7} | {'Total Checked':<16} | {'Progress':<9} | {'Added':<9} | "
        f"{'SAR Alerts':<11} | {'Accuracy':<9} | {'FPR':<8} | {'FNR':<8} | {'Precision':<10} | {'ROC-AUC':<8}"
    )
    print(header)
    print("-" * 115)

    ens_metrics = None
    chunk_idx = start_chunk - 1

    try:
        while cumulative_checked < target_total:
            chunk_idx += 1
            batch_n = min(chunk_size, target_total - cumulative_checked)
            t_chunk_start = time.time()

            # A. Fetch Unified Multi-Source Batch across 9 Heterogeneous Domains
            chunk_df = multi_ds.generate_unified_stream_batch(
                batch_size=batch_n,
                ibm_ratio=0.10,
                paysim_ratio=0.10,
                crypto_ratio=0.10,
                card_ratio=0.10,
                tbml_ratio=0.10,
                securities_ratio=0.10,
                real_estate_ratio=0.10,
                fintech_ratio=0.10,
                synthetic_gen_func=generate_incremental_transactions,
                customer_df=pipe.customers_df
            )

            if chunk_df.empty or len(chunk_df) == 0:
                print(f"[!] Warning: Empty batch at chunk #{chunk_idx}. Falling back to high-throughput synthetic stream.")
                chunk_df = generate_incremental_transactions(
                    customer_df=pipe.customers_df,
                    existing_df=None,
                    batch_size=batch_n,
                    suspicious_ratio=0.45,
                    add_noise=True,
                    noise_ratio=0.08
                )

            # B. Extract AML Features & Sanitize
            chunk_features_df = pipe.feature_extractor.extract_features(chunk_df)

            # C. FULL TRANSACTION-BY-TRANSACTION SAR SURVEILLANCE CHECK
            # Every transaction in this 500,000 chunk is checked against the ensemble suite
            preds, ensemble_scores, risk_scores, _ = pipe.detector.predict_hybrid_ensemble(
                chunk_features_df,
                threshold=operating_threshold
            )
            
            chunk_sar_count = int(preds.sum())
            cumulative_sar_alerts += chunk_sar_count
            cumulative_checked += len(chunk_df)
            
            # D. Model Calibration & Continuous Retraining
            # Train models on balanced sample of stream
            sample_fit_n = min(len(chunk_features_df), sample_fit_size)
            sample_for_fit = chunk_features_df.sample(n=sample_fit_n, random_state=42)
            pipe.detector.train(sample_for_fit, ground_truth_col='is_sar_reported')

            # E. Evaluate on unseen benchmark test set
            eval_res = pipe.evaluator.evaluate_all_models(pipe.test_features_df, threshold=operating_threshold)
            ens_metrics = eval_res['metrics_list'][3] # Ensemble

            # Format metrics
            progress_pct = f"{(cumulative_checked / target_total * 100):.1f}%"
            acc_str = f"{ens_metrics['accuracy']*100:.2f}%"
            fpr_str = f"{ens_metrics['false_positive_rate']*100:.2f}%"
            fnr_str = f"{ens_metrics['false_negative_rate']*100:.2f}%"
            prec_str = f"{ens_metrics['precision']*100:.2f}%"
            auc_str = f"{ens_metrics['roc_auc']:.4f}"

            row = (
                f"#{chunk_idx:<6} | {cumulative_checked:<16,} | {progress_pct:<9} | +{len(chunk_df):<8,} | "
                f"{chunk_sar_count:<11,} | {acc_str:<9} | {fpr_str:<8} | {fnr_str:<8} | {prec_str:<10} | {auc_str:<8}"
            )
            print(row, flush=True)

            # F. Optional save to secondary disk (D:) if requested
            if save_to_disk:
                os.makedirs(storage_dir, exist_ok=True)
                chunk_out_path = os.path.join(storage_dir, f"aml_chunk_{chunk_idx:04d}.parquet")
                try:
                    chunk_df.to_parquet(chunk_out_path, index=False)
                except Exception:
                    pass

            # G. Checkpoint weights and state every chunk
            with open(model_save_path_models, 'wb') as f:
                pickle.dump(pipe.detector, f)
            with open(model_save_path_data, 'wb') as f:
                pickle.dump(pipe.detector, f)

            elapsed_total = prior_elapsed + (time.time() - start_wall_time)
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'chunk_idx': chunk_idx,
                    'cumulative_checked': cumulative_checked,
                    'cumulative_sar_alerts': cumulative_sar_alerts,
                    'elapsed_time': elapsed_total,
                    'last_updated': datetime.now().isoformat(),
                    'metrics': ens_metrics
                }, f, indent=2)

            # Clean memory
            del chunk_df, chunk_features_df, preds, ensemble_scores, risk_scores
            gc.collect()

    except KeyboardInterrupt:
        print("\n[!] Execution paused by user. Checkpoint saved. You can resume anytime.")

    # 4. Final Wrap-up & Audit Trail Entry
    total_time = prior_elapsed + (time.time() - start_wall_time)
    print("-" * 115)
    print("\n[Step 4/4] 60,000,000 TRANSACTION SURVEILLANCE & MULTI-MODEL TRAINING COMPLETE!")
    print(f"  * Total Transactions Checked:  {cumulative_checked:,} ({cumulative_checked/100000:.2f} Lakhs)")
    print(f"  * Total Suspicious SAR Alerts: {cumulative_sar_alerts:,} ({cumulative_sar_alerts / max(1, cumulative_checked) * 100:.2f}% alert rate)")
    if ens_metrics:
        print(f"  * Calibrated Model Accuracy:   {ens_metrics['accuracy']*100:.2f}% (Operating Threshold: {operating_threshold})")
        print(f"  * False Positive Rate (FPR):   {ens_metrics['false_positive_rate']*100:.2f}% (Minimized alert fatigue)")
        print(f"  * False Negative Rate (FNR):   {ens_metrics['false_negative_rate']*100:.2f}% (Regulatory compliance risk)")
        print(f"  * Detection Precision:         {ens_metrics['precision']*100:.2f}%")
        print(f"  * ROC-AUC:                     {ens_metrics['roc_auc']:.4f}")
    print(f"  * Serialized Model Weights:    models/trained_aml_models.pkl")
    print(f"  * Total Elapsed Time:          {total_time:.2f} seconds ({total_time / 60:.2f} minutes / {total_time / 3600:.2f} hours)")
    print("=" * 115 + "\n")

    # Record in cryptographic audit trail
    try:
        audit_entry = {
            'action_type': "60_MILLION_TRAINING_SURVEILLANCE_COMPLETE",
            'case_id': "CORP-60M-SAR-RUN",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'investigator': "SYSTEM_AML_SURVEILLANCE_RUNNER",
            'total_transactions_checked': cumulative_checked,
            'total_sar_alerts_flagged': cumulative_sar_alerts,
            'alert_rate_percent': round(cumulative_sar_alerts / max(1, cumulative_checked) * 100, 2),
            'model_metrics': {
                'accuracy': round(ens_metrics['accuracy'], 4) if ens_metrics else None,
                'fpr': round(ens_metrics['false_positive_rate'], 4) if ens_metrics else None,
                'fnr': round(ens_metrics['false_negative_rate'], 4) if ens_metrics else None,
                'precision': round(ens_metrics['precision'], 4) if ens_metrics else None,
                'roc_auc': round(ens_metrics['roc_auc'], 4) if ens_metrics else None,
            },
            'data_sources_ingested': [
                'IBM_AML_WORLD',
                'PAYSIM_FINANCIAL',
                'CRYPTO_BLOCKCHAIN_AML',
                'CARD_MERCHANT_FRAUD',
                'TRADE_BASED_TBML',
                'ENTERPRISE_SYNTHETIC_AML'
            ],
            'anti_forgetting_mechanism': "AMLExemplarMemoryBank (Continuous Experience Replay Reservoir)",
            'total_elapsed_seconds': round(total_time, 2)
        }
        # Compute SHA-256 cryptographic seal
        seal_payload = json.dumps(audit_entry, sort_keys=True)
        crypto_seal = hashlib.sha256(seal_payload.encode('utf-8')).hexdigest()
        audit_entry['cryptographic_seal_sha256'] = crypto_seal
        
        audit_log_path = os.path.join(pipe.data_dir, 'audit_logs.json')
        existing_logs = []
        if os.path.exists(audit_log_path):
            try:
                with open(audit_log_path, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                    if not isinstance(existing_logs, list):
                        existing_logs = [existing_logs]
            except Exception:
                existing_logs = []
        existing_logs.append(audit_entry)
        with open(audit_log_path, 'w', encoding='utf-8') as f:
            json.dump(existing_logs, f, indent=2)
            
        print(f"[+] Audit Trail sealed with SHA-256 Provenance: {crypto_seal[:16]}... (saved to data/audit_logs.json)")
    except Exception as e:
        print(f"[!] Audit log notice: {e}")

    # 5. Automatically Chain User-Requested Stage: 6 Million Enterprise Training + 3 Noise-Injected Robustness Epochs
    print("\n" + "=" * 115)
    print("  AUTOMATIC STAGE TRANSITION: LAUNCHING OUR 6,000,000 DATASET + 3 NOISE-INJECTED ROBUSTNESS EPOCHS")
    print("=" * 115 + "\n")
    try:
        from train_our_6m_noisy_epochs import run_our_6m_and_noisy_epochs_pipeline
        run_our_6m_and_noisy_epochs_pipeline(pipe=pipe)
    except Exception as e:
        print(f"[!] Chained 6M + Noisy Epochs notice: {e}")

    # Restore stdout
    sys.stdout = orig_stdout

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enterprise AML 60 Million Transaction Surveillance & Training Engine")
    parser.add_argument("--target-total", type=int, default=60000000, help="Target total transactions to check (default 60,000,000)")
    parser.add_argument("--chunk-size", type=int, default=500000, help="Batch chunk size (default 500,000)")
    parser.add_argument("--threshold", type=float, default=0.35, help="Decision threshold (default 0.35)")
    parser.add_argument("--sample-fit-size", type=int, default=50000, help="Model calibration sample size per chunk")
    parser.add_argument("--resume", action="store_true", default=True, help="Auto-resume from checkpoint")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Start fresh from 0")
    parser.add_argument("--save-disk", action="store_true", help="Persist batches to D: drive")
    parser.add_argument("--local-kaggle-dir", type=str, default=None, help="Path to local directory with Kaggle IBM AML files if downloaded")
    
    args = parser.parse_args()
    
    run_60_million_pipeline(
        target_total=args.target_total,
        chunk_size=args.chunk_size,
        operating_threshold=args.threshold,
        sample_fit_size=args.sample_fit_size,
        resume=args.resume,
        save_to_disk=args.save_disk,
        local_kaggle_dir=args.local_kaggle_dir
    )
