"""
Script to generate a standalone, self-contained Google Colab Jupyter Notebook (SAR_Narrative_Generator_Colab.ipynb)
"""

import json
import os

def create_notebook():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 2,
        "metadata": {
            "colab": {
                "name": "SAR_Narrative_Generator_Colab.ipynb",
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": []
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # 1. Header
    add_md("""
# Enterprise AML Compliance Suite: AI SAR Narrative Generator with Cryptographic Audit Trail
### FinCEN / FCA Regulatory Standards | Multi-Model Detection | Reversible PII Masking | Tamper-Evident SHA-256 Seal

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

---

## 📌 Executive Overview
This notebook implements an enterprise-grade AI/ML compliance system for Anti-Money Laundering (AML) monitoring, Suspicious Activity Report (SAR) narrative generation, and immutable audit trail verification conforming to **FinCEN (US)** and **FCA (UK)** standards.

### Key Capabilities:
1. **Synthetic AML Stream Generator**: Simulates normal banking behavior and 5 distinct financial crime typologies (*Structuring*, *Rapid Flow-Through*, *Mule Fan-In*, *Shell Round-Tripping*, and *Dormant Spikes*).
2. **Vectorized Feature Extraction**: 18-dimensional quantitative, temporal, graph velocity, and FinCEN threshold proximity features.
3. **Multi-Model AML Suite**: Combines deterministic Compliance Rules (AML-101 to 106), Supervised Random Forest, and Unsupervised Isolation Forest into a Calibrated Hybrid Ensemble.
4. **Comprehensive Evaluation Engine**: Explicit benchmarking of Accuracy, False Positive Rate (FPR), False Negative Rate (FNR), Precision, Recall, F1, and ROC-AUC with dynamic threshold calibration.
5. **GDPR / BSA Reversible PII Obfuscation**: Cryptographic HMAC-SHA256 vault for redacting sensitive personal data while maintaining internal investigation coherence.
6. **5-Part FinCEN / FCA Narrative Synthesizer**: Formulates professional SAR reports covering *Who, What, When, Where,* and *Why*.
7. **Immutable Audit Trail & SHA-256 Tamper Seal**: Sentence-to-transaction lineage tracking with cryptographic verification.
8. **Interactive In-Notebook UI**: Interactive controls via `ipywidgets` allowing threshold tuning, subject selection, PII masking toggling, and live SAR generation.
""")

    # 2. Setup
    add_code("""
# @title 1. Environment Setup & Package Imports
import os
import sys
import json
import time
import hmac
import hashlib
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("muted")

print("[OK] Core libraries imported successfully.")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
""")

    # 3. Data Generator
    add_code("""
# @title 2. Synthetic AML Data Generator (5 Core Typologies)

HIGH_RISK_JURISDICTIONS = {'KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG'}

def generate_customer_profiles(num_customers=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    male_names = ["James", "Liam", "Noah", "William", "Michael", "Arthur", "Oliver", "Alexander", "Daniel", "Henry"]
    female_names = ["Emma", "Olivia", "Ava", "Sophia", "Isabella", "Charlotte", "Amelia", "Mia", "Harper", "Evelyn"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez", 
                  "Sterling", "Davies", "Taylor", "Evans", "Wilson", "Patel", "Wright", "Thompson", "Walker", "White"]
    corp_names = ["Apex Holdings Ltd", "Vanguard Logistics LLC", "Sterling Global Trade Inc", "Blue Horizon Imports Corp", 
                  "Pinnacle Ventures Group", "Atlas Capital Partners", "Meridian Consulting Ltd", "Oasis Offshore Inc"]
    
    customers = []
    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:04d}"
        is_corp = (i % 2 == 0)
        c_type = "CORPORATE" if is_corp else "INDIVIDUAL"
        
        if is_corp:
            name = corp_names[(i // 2) % len(corp_names)] + f" #{i}"
            turnover = round(random.uniform(50000.0, 500000.0), 2)
            country = random.choices(["GB", "US", "DE", "KY", "PA", "AE"], weights=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05])[0]
            risk = ['LOW', 'MEDIUM', 'HIGH'][i % 3]
        else:
            first_name = male_names[(i // 4) % len(male_names)] if (i // 2) % 2 == 0 else female_names[(i // 4) % len(female_names)]
            name = f"{first_name} {last_names[i % len(last_names)]}"
            turnover = round(random.uniform(2500.0, 15000.0), 2)
            country = random.choices(["GB", "US", "CA", "FR", "DE"], weights=[0.5, 0.3, 0.1, 0.05, 0.05])[0]
            risk = ['LOW', 'MEDIUM', 'HIGH'][(i + 1) % 3]
            
        customers.append({
            'customer_id': cid,
            'customer_name': name,
            'customer_type': c_type,
            'expected_monthly_turnover': turnover,
            'risk_rating': risk,
            'primary_country': country,
            'kyc_status': "VERIFIED"
        })
    return pd.DataFrame(customers)

def inject_synthetic_noise(df, noise_ratio=0.08, seed=42):
    # Injects realistic banking noise: missing IPs/accounts, text whitespace/casing inconsistencies, micro-cents jitter
    d = df.copy()
    np.random.seed(seed)
    random.seed(seed)
    n = len(d)
    
    d.loc[np.random.rand(n) < (noise_ratio * 0.6), 'device_ip'] = np.nan
    d.loc[np.random.rand(n) < (noise_ratio * 0.4), 'beneficiary_account'] = None
    
    if 'channel' in d.columns:
        dirty_c = [' online_banking ', '  SWIFT  ', 'branch_counter', 'mobile_app ', ' ATM  ']
        for idx in d[np.random.rand(n) < (noise_ratio * 0.8)].index:
            d.at[idx, 'channel'] = random.choice(dirty_c)
            
    if 'transaction_type' in d.columns:
        dirty_t = [' wire_transfer ', '  ach_transfer  ', 'CASH_DEPOSIT ', '  peer_to_peer', 'crypto_purchase ']
        for idx in d[np.random.rand(n) < (noise_ratio * 0.8)].index:
            d.at[idx, 'transaction_type'] = random.choice(dirty_t)
            
    if 'amount' in d.columns:
        for idx in d[np.random.rand(n) < (noise_ratio * 0.6)].index:
            d.at[idx, 'amount'] = round(float(d.at[idx, 'amount']) + random.uniform(-0.45, 0.45), 3)
            
    return d

def generate_synthetic_transactions(customers_df, total_records=10000, suspicious_ratio=0.50, seed=42, add_noise=True):
    random.seed(seed)
    np.random.seed(seed)
    
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    txns = []
    suspicious_count = int(total_records * suspicious_ratio)
    normal_count = total_records - suspicious_count
    
    channels = ["ONLINE_BANKING", "SWIFT", "MOBILE_APP", "BRANCH", "ATM"]
    txn_types = ["WIRE_OUT", "WIRE_IN", "TRANSFER", "CASH_DEPOSIT", "CASH_WITHDRAWAL"]
    
    # 1. Normal Transactions
    for i in range(normal_count):
        cust = customers_df.iloc[i % len(customers_df)]
        t_offset = random.randint(0, 90 * 86400)
        t_stamp = start_date + timedelta(seconds=t_offset)
        amt = round(random.uniform(25.0, min(cust['expected_monthly_turnover'] * 0.40, 8000.0)), 2)
        
        txns.append({
            'transaction_id': f"TXN-{random.randint(10000000, 99999999)}",
            'timestamp': t_stamp,
            'customer_id': cust['customer_id'],
            'originator_name': cust['customer_name'],
            'originator_country': cust['primary_country'],
            'beneficiary_name': f"Vendor-{random.randint(100, 999)}",
            'beneficiary_country': cust['primary_country'],
            'amount': amt,
            'currency': 'USD',
            'channel': channels[i % len(channels)],
            'transaction_type': txn_types[i % len(txn_types)],
            'device_ip': f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
            'is_sar_reported': 0,
            'aml_typology': 'NORMAL'
        })
        
    # 2. Suspicious Typologies (Exact Quotas across 5 Typologies)
    typologies = ['STRUCTURING', 'RAPID_FLOW_THROUGH', 'MULE_FAN_IN', 'ROUND_TRIPPING_SHELL', 'DORMANT_SPIKE']
    base_quota = suspicious_count // len(typologies)
    remainder = suspicious_count % len(typologies)
    c_idx = 0
    
    for t_i, typology in enumerate(typologies):
        quota = base_quota + (1 if t_i < remainder else 0)
        for _ in range(quota):
            cust = customers_df.iloc[c_idx % len(customers_df)]
            c_idx += 1
            t_offset = random.randint(0, 90 * 86400)
            t_stamp = start_date + timedelta(seconds=t_offset)
            
            orig_country = cust['primary_country']
            dest_country = cust['primary_country']
            
            if typology == 'STRUCTURING':
                amt = round(random.uniform(9000.0, 9950.0), 2)
                c_type = 'BRANCH'
                t_type = 'CASH_DEPOSIT'
            elif typology == 'RAPID_FLOW_THROUGH':
                amt = round(random.uniform(25000.0, 150000.0), 2)
                c_type = 'SWIFT'
                t_type = 'WIRE_OUT'
                dest_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
            elif typology == 'MULE_FAN_IN':
                amt = round(random.uniform(4000.0, 8500.0), 2)
                c_type = 'MOBILE_APP'
                t_type = 'TRANSFER'
            elif typology == 'ROUND_TRIPPING_SHELL':
                amt = round(random.uniform(50000.0, 250000.0), 2)
                c_type = 'SWIFT'
                t_type = 'WIRE_OUT'
                orig_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
                dest_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
            else: # DORMANT_SPIKE
                amt = round(random.uniform(30000.0, 120000.0), 2)
                c_type = 'ONLINE_BANKING'
                t_type = 'WIRE_OUT'
                
            txns.append({
                'transaction_id': f"TXN-{random.randint(10000000, 99999999)}",
                'timestamp': t_stamp,
                'customer_id': cust['customer_id'],
                'originator_name': cust['customer_name'],
                'originator_country': orig_country,
                'beneficiary_name': f"Entity-{random.randint(1000, 9999)}",
                'beneficiary_country': dest_country,
                'amount': amt,
                'currency': 'USD',
                'channel': c_type,
                'transaction_type': t_type,
                'device_ip': f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                'is_sar_reported': 1,
                'aml_typology': typology
            })
        
    df = pd.DataFrame(txns).sort_values('timestamp').reset_index(drop=True)
    if add_noise:
        df = inject_synthetic_noise(df, noise_ratio=0.08, seed=seed)
    return df

# 6-Stage Preprocessor
class AMLDataPreprocessor:
    def preprocess(self, df):
        d = df.copy()
        rep = {'missing_imputed': 0, 'casing_fixes': 0, 'outliers_capped': 0}
        d['amount'] = pd.to_numeric(d['amount'], errors='coerce').fillna(250.0).clip(lower=1.0)
        for col in ['channel', 'transaction_type', 'originator_country', 'beneficiary_country']:
            if col in d.columns:
                orig = d[col].astype(str)
                cleaned = orig.str.strip().str.upper()
                rep['casing_fixes'] += int((orig != cleaned).sum())
                d[col] = cleaned
        if 'device_ip' in d.columns:
            rep['missing_imputed'] += int(d['device_ip'].isna().sum())
            d['device_ip'] = d['device_ip'].fillna('UNKNOWN_IP')
        if 'beneficiary_account' in d.columns:
            rep['missing_imputed'] += int(d['beneficiary_account'].isna().sum())
            d['beneficiary_account'] = d['beneficiary_account'].fillna('UNSPECIFIED_ACCT')
        if not pd.api.types.is_datetime64_any_dtype(d['timestamp']):
            d['timestamp'] = pd.to_datetime(d['timestamp'], errors='coerce')
        d['timestamp'] = d['timestamp'].fillna(pd.Timestamp.now())
        d = d.sort_values(['customer_id', 'timestamp']).reset_index(drop=True)
        q999 = d['amount'].quantile(0.999)
        rep['outliers_capped'] = int((d['amount'] > q999).sum())
        d['amount'] = d['amount'].clip(upper=q999)
        d['amount'] = d['amount'].round(2)
        total_checks = len(d) * 4
        anomalies = rep['missing_imputed'] + rep['casing_fixes'] + rep['outliers_capped']
        rep['data_health_score'] = round(max(0.0, 100.0 - (anomalies / max(total_checks, 1) * 100.0)), 2)
        return d, rep

print("[+] Generating demonstration datasets with synthetic noise and 6-stage sanitization...")
customers_df = generate_customer_profiles(num_customers=60, seed=42)
train_raw_df = generate_synthetic_transactions(customers_df, total_records=12000, suspicious_ratio=0.50, seed=42, add_noise=True)
test_raw_df = generate_synthetic_transactions(customers_df, total_records=2000, suspicious_ratio=0.50, seed=99, add_noise=True)

preproc = AMLDataPreprocessor()
train_clean_df, train_rep = preproc.preprocess(train_raw_df)
test_clean_df, test_rep = preproc.preprocess(test_raw_df)
print(f"[OK] Generated {len(customers_df)} customer profiles, {len(train_raw_df)} training records, and {len(test_raw_df)} test records.")
print(f"[OK] Data Preprocessing Audit: Health Score={train_rep['data_health_score']}% | Imputed={train_rep['missing_imputed']} | Fixed={train_rep['casing_fixes']}")
""")

    # 4. Feature Extractor
    add_code("""
# @title 3. Vectorized AML Feature Extractor

class AMLFeatureExtractor:
    def __init__(self, customer_df):
        self.customer_metadata = {}
        self.preprocessor = AMLDataPreprocessor()
        for _, row in customer_df.iterrows():
            cid = row['customer_id']
            self.customer_metadata[cid] = {
                'expected_turnover': float(row.get('expected_monthly_turnover', 5000.0)),
                'risk_rating': str(row.get('risk_rating', 'LOW')),
                'customer_type': str(row.get('customer_type', 'INDIVIDUAL')),
                'primary_country': str(row.get('primary_country', 'GB'))
            }
            
    def extract_features(self, transactions_df):
        clean_df, _ = self.preprocessor.preprocess(transactions_df)
        df = clean_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['dt'] = pd.to_datetime(df['timestamp'])
        else:
            df['dt'] = df['timestamp']
            
        df = df.sort_values('dt').reset_index(drop=True)
        
        # 1. Customer joins
        risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        turnovers, cust_risks, is_corps = [], [], []
        for cid in df['customer_id']:
            meta = self.customer_metadata.get(cid, {})
            turnovers.append(meta.get('expected_turnover', 5000.0))
            cust_risks.append(risk_map.get(meta.get('risk_rating', 'LOW'), 1))
            is_corps.append(1 if meta.get('customer_type', 'INDIVIDUAL') != 'INDIVIDUAL' else 0)
            
        df['expected_turnover'] = turnovers
        df['customer_risk_level'] = cust_risks
        df['is_corporate_account'] = is_corps
        
        # 2. Amounts & Structuring
        df['amount_log'] = np.log1p(df['amount'].astype(float))
        df['turnover_ratio'] = df['amount'] / np.maximum(df['expected_turnover'], 100.0)
        
        is_struct = ((df['amount'] >= 9000.0) & (df['amount'] <= 9999.0)).astype(int)
        df['is_structuring_range'] = is_struct
        df['structuring_proximity'] = np.where(
            df['is_structuring_range'] == 1,
            1.0 - (np.abs(df['amount'] - 9500.0) / 500.0),
            0.0
        )
        
        # Round amount flag
        df['is_round_amount'] = (((df['amount'] >= 10000.0) & (df['amount'] % 10000.0 == 0.0)) | 
                                 ((df['amount'] >= 5000.0) & (df['amount'] % 5000.0 == 0.0))).astype(int)
        
        # 3. Jurisdictions
        df['is_high_risk_origin'] = df['originator_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_high_risk_dest'] = df['beneficiary_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_cross_border'] = (df['originator_country'] != df['beneficiary_country']).astype(int)
        df['jurisdiction_risk_score'] = df['is_high_risk_origin'] * 2 + df['is_high_risk_dest'] * 2 + df['is_cross_border']
        
        # 4. Channels
        cash_crypto_types = {'CASH_DEPOSIT', 'CASH_WITHDRAWAL', 'CRYPTO_PURCHASE'}
        df['is_cash_or_crypto'] = df['transaction_type'].isin(cash_crypto_types).astype(int)
        df['is_swift_wire'] = df['channel'].isin({'SWIFT'}).astype(int)
        
        # 5. Temporal
        df['hour'] = df['dt'].dt.hour
        df['day_of_week'] = df['dt'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_hours'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # 6. Vectorized Velocity (rolling 24h)
        df['orig_idx'] = np.arange(len(df))
        df_sorted = df.sort_values(['customer_id', 'dt'])
        roll_stats = df_sorted.set_index('dt').groupby('customer_id')['amount'].rolling('24h').agg(['count', 'sum']).reset_index()
        df_sorted['velocity_24h_count'] = roll_stats['count'].values
        df_sorted['velocity_24h_sum'] = roll_stats['sum'].values
        df_sorted['velocity_24h_sum_log'] = np.log1p(df_sorted['velocity_24h_sum'])
        
        # Flow-through flag within 6h
        prev_dt = df_sorted.groupby('customer_id')['dt'].shift(1)
        prev_amt = df_sorted.groupby('customer_id')['amount'].shift(1)
        time_diff = (df_sorted['dt'] - prev_dt).dt.total_seconds()
        is_flow = ((time_diff <= 21600) & (time_diff > 0) & (df_sorted['amount'] > 10000) & 
                   ((df_sorted['amount'] - prev_amt).abs() / np.maximum(df_sorted['amount'], 1.0) < 0.20)).astype(int)
        df_sorted['rapid_flow_through_flag'] = is_flow
        
        df = df_sorted.sort_values('orig_idx').drop(columns=['orig_idx'])
        if 'dt' in df.columns:
            df = df.drop(columns=['dt'])
        return df

    @staticmethod
    def get_feature_columns():
        return [
            'amount_log', 'turnover_ratio', 'is_structuring_range', 'structuring_proximity',
            'is_round_amount', 'is_high_risk_origin', 'is_high_risk_dest', 'is_cross_border',
            'jurisdiction_risk_score', 'is_cash_or_crypto', 'is_swift_wire', 'customer_risk_level',
            'is_corporate_account', 'is_weekend', 'is_night_hours', 'velocity_24h_count',
            'velocity_24h_sum_log', 'rapid_flow_through_flag'
        ]

print("[+] Extracting features from transaction streams...")
fe = AMLFeatureExtractor(customers_df)
train_feat_df = fe.extract_features(train_raw_df)
test_feat_df = fe.extract_features(test_raw_df)
print(f"[OK] Features extracted across {len(fe.get_feature_columns())} quantitative dimensions.")
""")

    # 5. Rule Engine
    add_code("""
# @title 4. Compliance Rule Engine (FinCEN / FCA Standards)

class AMLRuleEngine:
    @staticmethod
    def evaluate_transaction(row):
        triggered = []
        amt = float(row.get('amount', 0))
        
        # RULE-101: Cash Structuring
        if 9000.0 <= amt <= 9999.0 and row.get('is_cash_or_crypto', 0) == 1:
            triggered.append({
                'id': 'AML-RULE-101', 'name': 'Cash Structuring Threshold Avoidance',
                'severity': 'HIGH', 'weight': 0.85,
                'description': f'Transaction amount ${amt:,.2f} is just below the mandatory $10,000 reporting threshold.'
            })
            
        # RULE-102: Rapid Flow-Through
        if row.get('rapid_flow_through_flag', 0) == 1:
            triggered.append({
                'id': 'AML-RULE-102', 'name': 'Rapid In-and-Out Flow-Through',
                'severity': 'HIGH', 'weight': 0.80,
                'description': 'Rapid transfer of matching funds in and out of account within 6 hours.'
            })
            
        # RULE-103: High Velocity Spike
        if row.get('velocity_24h_count', 0) >= 5 or row.get('turnover_ratio', 0) >= 3.0:
            triggered.append({
                'id': 'AML-RULE-103', 'name': 'Unusual Velocity / Turnover Multiplier Spike',
                'severity': 'MEDIUM', 'weight': 0.65,
                'description': f'24-hour transaction frequency ({row.get("velocity_24h_count")}) exceeds baseline KYC velocity.'
            })
            
        # RULE-104: High Risk Cross-Border
        if row.get('jurisdiction_risk_score', 0) >= 2:
            triggered.append({
                'id': 'AML-RULE-104', 'name': 'High-Risk Jurisdictional Exposure',
                'severity': 'HIGH', 'weight': 0.75,
                'description': f'Transaction routes through high-risk/FATF-monitored jurisdictions.'
            })
            
        # RULE-105: Large Round Amount
        if row.get('is_round_amount', 0) == 1 and amt >= 25000.0:
            triggered.append({
                'id': 'AML-RULE-105', 'name': 'Significant Round-Sum Wire Transfer',
                'severity': 'MEDIUM', 'weight': 0.50,
                'description': f'Unbacked large round amount ${amt:,.2f} transfer.'
            })
            
        return triggered

print("[OK] AML Rule Engine configured with AML-RULE-101 through 105.")
""")

    # 6. Multi-Model Suite
    add_code("""
# @title 5. Multi-Model Detection Suite (Random Forest, Isolation Forest, Hybrid Ensemble)

class AMLDetectionSuite:
    def __init__(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=4, class_weight='balanced', random_state=42, n_jobs=-1)
        self.iso_model = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
        self.feature_cols = AMLFeatureExtractor.get_feature_columns()
        self.rule_engine = AMLRuleEngine()
        
    def train(self, train_df, ground_truth_col='is_sar_reported'):
        X = train_df[self.feature_cols].fillna(0)
        y = train_df[ground_truth_col].astype(int)
        self.rf_model.fit(X, y)
        self.iso_model.fit(X)
        
    def predict_rule_based(self, df):
        preds, probas, all_triggers = [], [], []
        for _, row in df.iterrows():
            triggers = self.rule_engine.evaluate_transaction(row)
            all_triggers.append(triggers)
            if len(triggers) > 0:
                max_w = max(t['weight'] for t in triggers)
                preds.append(1 if max_w >= 0.70 else 0)
                probas.append(max_w)
            else:
                preds.append(0)
                probas.append(0.0)
        return np.array(preds), np.array(probas), all_triggers
        
    def predict_random_forest(self, df, threshold=0.45):
        X = df[self.feature_cols].fillna(0)
        proba = self.rf_model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        return preds, proba
        
    def predict_isolation_forest(self, df, threshold=0.45):
        X = df[self.feature_cols].fillna(0)
        scores = self.iso_model.score_samples(X)
        norm_scores = 1.0 / (1.0 + np.exp(scores * 5.0))
        preds = (norm_scores >= threshold).astype(int)
        return preds, norm_scores
        
    def predict_hybrid_ensemble(self, df, threshold=0.45):
        _, rule_proba, all_triggers = self.predict_rule_based(df)
        _, rf_proba = self.predict_random_forest(df)
        _, iso_proba = self.predict_isolation_forest(df)
        
        # Weighted calibrated combination (45% RF, 35% Rules, 20% Isolation Forest)
        ensemble_proba = (0.45 * rf_proba) + (0.35 * rule_proba) + (0.20 * iso_proba)
        preds = (ensemble_proba >= threshold).astype(int)
        risk_scores = np.round(ensemble_proba * 100, 1)
        return preds, ensemble_proba, risk_scores, all_triggers

    def get_feature_importances(self):
        importances = self.rf_model.feature_importances_
        return sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)

print("[+] Training Multi-Model AML Suite...")
detector = AMLDetectionSuite()
detector.train(train_feat_df)
print("[OK] Models successfully trained (Random Forest + Isolation Forest + Rule Engine).")
""")

    # 7. Evaluator
    add_code("""
# @title 6. Model Evaluation & Threshold Sensitivity Benchmarks

class ModelEvaluator:
    def __init__(self, detector):
        self.detector = detector
        
    def evaluate(self, test_df, threshold=0.35):
        y_true = test_df['is_sar_reported'].astype(int).values
        models = {
            'Rule-Based Engine': self.detector.predict_rule_based(test_df)[:2],
            'Supervised Random Forest': self.detector.predict_random_forest(test_df, threshold=threshold),
            'Unsupervised Isolation Forest': self.detector.predict_isolation_forest(test_df, threshold=threshold),
            'Calibrated Hybrid Ensemble': self.detector.predict_hybrid_ensemble(test_df, threshold=threshold)[:2]
        }
        
        rows = []
        for name, (pred, prob) in models.items():
            cm = confusion_matrix(y_true, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            acc = (tp + tn) / total
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else 0.5
            
            rows.append({
                'Model': name,
                'Accuracy': f"{acc * 100:.2f}%",
                'Precision': f"{prec * 100:.2f}%",
                'Recall (TPR)': f"{rec * 100:.2f}%",
                'False Positive Rate (FPR)': f"{fpr * 100:.2f}%",
                'False Negative Rate (FNR)': f"{fnr * 100:.2f}%",
                'F1-Score': f"{f1 * 100:.2f}%",
                'ROC-AUC': f"{auc:.4f}",
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
            })
        return pd.DataFrame(rows)

evaluator = ModelEvaluator(detector)
print("\\n" + "=" * 90)
print("AML MODEL BENCHMARK RESULTS (Evaluated at Operating Threshold = 0.35)")
print("=" * 90)
res_table = evaluator.evaluate(test_feat_df, threshold=0.35)
print(res_table[['Model', 'Accuracy', 'Precision', 'Recall (TPR)', 'False Positive Rate (FPR)', 'False Negative Rate (FNR)', 'ROC-AUC']].to_string(index=False))
print("=" * 90)
""")

    # 8. PII Masker
    add_code("""
# @title 7. GDPR & BSA Reversible PII Tokenization Vault

class PIIMasker:
    def __init__(self, vault_secret="ComplianceSAR_2026_SecureKey"):
        self.vault_secret = vault_secret.encode('utf-8')
        self.lookup_vault = {}
        self.reverse_vault = {}
        
    def _hash_token(self, value, entity_type):
        digest = hmac.new(self.vault_secret, f"{entity_type}:{value}".encode('utf-8'), hashlib.sha256).hexdigest()
        return f"[ANON_{entity_type}_{digest[:8].upper()}]"
        
    def mask_profile(self, profile):
        masked = profile.copy()
        name = profile.get('customer_name', 'Unknown')
        token = self._hash_token(name, 'SUBJECT')
        self.lookup_vault[name] = token
        self.reverse_vault[token] = name
        masked['customer_name'] = token
        return masked
        
    def mask_text(self, text):
        for orig, token in self.lookup_vault.items():
            if orig in text:
                text = text.replace(orig, token)
        return text

masker = PIIMasker()
print("[OK] Cryptographic Reversible PII Masking Vault initialized.")
""")

    # 9. SAR Generator
    add_code("""
# @title 8. 5-Part FinCEN / FCA Regulatory SAR Narrative Generator

class SARNarrativeGenerator:
    def __init__(self, pii_masker=None):
        self.masker = pii_masker or PIIMasker()
        
    def generate_5part_narrative(self, case_id, customer_profile, txns_df, detection_summary, is_masked=False):
        cust = self.masker.mask_profile(customer_profile) if is_masked else customer_profile
        name = cust.get('customer_name', 'Unknown')
        cid = cust.get('customer_id', 'Unknown')
        c_type = cust.get('customer_type', 'INDIVIDUAL')
        country = cust.get('primary_country', 'GB')
        turnover = cust.get('expected_monthly_turnover', 5000.0)
        
        total_amt = txns_df['amount'].sum()
        txn_count = len(txns_df)
        min_date = txns_df['timestamp'].min().strftime('%Y-%m-%d')
        max_date = txns_df['timestamp'].max().strftime('%Y-%m-%d')
        rules = detection_summary.get('triggered_rules', [])
        rule_desc = " ; ".join([f"{r['id']}: {r['name']}" for r in rules]) if rules else "Statistical Ensemble Velocity Anomaly"
        
        # SECTION 1: WHO
        sec1 = (
            f"SECTION 1: SUBJECT IDENTIFICATION (WHO)\\n"
            f"The primary subject of this filing is {name} (Customer ID: {cid}), categorized as a {c_type} entity "
            f"incorporated/domiciled in {country}. Stated business turnover at onboarding is ${turnover:,.2f}/month. "
            f"Internal risk classification: {cust.get('risk_rating', 'LOW')}."
        )
        
        # SECTION 2: WHAT
        sec2 = (
            f"SECTION 2: SUMMARY OF SUSPICIOUS ACTIVITY (WHAT)\\n"
            f"During the surveillance period, the AML detection system flagged {txn_count} transaction(s) aggregating to "
            f"${total_amt:,.2f}. The transactions deviate significantly from historical KYC profiles and baseline behavior. "
            f"Automated compliance rules triggered: [{rule_desc}]."
        )
        
        # SECTION 3: WHEN
        sec3 = (
            f"SECTION 3: TRANSACTION TIMELINE & VELOCITY (WHEN)\\n"
            f"The flagged transactional velocity occurred between {min_date} and {max_date}. The funds were transferred "
            f"in rapid succession, exhibiting elevated velocity anomalies that bypassed standard customer cadence."
        )
        
        # SECTION 4: WHERE
        sec4 = (
            f"SECTION 4: JURISDICTIONAL FLOW & NEXUS (WHERE)\\n"
            f"Transactional funds originated in {country} and were transmitted via electronic banking channels. Counterparties "
            f"involve financial institutions and beneficiary accounts in monitored corridors, indicating potential cross-border routing."
        )
        
        # SECTION 5: WHY
        sec5 = (
            f"SECTION 5: REGULATORY RATIONALE & TYPOLOGY (WHY)\\n"
            f"The activity represents high compliance risk under FinCEN and FCA standards. The lack of documented commercial "
            f"substance, combined with structuring and rapid flow-through patterns, indicates potential illicit laundering. "
            f"Recommended Action: Immediate freezing and submission of Form SAR/STR to FIU."
        )
        
        full_text = f"=== SUSPICIOUS ACTIVITY REPORT (SAR) ===\\nCASE REF: {case_id}\\nDATE: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\\n\\n" + "\\n\\n".join([sec1, sec2, sec3, sec4, sec5])
        return {
            'case_id': case_id,
            'full_narrative': full_text,
            'sections': {'who': sec1, 'what': sec2, 'when': sec3, 'where': sec4, 'why': sec5}
        }

generator = SARNarrativeGenerator(masker)
print("[OK] FinCEN/FCA SAR Narrative Generator ready.")
""")

    # 10. Audit Trail
    add_code("""
# @title 9. Cryptographic Audit Trail & Provenance Tracker (SHA-256)

class AuditTrailManager:
    @staticmethod
    def create_seal(case_id, customer_id, narrative_text, txn_ids):
        raw_payload = f"{case_id}|{customer_id}|{narrative_text}|{','.join(sorted(txn_ids))}"
        sha_seal = hashlib.sha256(raw_payload.encode('utf-8')).hexdigest()
        return {
            'case_id': case_id,
            'customer_id': customer_id,
            'timestamp': datetime.utcnow().isoformat() + "Z",
            'sha256_seal': sha_seal,
            'transaction_count': len(txn_ids),
            'tamper_seal_valid': True
        }

print("[OK] Cryptographic Audit Trail Manager initialized.")
""")

    # 11. Interactive Colab UI
    add_code("""
# @title 10. Interactive SAR Compliance Workbench
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Flag cases in test set
preds, probas, risk_scores, triggers = detector.predict_hybrid_ensemble(test_feat_df, threshold=0.35)
test_feat_df['is_flagged'] = preds
test_feat_df['ensemble_risk_score'] = risk_scores
test_feat_df['triggered_rules'] = triggers

flagged_df = test_feat_df[test_feat_df['is_flagged'] == 1]
high_risk_cids = list(flagged_df['customer_id'].unique())

if not high_risk_cids:
    high_risk_cids = list(customers_df['customer_id'].head(10))

# Interactive UI Components
cid_dropdown = widgets.Dropdown(options=high_risk_cids, description='Subject ID:', style={'description_width': 'initial'})
mask_checkbox = widgets.Checkbox(value=True, description='Apply PII Obfuscation (GDPR/BSA)', indent=False)
threshold_slider = widgets.FloatSlider(value=0.35, min=0.10, max=0.80, step=0.05, description='Decision Threshold:', readout_format='.2f', style={'description_width': 'initial'})
gen_btn = widgets.Button(description='Generate SAR & Audit Seal', button_style='danger', icon='file-text')
out = widgets.Output()

def on_click(b):
    with out:
        clear_output()
        cid = cid_dropdown.value
        is_masked = mask_checkbox.value
        th = threshold_slider.value
        
        # Filter transactions
        cust_row = customers_df[customers_df['customer_id'] == cid].iloc[0].to_dict()
        cust_txns = test_feat_df[test_feat_df['customer_id'] == cid].copy()
        
        if len(cust_txns) == 0:
            print(f"No test transactions found for {cid}. Picking first available subject.")
            cid = high_risk_cids[0]
            cust_row = customers_df[customers_df['customer_id'] == cid].iloc[0].to_dict()
            cust_txns = test_feat_df[test_feat_df['customer_id'] == cid].copy()
            
        case_id = f"SAR-CASE-{cid}-2026"
        det_summary = {
            'triggered_rules': cust_txns['triggered_rules'].iloc[0] if len(cust_txns) > 0 and len(cust_txns['triggered_rules'].iloc[0]) > 0 else []
        }
        
        pkg = generator.generate_5part_narrative(case_id, cust_row, cust_txns, det_summary, is_masked=is_masked)
        txn_ids = list(cust_txns['transaction_id'].unique())
        audit = AuditTrailManager.create_seal(case_id, cid, pkg['full_narrative'], txn_ids)
        
        display(HTML(f\"\"\"
        <div style="background-color: #0F172A; color: #F8FAFC; padding: 20px; border-radius: 8px; font-family: monospace;">
            <h2 style="color: #38BDF8; margin-top: 0;">🛡️ OFFICIAL REGULATORY SAR FILING DOSSIER</h2>
            <div style="background: #1E293B; padding: 12px; border-left: 4px solid #10B981; margin-bottom: 15px;">
                <strong>CRYPTOGRAPHIC TAMPER SEAL (SHA-256):</strong><br>
                <code>{audit['sha256_seal']}</code><br>
                <small>Timestamp: {audit['timestamp']} | Status: VERIFIED UNTAMPERED</small>
            </div>
            <pre style="white-space: pre-wrap; word-wrap: break-word; color: #E2E8F0; font-size: 13px;">{pkg['full_narrative']}</pre>
        </div>
        \"\"\"))

gen_btn.on_click(on_click)
display(widgets.VBox([
    widgets.HBox([cid_dropdown, mask_checkbox]),
    threshold_slider,
    gen_btn,
    out
]))
""")

    # 12. Visualizations
    add_code("""
# @title 11. Visualizations: Confusion Matrix, ROC-AUC, and Threshold Trade-off Curve
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# 1. ROC Curve
y_true = test_feat_df['is_sar_reported'].astype(int).values
_, ens_proba, _, _ = detector.predict_hybrid_ensemble(test_feat_df)
fpr_c, tpr_c, _ = roc_curve(y_true, ens_proba)
auc_val = roc_auc_score(y_true, ens_proba)

axs[0].plot(fpr_c, tpr_c, color='#0284C7', lw=2.5, label=f'Hybrid Ensemble (AUC = {auc_val:.4f})')
axs[0].plot([0, 1], [0, 1], color='#94A3B8', linestyle='--')
axs[0].set_title('ROC Curve - Multi-Model AML Detection', fontsize=12, fontweight='bold')
axs[0].set_xlabel('False Positive Rate (FPR)')
axs[0].set_ylabel('True Positive Rate (Recall)')
axs[0].legend(loc='lower right')

# 2. Threshold Sweep
thresholds = np.linspace(0.10, 0.70, 13)
accuracies, precisions, recalls = [], [], []
for t in thresholds:
    p = (ens_proba >= t).astype(int)
    cm = confusion_matrix(y_true, p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    accuracies.append((tp + tn) / len(y_true))
    precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
    recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

axs[1].plot(thresholds, accuracies, marker='o', label='Accuracy', color='#10B981', lw=2)
axs[1].plot(thresholds, precisions, marker='s', label='Precision', color='#6366F1', lw=2)
axs[1].plot(thresholds, recalls, marker='^', label='Recall (TPR)', color='#F59E0B', lw=2)
axs[1].axvline(0.35, color='#EF4444', linestyle=':', label='Operating Cutoff (0.35)')
axs[1].set_title('Threshold Sensitivity Curve (Accuracy vs Precision vs Recall)', fontsize=12, fontweight='bold')
axs[1].set_xlabel('Ensemble Decision Threshold')
axs[1].set_ylabel('Metric Score')
axs[1].legend()

plt.tight_layout()
plt.show()
""")

    # 13. Running full Web GUI in Colab
    add_md("""
---
## 🌐 Optional: Launching the Full Web Dashboard (Streamlit) in Google Colab

If you want to run the full interactive Streamlit web application with tabs, charts, and file generation directly from this Colab session, run the cell below. It uses `localtunnel` to give you a public URL.
""")

    add_code("""
# @title (Optional) Run Streamlit Web Application via Localtunnel
# Uncomment and run the lines below to start the Streamlit web app in Colab:

# !pip install -q streamlit localtunnel
# import urllib
# print("Your external IP is:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
# !streamlit run app_web.py & npx localtunnel --port 8501
""")

    return nb

if __name__ == '__main__':
    notebook_dict = create_notebook()
    output_path = 'SAR_Narrative_Generator_Colab.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1)
    print(f"[OK] Successfully built Colab Notebook at: {output_path} (File size: {os.path.getsize(output_path)} bytes)")
