"""
Enterprise AML Compliance Suite: AI SAR Narrative Generator & Audit Trail
Single-Cell Standalone Google Colab Script
Paste this directly into any empty Google Colab cell and press Shift+Enter!
"""

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

try:
    import ipywidgets as widgets
    from IPython.display import display, HTML, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

print("[1/6] Initializing AML surveillance environment...")
HIGH_RISK_JURISDICTIONS = {'KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG'}

# ==========================================
# 1. SYNTHETIC GENERATOR (5 AML TYPOLOGIES)
# ==========================================
def generate_customer_profiles(num_customers=80, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    male_names = ["James", "Liam", "Noah", "William", "Michael", "Arthur"]
    female_names = ["Emma", "Olivia", "Ava", "Sophia", "Isabella", "Charlotte"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Sterling", "Taylor", "Evans", "Patel"]
    corp_names = ["Apex Holdings Ltd", "Vanguard Logistics LLC", "Sterling Global Trade Inc", "Atlas Capital Partners", "Oasis Offshore Inc"]
    customers = []
    for i in range(1, num_customers + 1):
        cid = f"CUST-{i:04d}"
        is_corp = (i % 2 == 0)
        c_type = "CORPORATE" if is_corp else "INDIVIDUAL"
        if is_corp:
            name = corp_names[(i // 2) % len(corp_names)] + f" #{i}"
            turnover = round(random.uniform(50000.0, 500000.0), 2)
            risk = ['LOW', 'MEDIUM', 'HIGH'][i % 3]
        else:
            first_name = male_names[(i // 4) % len(male_names)] if (i // 2) % 2 == 0 else female_names[(i // 4) % len(female_names)]
            name = f"{first_name} {last_names[i % len(last_names)]}"
            turnover = round(random.uniform(2500.0, 15000.0), 2)
            risk = ['LOW', 'MEDIUM', 'HIGH'][(i + 1) % 3]
        country = random.choices(["GB", "US", "DE", "KY", "PA", "AE"], weights=[0.4, 0.3, 0.15, 0.05, 0.05, 0.05])[0]
        customers.append({
            'customer_id': cid, 'customer_name': name, 'customer_type': c_type,
            'expected_monthly_turnover': turnover, 'risk_rating': risk,
            'primary_country': country, 'kyc_status': "VERIFIED"
        })
    return pd.DataFrame(customers)

def inject_synthetic_noise(df, noise_ratio=0.08, seed=42):
    """Injects realistic banking noise: missing IPs/accounts, text whitespace/casing inconsistencies, micro-cents jitter."""
    d = df.copy()
    np.random.seed(seed)
    random.seed(seed)
    n = len(d)
    
    # Missing IPs and accounts
    d.loc[np.random.rand(n) < (noise_ratio * 0.6), 'device_ip'] = np.nan
    d.loc[np.random.rand(n) < (noise_ratio * 0.4), 'beneficiary_account'] = None
    
    # Casing and whitespace noise
    if 'channel' in d.columns:
        dirty_c = [' online_banking ', '  SWIFT  ', 'branch_counter', 'mobile_app ', ' ATM  ']
        for idx in d[np.random.rand(n) < (noise_ratio * 0.8)].index:
            d.at[idx, 'channel'] = random.choice(dirty_c)
            
    if 'transaction_type' in d.columns:
        dirty_t = [' wire_transfer ', '  ach_transfer  ', 'CASH_DEPOSIT ', '  peer_to_peer', 'crypto_purchase ']
        for idx in d[np.random.rand(n) < (noise_ratio * 0.8)].index:
            d.at[idx, 'transaction_type'] = random.choice(dirty_t)
            
    # Micro-cents jitter
    if 'amount' in d.columns:
        for idx in d[np.random.rand(n) < (noise_ratio * 0.6)].index:
            d.at[idx, 'amount'] = round(float(d.at[idx, 'amount']) + random.uniform(-0.45, 0.45), 3)
            
    return d

def generate_synthetic_transactions(customers_df, total_records=12000, suspicious_ratio=0.50, seed=42, add_noise=True):
    random.seed(seed)
    np.random.seed(seed)
    start_date = datetime(2026, 1, 1, 0, 0, 0)
    txns = []
    suspicious_count = int(total_records * suspicious_ratio)
    normal_count = total_records - suspicious_count
    channels = ["ONLINE_BANKING", "SWIFT", "MOBILE_APP", "BRANCH", "ATM"]
    txn_types = ["WIRE_OUT", "WIRE_IN", "TRANSFER", "CASH_DEPOSIT", "CASH_WITHDRAWAL"]
    
    for i in range(normal_count):
        cust = customers_df.iloc[i % len(customers_df)]
        t_stamp = start_date + timedelta(seconds=random.randint(0, 90 * 86400))
        amt = round(random.uniform(25.0, min(cust['expected_monthly_turnover'] * 0.40, 8000.0)), 2)
        txns.append({
            'transaction_id': f"TXN-{random.randint(10000000, 99999999)}", 'timestamp': t_stamp,
            'customer_id': cust['customer_id'], 'originator_name': cust['customer_name'],
            'originator_country': cust['primary_country'], 'beneficiary_name': f"Vendor-{random.randint(100, 999)}",
            'beneficiary_country': cust['primary_country'], 'amount': amt, 'currency': 'USD',
            'channel': channels[i % len(channels)], 'transaction_type': txn_types[i % len(txn_types)],
            'device_ip': f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
            'is_sar_reported': 0, 'aml_typology': 'NORMAL'
        })
        
    typologies = ['STRUCTURING', 'RAPID_FLOW_THROUGH', 'MULE_FAN_IN', 'ROUND_TRIPPING_SHELL', 'DORMANT_SPIKE']
    base_quota = suspicious_count // len(typologies)
    remainder = suspicious_count % len(typologies)
    c_idx = 0
    
    for t_i, typology in enumerate(typologies):
        quota = base_quota + (1 if t_i < remainder else 0)
        for _ in range(quota):
            cust = customers_df.iloc[c_idx % len(customers_df)]
            c_idx += 1
            t_stamp = start_date + timedelta(seconds=random.randint(0, 90 * 86400))
            orig_country, dest_country = cust['primary_country'], cust['primary_country']
            if typology == 'STRUCTURING':
                amt = round(random.uniform(9000.0, 9950.0), 2)
                c_type, t_type = 'BRANCH', 'CASH_DEPOSIT'
            elif typology == 'RAPID_FLOW_THROUGH':
                amt = round(random.uniform(25000.0, 150000.0), 2)
                c_type, t_type = 'SWIFT', 'WIRE_OUT'
                dest_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
            elif typology == 'MULE_FAN_IN':
                amt = round(random.uniform(4000.0, 8500.0), 2)
                c_type, t_type = 'MOBILE_APP', 'TRANSFER'
            elif typology == 'ROUND_TRIPPING_SHELL':
                amt = round(random.uniform(50000.0, 250000.0), 2)
                c_type, t_type = 'SWIFT', 'WIRE_OUT'
                orig_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
                dest_country = random.choice(list(HIGH_RISK_JURISDICTIONS))
            else:
                amt = round(random.uniform(30000.0, 120000.0), 2)
                c_type, t_type = 'ONLINE_BANKING', 'WIRE_OUT'
            txns.append({
                'transaction_id': f"TXN-{random.randint(10000000, 99999999)}", 'timestamp': t_stamp,
                'customer_id': cust['customer_id'], 'originator_name': cust['customer_name'],
                'originator_country': orig_country, 'beneficiary_name': f"Entity-{random.randint(1000, 9999)}",
                'beneficiary_country': dest_country, 'amount': amt, 'currency': 'USD',
                'channel': c_type, 'transaction_type': t_type,
                'device_ip': f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
                'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                'is_sar_reported': 1, 'aml_typology': typology
            })
    df_out = pd.DataFrame(txns).sort_values('timestamp').reset_index(drop=True)
    if add_noise:
        df_out = inject_synthetic_noise(df_out, noise_ratio=0.08, seed=seed)
    return df_out

# ==========================================
# 2. DATA PREPROCESSOR (6-STAGE SANITIZATION)
# ==========================================
class AMLDataPreprocessor:
    def preprocess(self, df):
        d = df.copy()
        rep = {'missing_imputed': 0, 'casing_fixes': 0, 'outliers_capped': 0}
        
        # 1. Amount coercion
        d['amount'] = pd.to_numeric(d['amount'], errors='coerce').fillna(250.0).clip(lower=1.0)
        
        # 2. Text normalization
        for col in ['channel', 'transaction_type', 'originator_country', 'beneficiary_country']:
            if col in d.columns:
                orig = d[col].astype(str)
                cleaned = orig.str.strip().str.upper()
                rep['casing_fixes'] += int((orig != cleaned).sum())
                d[col] = cleaned
                
        # 3. Missing value imputation
        if 'device_ip' in d.columns:
            rep['missing_imputed'] += int(d['device_ip'].isna().sum())
            d['device_ip'] = d['device_ip'].fillna('UNKNOWN_IP')
        if 'beneficiary_account' in d.columns:
            rep['missing_imputed'] += int(d['beneficiary_account'].isna().sum())
            d['beneficiary_account'] = d['beneficiary_account'].fillna('UNSPECIFIED_ACCT')
            
        # 4. Outlier capping (99.9th percentile)
        q999 = float(d['amount'].quantile(0.999))
        cap = max(q999, 450000.0)
        outlier_mask = d['amount'] > cap
        rep['outliers_capped'] = int(outlier_mask.sum())
        d.loc[outlier_mask, 'amount'] = cap
        
        # 5. Timestamp parse & sort
        d['dt'] = pd.to_datetime(d['timestamp'], errors='coerce')
        d = d.sort_values(['customer_id', 'dt']).reset_index(drop=True)
        return d, rep

# ==========================================
# 3. FEATURE EXTRACTOR
# ==========================================
class AMLFeatureExtractor:
    def __init__(self, customer_df):
        self.meta = {row['customer_id']: row.to_dict() for _, row in customer_df.iterrows()}
        self.preprocessor = AMLDataPreprocessor()
        
    def extract_features(self, transactions_df):
        df, _ = self.preprocessor.preprocess(transactions_df)
        risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        df['expected_turnover'] = [self.meta.get(c, {}).get('expected_monthly_turnover', 5000.0) for c in df['customer_id']]
        df['customer_risk_level'] = [risk_map.get(self.meta.get(c, {}).get('risk_rating', 'LOW'), 1) for c in df['customer_id']]
        df['is_corporate_account'] = [1 if self.meta.get(c, {}).get('customer_type', 'INDIVIDUAL') != 'INDIVIDUAL' else 0 for c in df['customer_id']]
        df['amount_log'] = np.log1p(df['amount'].astype(float))
        df['turnover_ratio'] = df['amount'] / np.maximum(df['expected_turnover'], 100.0)
        df['is_structuring_range'] = ((df['amount'] >= 9000.0) & (df['amount'] <= 9999.0)).astype(int)
        df['structuring_proximity'] = np.where(df['is_structuring_range'] == 1, 1.0 - (np.abs(df['amount'] - 9500.0) / 500.0), 0.0)
        df['is_round_amount'] = (((df['amount'] >= 10000.0) & (df['amount'] % 10000.0 == 0.0)) | ((df['amount'] >= 5000.0) & (df['amount'] % 5000.0 == 0.0))).astype(int)
        df['is_high_risk_origin'] = df['originator_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_high_risk_dest'] = df['beneficiary_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_cross_border'] = (df['originator_country'] != df['beneficiary_country']).astype(int)
        df['jurisdiction_risk_score'] = df['is_high_risk_origin'] * 2 + df['is_high_risk_dest'] * 2 + df['is_cross_border']
        df['is_cash_or_crypto'] = df['transaction_type'].isin({'CASH_DEPOSIT', 'CASH_WITHDRAWAL', 'CRYPTO_PURCHASE'}).astype(int)
        df['is_swift_wire'] = df['channel'].isin({'SWIFT'}).astype(int)
        df['hour'] = df['dt'].dt.hour
        df['day_of_week'] = df['dt'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_hours'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        df['orig_idx'] = np.arange(len(df))
        df_sorted = df.sort_values(['customer_id', 'dt'])
        roll = df_sorted.set_index('dt').groupby('customer_id')['amount'].rolling('24h').agg(['count', 'sum']).reset_index()
        df_sorted['velocity_24h_count'] = roll['count'].values
        df_sorted['velocity_24h_sum'] = roll['sum'].values
        df_sorted['velocity_24h_sum_log'] = np.log1p(df_sorted['velocity_24h_sum'])
        prev_dt = df_sorted.groupby('customer_id')['dt'].shift(1)
        prev_amt = df_sorted.groupby('customer_id')['amount'].shift(1)
        td = (df_sorted['dt'] - prev_dt).dt.total_seconds()
        df_sorted['rapid_flow_through_flag'] = ((td <= 21600) & (td > 0) & (df_sorted['amount'] > 10000) & ((df_sorted['amount'] - prev_amt).abs() / np.maximum(df_sorted['amount'], 1.0) < 0.20)).astype(int)
        return df_sorted.sort_values('orig_idx').drop(columns=['orig_idx', 'dt'])

    @staticmethod
    def get_feature_columns():
        return ['amount_log', 'turnover_ratio', 'is_structuring_range', 'structuring_proximity', 'is_round_amount',
                'is_high_risk_origin', 'is_high_risk_dest', 'is_cross_border', 'jurisdiction_risk_score',
                'is_cash_or_crypto', 'is_swift_wire', 'customer_risk_level', 'is_corporate_account',
                'is_weekend', 'is_night_hours', 'velocity_24h_count', 'velocity_24h_sum_log', 'rapid_flow_through_flag']

# ==========================================
# 3. RULE ENGINE & MULTI-MODEL DETECTOR
# ==========================================
class AMLDetectionSuite:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        self.iso = IsolationForest(n_estimators=100, contamination=0.15, random_state=42, n_jobs=-1)
        self.feature_cols = AMLFeatureExtractor.get_feature_columns()
    def train(self, train_df):
        X = train_df[self.feature_cols].fillna(0)
        y = train_df['is_sar_reported'].astype(int)
        self.rf.fit(X, y)
        self.iso.fit(X)
    def evaluate_rules(self, row):
        t = []
        amt = float(row.get('amount', 0))
        if 9000.0 <= amt <= 9999.0 and row.get('is_cash_or_crypto', 0) == 1:
            t.append({'id': 'AML-RULE-101', 'name': 'Cash Structuring Threshold Avoidance', 'weight': 0.85})
        if row.get('rapid_flow_through_flag', 0) == 1:
            t.append({'id': 'AML-RULE-102', 'name': 'Rapid In-and-Out Flow-Through', 'weight': 0.80})
        if row.get('velocity_24h_count', 0) >= 5 or row.get('turnover_ratio', 0) >= 3.0:
            t.append({'id': 'AML-RULE-103', 'name': 'Velocity Spike', 'weight': 0.65})
        if row.get('jurisdiction_risk_score', 0) >= 2:
            t.append({'id': 'AML-RULE-104', 'name': 'High-Risk Jurisdictional Exposure', 'weight': 0.75})
        return t
    def predict(self, df, threshold=0.35):
        X = df[self.feature_cols].fillna(0)
        rf_p = self.rf.predict_proba(X)[:, 1]
        iso_s = self.iso.score_samples(X)
        iso_p = 1.0 / (1.0 + np.exp(iso_s * 5.0))
        rule_p, all_triggers = [], []
        for _, row in df.iterrows():
            triggers = self.evaluate_rules(row)
            all_triggers.append(triggers)
            rule_p.append(max([x['weight'] for x in triggers]) if triggers else 0.0)
        rule_p = np.array(rule_p)
        ens_p = (0.45 * rf_p) + (0.35 * rule_p) + (0.20 * iso_p)
        preds = (ens_p >= threshold).astype(int)
        return preds, ens_p, np.round(ens_p * 100, 1), all_triggers

# ==========================================
# 4. PII MASKER, SAR NARRATIVE & AUDIT SEAL
# ==========================================
class ComplianceModule:
    @staticmethod
    def mask_name(name):
        h = hashlib.sha256(f"VAULT:{name}".encode()).hexdigest()[:8].upper()
        return f"[ANON_SUBJECT_{h}]"
    @staticmethod
    def generate_sar(case_id, cust, txns_df, triggers, is_masked=True):
        name = ComplianceModule.mask_name(cust['customer_name']) if is_masked else cust['customer_name']
        amt_sum = txns_df['amount'].sum()
        min_d = txns_df['timestamp'].min().strftime('%Y-%m-%d')
        max_d = txns_df['timestamp'].max().strftime('%Y-%m-%d')
        rule_str = " ; ".join([f"{r['id']}: {r['name']}" for r in triggers]) if triggers else "Machine Learning Anomaly"
        text = (
            f"=== SUSPICIOUS ACTIVITY REPORT (SAR) ===\\nCASE: {case_id} | DATE: {datetime.utcnow().strftime('%Y-%m-%d UTC')}\\n\\n"
            f"SECTION 1 (WHO): Subject: {name} (ID: {cust['customer_id']}), {cust['customer_type']} entity, {cust['primary_country']}. Turnover: ${cust['expected_monthly_turnover']:,.2f}/mo.\\n\\n"
            f"SECTION 2 (WHAT): Surveillance identified {len(txns_df)} transactions totaling ${amt_sum:,.2f} exceeding KYC baseline. Triggers: [{rule_str}].\\n\\n"
            f"SECTION 3 (WHEN): Activity transpired from {min_d} to {max_d} with concentrated velocity bypassing normal customer behavior.\\n\\n"
            f"SECTION 4 (WHERE): Funds routed through electronic banking corridors and cross-border counterparties.\\n\\n"
            f"SECTION 5 (WHY): Unexplained economic substance and structuring indicators present high money laundering risk. Action: Submit Form SAR to FIU."
        )
        seal = hashlib.sha256(f"{case_id}|{cust['customer_id']}|{text}".encode()).hexdigest()
        return text, seal

# ==========================================
# 5. EXECUTION & PIPELINE INITIALIZATION
# ==========================================
print("[2/6] Generating synthetic datasets (Normal + 5 AML Typologies)...")
customers_df = generate_customer_profiles(num_customers=80, seed=42)
train_df = generate_synthetic_transactions(customers_df, total_records=15000, suspicious_ratio=0.50, seed=42)
test_df = generate_synthetic_transactions(customers_df, total_records=2500, suspicious_ratio=0.50, seed=99)

print("[3/6] Extracting quantitative compliance features...")
fe = AMLFeatureExtractor(customers_df)
train_feat = fe.extract_features(train_df)
test_feat = fe.extract_features(test_df)

print("[4/6] Training multi-model AML detection suite (Random Forest + Isolation Forest + Rules)...")
detector = AMLDetectionSuite()
detector.train(train_feat)

print("[5/6] Benchmarking model performance on unseen test data...")
y_true = test_feat['is_sar_reported'].values
preds, probas, scores, triggers = detector.predict(test_feat, threshold=0.35)
cm = confusion_matrix(y_true, preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
acc = (tp + tn) / len(y_true)
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
auc = roc_auc_score(y_true, probas)

print("\\n" + "=" * 80)
print("MULTI-MODEL AML PERFORMANCE REPORT (Threshold = 0.35)")
print("=" * 80)
print(f"  Accuracy:                  {acc * 100:.2f}%")
print(f"  Precision:                 {prec * 100:.2f}%")
print(f"  Recall (Detection Rate):   {rec * 100:.2f}%")
print(f"  False Positive Rate (FPR): {fpr * 100:.2f}% (Customer False Alarms)")
print(f"  False Negative Rate (FNR): {fnr * 100:.2f}% (Regulatory Missed Risk)")
print(f"  ROC-AUC:                   {auc:.4f}")
print("=" * 80)

# Flag subjects
test_feat['is_flagged'] = preds
flagged = test_feat[test_feat['is_flagged'] == 1]
high_risk_cids = list(flagged['customer_id'].unique())
sample_cid = high_risk_cids[0]

print(f"\\n[6/6] Generating Demo SAR Dossier for Suspicious Subject: {sample_cid}...")
cust_profile = customers_df[customers_df['customer_id'] == sample_cid].iloc[0].to_dict()
cust_txns = test_feat[test_feat['customer_id'] == sample_cid]
all_t = [t for row_t in triggers for t in row_t if len(t) > 0]
narrative_text, sha_seal = ComplianceModule.generate_sar(f"SAR-{sample_cid}-2026", cust_profile, cust_txns, all_t[:2], is_masked=True)

print("\\n" + "#" * 80)
print(f"CRYPTOGRAPHIC AUDIT SEAL (SHA-256): {sha_seal}")
print("#" * 80)
print(narrative_text)
print("#" * 80)
print("\\n[SUCCESS] Pipeline executed cleanly in Google Colab!")
