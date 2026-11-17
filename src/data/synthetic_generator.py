"""
Synthetic Banking Transaction & AML Typology Generator
Generates realistic financial transaction streams with legitimate consumer/business
activity and realistic Money Laundering typologies (Structuring, Layering, Mule Networks,
Shell Company Round-Tripping, and Dormant Influx).
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

HIGH_RISK_JURISDICTIONS = ['KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG']
NORMAL_JURISDICTIONS = ['GB', 'US', 'DE', 'FR', 'NL', 'CA', 'AU', 'SG', 'CH', 'JP']

MALE_FIRST_NAMES = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel']
FEMALE_FIRST_NAMES = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica', 'Sarah', 'Karen', 'Nancy', 'Margaret']
FIRST_NAMES = MALE_FIRST_NAMES + FEMALE_FIRST_NAMES
LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
              'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
              'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Perez', 'Thompson', 'White']

BUSINESS_PREFIXES = ['Apex', 'Vanguard', 'Global', 'Summit', 'Nexus', 'Horizon', 'Meridian', 'Zenith', 'Sterling', 'Atlas']
BUSINESS_SUFFIXES = ['Holdings Ltd', 'Trading FZE', 'Logistics LLC', 'Capital Partners', 'Consulting Group', 'Imports Inc', 'Enterprises']

def generate_random_ip():
    return f"{random.randint(11, 220)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"

def generate_customer_profiles(num_customers=300, seed=42):
    """
    Generates balanced, unbiased customer KYC baseline profiles:
    - 50% Individual retail accounts, 50% Corporate/SME commercial accounts
    - 50/50 Gender representation in individual accounts
    - Balanced stratification across LOW, MEDIUM, and HIGH risk tiers
    - Diverse jurisdictional presence across major commercial regions
    """
    random.seed(seed)
    np.random.seed(seed)
    
    customers = []
    for i in range(1, num_customers + 1):
        cust_id = f"CUST-{i:04d}"
        # Exact 50/50 balance between Corporate and Individual accounts
        is_corp = (i % 2 == 0)
        
        if is_corp:
            b_prefix = BUSINESS_PREFIXES[(i // 2) % len(BUSINESS_PREFIXES)]
            b_suffix = BUSINESS_SUFFIXES[(i // 2) % len(BUSINESS_SUFFIXES)]
            name = f"{b_prefix} {b_suffix}"
            c_type = ['SME_BUSINESS', 'COMMERCIAL_ENTERPRISE', 'OFFSHORE_CORP'][i % 3]
            expected_turnover = round(float(np.random.lognormal(mean=11.5, sigma=0.8)), 2) # ~$50k to $500k
            tax_id = f"TAX-CORP-{random.randint(100000, 999999)}"
            risk_rating = ['LOW', 'MEDIUM', 'HIGH'][i % 3]
        else:
            # 50/50 Gender split
            if (i // 2) % 2 == 0:
                first_name = MALE_FIRST_NAMES[(i // 4) % len(MALE_FIRST_NAMES)]
            else:
                first_name = FEMALE_FIRST_NAMES[(i // 4) % len(FEMALE_FIRST_NAMES)]
            last_name = LAST_NAMES[i % len(LAST_NAMES)]
            name = f"{first_name} {last_name}"
            c_type = 'INDIVIDUAL'
            expected_turnover = round(float(np.random.lognormal(mean=8.2, sigma=0.6)), 2) # ~$2k to $15k
            tax_id = f"TAX-IND-{random.randint(100000, 999999)}"
            risk_rating = ['LOW', 'MEDIUM', 'HIGH'][(i + 1) % 3]
            
        country = random.choices(
            NORMAL_JURISDICTIONS + HIGH_RISK_JURISDICTIONS,
            weights=[0.85/len(NORMAL_JURISDICTIONS)]*len(NORMAL_JURISDICTIONS) + [0.15/len(HIGH_RISK_JURISDICTIONS)]*len(HIGH_RISK_JURISDICTIONS)
        )[0]
        
        kyc_status = random.choices(['VERIFIED', 'ENHANCED_DUE_DILIGENCE', 'REQUIRES_REFRESH'], weights=[0.8, 0.15, 0.05])[0]
        account_no = f"ACCT-{random.randint(10000000, 99999999)}"
        opened_days_ago = random.randint(180, 2500)
        opened_date = (datetime.now() - timedelta(days=opened_days_ago)).strftime('%Y-%m-%d')
        
        customers.append({
            'customer_id': cust_id,
            'customer_name': name,
            'customer_type': c_type,
            'account_number': account_no,
            'primary_country': country,
            'expected_monthly_turnover': expected_turnover,
            'risk_rating': risk_rating,
            'kyc_status': kyc_status,
            'tax_id': tax_id,
            'account_opened_date': opened_date
        })
        
    return pd.DataFrame(customers)

def inject_synthetic_noise(transactions_df, noise_ratio=0.08, seed=42):
    """
    Injects realistic banking noise into synthetic transaction stream:
    1. Missing/Null values in optional fields (device_ip, beneficiary_account, beneficiary_name)
    2. Inconsistent text casing and whitespace padding (dirty channel / transaction_type formatting)
    3. Micro-cents jitter on amounts (currency conversion artifacts)
    4. Benign high-value extreme outlier transactions (false alarm edge cases)
    5. Slight out-of-order timestamp jitter (network delay)
    """
    if transactions_df is None or len(transactions_df) == 0:
        return transactions_df
        
    df = transactions_df.copy()
    np.random.seed(seed)
    random.seed(seed)
    n = len(df)
    
    # 1. Missing / Null Values
    mask_ip = np.random.rand(n) < (noise_ratio * 0.6)
    df.loc[mask_ip, 'device_ip'] = np.nan
    
    mask_ba = np.random.rand(n) < (noise_ratio * 0.4)
    df.loc[mask_ba, 'beneficiary_account'] = None
    
    mask_bn = np.random.rand(n) < (noise_ratio * 0.3)
    df.loc[mask_bn, 'beneficiary_name'] = ""
    
    # 2. Text Inconsistencies & Whitespace Padding
    if 'channel' in df.columns:
        dirty_channel_mask = np.random.rand(n) < (noise_ratio * 0.8)
        dirty_variants = [' online_banking ', '  SWIFT  ', 'branch_counter', 'mobile_app ', ' ATM  ']
        for idx in df[dirty_channel_mask].index:
            df.at[idx, 'channel'] = random.choice(dirty_variants)
            
    if 'transaction_type' in df.columns:
        dirty_type_mask = np.random.rand(n) < (noise_ratio * 0.8)
        dirty_types = [' wire_transfer ', '  ach_transfer  ', 'CASH_DEPOSIT ', '  peer_to_peer', 'crypto_purchase ']
        for idx in df[dirty_type_mask].index:
            df.at[idx, 'transaction_type'] = random.choice(dirty_types)
            
    if 'originator_country' in df.columns:
        dirty_c_mask = np.random.rand(n) < (noise_ratio * 0.3)
        for idx in df[dirty_c_mask].index:
            c = str(df.at[idx, 'originator_country'])
            df.at[idx, 'originator_country'] = c.lower() if random.random() < 0.5 else f" {c} "

    # 3. Micro-Cents Jitter on Amounts (3 decimal places or small jitter)
    if 'amount' in df.columns:
        jitter_mask = np.random.rand(n) < (noise_ratio * 0.7)
        for idx in df[jitter_mask].index:
            orig_amt = float(df.at[idx, 'amount'])
            df.at[idx, 'amount'] = round(orig_amt + random.uniform(-0.45, 0.45), 3)

    # 4. Benign Extreme Outlier Transactions (Legitimate high-value normal transactions)
    benign_normal_indices = df[df['is_sar_reported'] == 0].index
    if len(benign_normal_indices) > 10:
        outlier_sample = np.random.choice(benign_normal_indices, size=min(max(1, int(n * 0.005)), 10), replace=False)
        for idx in outlier_sample:
            df.at[idx, 'amount'] = round(random.uniform(120000.0, 280000.0), 2)
            df.at[idx, 'red_flag_notes'] = 'Authorized high-value corporate treasury reallocation (benign high-turnover outlier).'

    return df

def generate_synthetic_transactions(customer_df, total_records=6000, suspicious_ratio=0.50, seed=42, start_counter=1, start_date=None, add_noise=True, noise_ratio=0.08):
    """
    Generates statistically balanced, unbiased financial transactions:
    - Default 1:1 Class Balance: 50% Legitimate Normal (0) and 50% Suspicious AML (1)
    - Exact 20% Typology Balance: Equal quota (10% of total records) across all 5 typologies:
      1. STRUCTURING: repetitive sub-threshold cash deposits ($9,100 - $9,950)
      2. RAPID_FLOW_THROUGH: immediate fund movement through account in <2 hours
      3. MULE_FAN_IN: multiple low-value P2P deposits followed by sweep wire/crypto
      4. ROUND_TRIPPING_SHELL: high-value exact round sum wires ($50k - $250k) to offshore shells
      5. DORMANT_SPIKE: sudden 15x-40x volume spike above customer monthly baseline
    - Unbiased Customer Sampling: Activity distributed evenly across all customer risk tiers
      (LOW, MEDIUM, HIGH) and entity types (Individual, Corporate).
    - Balanced Channel Representation: Even spread across Mobile, Online, Branch, ATM, SWIFT.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    records = []
    txn_counter = start_counter
    
    customers_dict = customer_df.set_index('customer_id').to_dict(orient='index')
    cust_ids = list(customers_dict.keys())
    
    num_suspicious = int(total_records * suspicious_ratio)
    num_normal = total_records - num_suspicious
    
    # -------------------------------------------------------------
    # 1. BALANCED NORMAL TRANSACTIONS (50% Class 0)
    # -------------------------------------------------------------
    normal_channels = ['MOBILE_APP', 'ONLINE_BANKING', 'BRANCH_COUNTER', 'ATM', 'SWIFT']
    txn_types_ind = ['PEER_TO_PEER', 'CASH_WITHDRAWAL', 'ACH_TRANSFER', 'POINT_OF_SALE']
    txn_types_corp = ['WIRE_TRANSFER', 'ACH_TRANSFER', 'CORPORATE_PAYROLL']
    
    for idx in range(num_normal):
        # Round-robin customer selection ensures completely uniform, unbiased customer representation
        cid = cust_ids[idx % len(cust_ids)]
        c_prof = customers_dict[cid]
        
        # Normal amount based on expected monthly turnover
        if c_prof['customer_type'] == 'INDIVIDUAL':
            amt = round(float(np.random.lognormal(mean=5.5, sigma=0.8)), 2) # ~$50 - $1,500
            amt = max(10.0, min(amt, 8000.0))
            txn_type = txn_types_ind[idx % len(txn_types_ind)]
            channel = normal_channels[idx % len(normal_channels)]
        else:
            amt = round(float(np.random.lognormal(mean=8.5, sigma=1.0)), 2) # ~$2,000 - $35,000
            amt = max(100.0, min(amt, 150000.0))
            txn_type = txn_types_corp[idx % len(txn_types_corp)]
            channel = normal_channels[(idx + 2) % len(normal_channels)]
            
        txn_date = start_date + timedelta(
            days=random.randint(0, 89),
            hours=random.randint(8, 20),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        orig_country = c_prof['primary_country']
        dest_country = NORMAL_JURISDICTIONS[idx % len(NORMAL_JURISDICTIONS)]
        beneficiary_name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}" if c_prof['customer_type'] == 'INDIVIDUAL' else f"{random.choice(BUSINESS_PREFIXES)} Services"
        
        records.append({
            'transaction_id': f"TXN-{txn_counter:06d}",
            'timestamp': txn_date.strftime('%Y-%m-%d %H:%M:%S'),
            'customer_id': cid,
            'originator_name': c_prof['customer_name'],
            'originator_account': c_prof['account_number'],
            'originator_country': orig_country,
            'beneficiary_name': beneficiary_name,
            'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
            'beneficiary_country': dest_country,
            'transaction_type': txn_type,
            'amount': amt,
            'currency': 'USD',
            'channel': channel,
            'device_ip': generate_random_ip(),
            'is_sar_reported': 0,
            'aml_typology': 'NORMAL',
            'red_flag_notes': 'Routine recurring commercial/retail banking activity.'
        })
        txn_counter += 1

    # -------------------------------------------------------------
    # 2. EXACT-QUOTA UNBIASED SUSPICIOUS AML TYPOLOGIES (50% Class 1, 20% each typology)
    # -------------------------------------------------------------
    suspicious_types = ['STRUCTURING', 'RAPID_FLOW_THROUGH', 'MULE_FAN_IN', 'ROUND_TRIPPING_SHELL', 'DORMANT_SPIKE']
    base_quota = num_suspicious // len(suspicious_types)
    remainder = num_suspicious % len(suspicious_types)
    typology_quotas = {t: base_quota + (1 if i < remainder else 0) for i, t in enumerate(suspicious_types)}
    
    # Unbiased candidate selection across ALL customer profiles (LOW, MEDIUM, HIGH, INDIVIDUAL, CORPORATE)
    all_candidate_cids = list(cust_ids)
    random.shuffle(all_candidate_cids)
    candidate_idx = 0

    for typology in suspicious_types:
        target_quota = typology_quotas[typology]
        generated_for_typology = 0
        
        while generated_for_typology < target_quota:
            target_cid = all_candidate_cids[candidate_idx % len(all_candidate_cids)]
            candidate_idx += 1
            c_prof = customers_dict[target_cid]
            base_time = start_date + timedelta(days=random.randint(15, 80), hours=random.randint(9, 16))
            
            if typology == 'STRUCTURING':
                # Sub-threshold cash deposits or transfers: $9,100 to $9,950
                num_structs = min(random.randint(3, 5), target_quota - generated_for_typology)
                for s_idx in range(num_structs):
                    s_amt = round(random.uniform(9100.0, 9920.0), 2)
                    s_time = base_time + timedelta(days=s_idx, hours=random.randint(1, 4), minutes=random.randint(0, 50))
                    records.append({
                        'transaction_id': f"TXN-{txn_counter:06d}",
                        'timestamp': s_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'customer_id': target_cid,
                        'originator_name': c_prof['customer_name'],
                        'originator_account': c_prof['account_number'],
                        'originator_country': c_prof['primary_country'],
                        'beneficiary_name': f"{random.choice(BUSINESS_PREFIXES)} Vault Ltd",
                        'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                        'beneficiary_country': random.choice(['US', 'GB', 'AE']),
                        'transaction_type': random.choice(['CASH_DEPOSIT', 'WIRE_TRANSFER']),
                        'amount': s_amt,
                        'currency': 'USD',
                        'channel': random.choice(['BRANCH_COUNTER', 'ATM']),
                        'device_ip': generate_random_ip(),
                        'is_sar_reported': 1,
                        'aml_typology': 'STRUCTURING',
                        'red_flag_notes': f'Structuring alert: Deposit of ${s_amt:,.2f} intentionally kept below $10,000 CTR statutory reporting threshold.'
                    })
                    txn_counter += 1
                    generated_for_typology += 1
                    
            elif typology == 'RAPID_FLOW_THROUGH':
                inbound_amt = round(random.uniform(45000.0, 220000.0), 2)
                outbound_amt = round(inbound_amt * random.uniform(0.92, 0.98), 2)
                
                # Inbound
                records.append({
                    'transaction_id': f"TXN-{txn_counter:06d}",
                    'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'customer_id': target_cid,
                    'originator_name': f"External Partner {random.choice(BUSINESS_SUFFIXES)}",
                    'originator_account': f"ACCT-{random.randint(10000000, 99999999)}",
                    'originator_country': random.choice(HIGH_RISK_JURISDICTIONS),
                    'beneficiary_name': c_prof['customer_name'],
                    'beneficiary_account': c_prof['account_number'],
                    'beneficiary_country': c_prof['primary_country'],
                    'transaction_type': 'WIRE_TRANSFER',
                    'amount': inbound_amt,
                    'currency': 'USD',
                    'channel': 'SWIFT',
                    'device_ip': generate_random_ip(),
                    'is_sar_reported': 1,
                    'aml_typology': 'RAPID_FLOW_THROUGH',
                    'red_flag_notes': f'Rapid Flow-Through Stage 1: Received high-value international wire of ${inbound_amt:,.2f} from high-risk jurisdiction.'
                })
                txn_counter += 1
                generated_for_typology += 1
                
                if generated_for_typology < target_quota:
                    outbound_time = base_time + timedelta(hours=random.randint(1, 2), minutes=random.randint(10, 45))
                    records.append({
                        'transaction_id': f"TXN-{txn_counter:06d}",
                        'timestamp': outbound_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'customer_id': target_cid,
                        'originator_name': c_prof['customer_name'],
                        'originator_account': c_prof['account_number'],
                        'originator_country': c_prof['primary_country'],
                        'beneficiary_name': f"Offshore Wealth Entity {random.choice(BUSINESS_SUFFIXES)}",
                        'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                        'beneficiary_country': random.choice(HIGH_RISK_JURISDICTIONS),
                        'transaction_type': 'WIRE_TRANSFER',
                        'amount': outbound_amt,
                        'currency': 'USD',
                        'channel': 'ONLINE_BANKING',
                        'device_ip': generate_random_ip(),
                        'is_sar_reported': 1,
                        'aml_typology': 'RAPID_FLOW_THROUGH',
                        'red_flag_notes': f'Rapid Flow-Through Stage 2: Immediate pass-through of ${outbound_amt:,.2f} (<2h dwell time) to offshore entity.'
                    })
                    txn_counter += 1
                    generated_for_typology += 1
                    
            elif typology == 'MULE_FAN_IN':
                fan_count = min(random.randint(3, 5), max(1, target_quota - generated_for_typology - 1))
                if fan_count < 1:
                    fan_count = 1
                total_collected = 0.0
                for f_idx in range(fan_count):
                    f_amt = round(random.uniform(800.0, 2400.0), 2)
                    total_collected += f_amt
                    f_time = base_time + timedelta(hours=f_idx * 3, minutes=random.randint(5, 45))
                    records.append({
                        'transaction_id': f"TXN-{txn_counter:06d}",
                        'timestamp': f_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'customer_id': target_cid,
                        'originator_name': f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                        'originator_account': f"ACCT-{random.randint(10000000, 99999999)}",
                        'originator_country': 'GB',
                        'beneficiary_name': c_prof['customer_name'],
                        'beneficiary_account': c_prof['account_number'],
                        'beneficiary_country': c_prof['primary_country'],
                        'transaction_type': 'PEER_TO_PEER',
                        'amount': f_amt,
                        'currency': 'USD',
                        'channel': 'MOBILE_APP',
                        'device_ip': generate_random_ip(),
                        'is_sar_reported': 1,
                        'aml_typology': 'MULE_FAN_IN',
                        'red_flag_notes': f'Mule Fan-In: Micro-aggregation deposit of ${f_amt:,.2f} from unconnected 3rd party.'
                    })
                    txn_counter += 1
                    generated_for_typology += 1
                    
                if generated_for_typology < target_quota:
                    sweep_time = base_time + timedelta(hours=fan_count * 3 + 1)
                    records.append({
                        'transaction_id': f"TXN-{txn_counter:06d}",
                        'timestamp': sweep_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'customer_id': target_cid,
                        'originator_name': c_prof['customer_name'],
                        'originator_account': c_prof['account_number'],
                        'originator_country': c_prof['primary_country'],
                        'beneficiary_name': f"Crypto Asset Gateway Inc",
                        'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                        'beneficiary_country': 'PA',
                        'transaction_type': 'CRYPTO_PURCHASE',
                        'amount': round(total_collected * 0.96, 2),
                        'currency': 'USD',
                        'channel': 'ONLINE_BANKING',
                        'device_ip': generate_random_ip(),
                        'is_sar_reported': 1,
                        'aml_typology': 'MULE_FAN_IN',
                        'red_flag_notes': f'Mule Account Cash-Out: Aggregate sweep of ${round(total_collected * 0.96, 2):,.2f} to crypto exchange.'
                    })
                    txn_counter += 1
                    generated_for_typology += 1

            elif typology == 'ROUND_TRIPPING_SHELL':
                round_amt = float(random.choice([50000, 75000, 100000, 150000, 200000]))
                records.append({
                    'transaction_id': f"TXN-{txn_counter:06d}",
                    'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'customer_id': target_cid,
                    'originator_name': c_prof['customer_name'],
                    'originator_account': c_prof['account_number'],
                    'originator_country': c_prof['primary_country'],
                    'beneficiary_name': f"Panama Horizon Holdings Corp",
                    'beneficiary_account': f"ACCT-{random.randint(10000000, 99999999)}",
                    'beneficiary_country': 'PA',
                    'transaction_type': 'WIRE_TRANSFER',
                    'amount': round_amt,
                    'currency': 'USD',
                    'channel': 'SWIFT',
                    'device_ip': generate_random_ip(),
                    'is_sar_reported': 1,
                    'aml_typology': 'ROUND_TRIPPING_SHELL',
                    'red_flag_notes': f'Round-Tripping: Exact round sum wire of ${round_amt:,.2f} to offshore shell corporation in secrecy jurisdiction.'
                })
                txn_counter += 1
                generated_for_typology += 1

            elif typology == 'DORMANT_SPIKE':
                spike_amt = round(max(35000.0, c_prof['expected_monthly_turnover'] * random.uniform(15.0, 40.0)), 2)
                records.append({
                    'transaction_id': f"TXN-{txn_counter:06d}",
                    'timestamp': base_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'customer_id': target_cid,
                    'originator_name': f"International Trading Alliance",
                    'originator_account': f"ACCT-{random.randint(10000000, 99999999)}",
                    'originator_country': random.choice(HIGH_RISK_JURISDICTIONS),
                    'beneficiary_name': c_prof['customer_name'],
                    'beneficiary_account': c_prof['account_number'],
                    'beneficiary_country': c_prof['primary_country'],
                    'transaction_type': 'WIRE_TRANSFER',
                    'amount': spike_amt,
                    'currency': 'USD',
                    'channel': 'SWIFT',
                    'device_ip': generate_random_ip(),
                    'is_sar_reported': 1,
                    'aml_typology': 'DORMANT_SPIKE',
                    'red_flag_notes': f'Velocity Spike: Sudden incoming transaction of ${spike_amt:,.2f} representing {(spike_amt/c_prof["expected_monthly_turnover"]):.1f}x monthly expected baseline.'
                })
                txn_counter += 1
                generated_for_typology += 1

    df_transactions = pd.DataFrame(records)
    
    if add_noise:
        df_transactions = inject_synthetic_noise(df_transactions, noise_ratio=noise_ratio, seed=seed)
        
    # Sort chronologically
    df_transactions['timestamp_dt'] = pd.to_datetime(df_transactions['timestamp'], errors='coerce')
    df_transactions = df_transactions.sort_values('timestamp_dt').reset_index(drop=True)
    df_transactions = df_transactions.drop(columns=['timestamp_dt'])
    
    return df_transactions

def generate_and_save_all_datasets(data_dir=None, train_size=6000, test_size=1500, seed=42):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"[+] Generating customer profiles...")
    customers_df = generate_customer_profiles(num_customers=300, seed=seed)
    customers_path = os.path.join(data_dir, 'customer_profiles.csv')
    customers_df.to_csv(customers_path, index=False)
    print(f"    Saved {len(customers_df)} customer profiles -> {customers_path}")
    
    print(f"[+] Generating synthetic training transactions ({train_size} records, balanced 50/50)...")
    train_df = generate_synthetic_transactions(customers_df, total_records=train_size, suspicious_ratio=0.50, seed=seed)
    train_path = os.path.join(data_dir, 'train_aml_transactions.csv')
    train_df.to_csv(train_path, index=False)
    print(f"    Saved {len(train_df)} training records -> {train_path}")
    
    print(f"[+] Generating synthetic testing transactions ({test_size} records, balanced 50/50)...")
    test_df = generate_synthetic_transactions(customers_df, total_records=test_size, suspicious_ratio=0.50, seed=seed+99)
    test_path = os.path.join(data_dir, 'test_aml_transactions.csv')
    test_df.to_csv(test_path, index=False)
    print(f"    Saved {len(test_df)} testing records -> {test_path}")
    
    return customers_df, train_df, test_df

def generate_incremental_transactions(customer_df, existing_df=None, batch_size=1000, suspicious_ratio=0.50, seed=None, add_noise=True, noise_ratio=0.08):
    """
    Generates a fresh batch of synthetic transactions continuing from the max
    transaction counter and latest timestamp of existing_df.
    """
    if seed is None:
        seed = int(datetime.now().timestamp() * 1000) % 1000000
        
    start_counter = 1
    start_date = datetime.now() - timedelta(days=30)
    
    if existing_df is not None and len(existing_df) > 0:
        max_num = 0
        for tid in existing_df['transaction_id'].dropna().astype(str):
            if tid.startswith('TXN-'):
                try:
                    num = int(tid.replace('TXN-', ''))
                    if num > max_num:
                        max_num = num
                except Exception:
                    pass
        start_counter = max_num + 1
        
        if 'timestamp' in existing_df.columns:
            try:
                latest_ts = pd.to_datetime(existing_df['timestamp']).max()
                start_date = latest_ts + timedelta(minutes=random.randint(2, 15))
            except Exception:
                start_date = datetime.now() - timedelta(days=5)
                
    return generate_synthetic_transactions(
        customer_df=customer_df,
        total_records=batch_size,
        suspicious_ratio=suspicious_ratio,
        seed=seed,
        start_counter=start_counter,
        start_date=start_date,
        add_noise=add_noise,
        noise_ratio=noise_ratio
    )

if __name__ == '__main__':
    generate_and_save_all_datasets()
