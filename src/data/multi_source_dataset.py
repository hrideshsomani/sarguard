"""
Multi-Source Financial AML Dataset Adapter & Streamer
Integrates 9 Heterogeneous Financial Crime Domains:
1. IBM Transactions for Anti-Money Laundering (AMLWorld / Kaggle: ealtman2019) - Cross-Bank Wires
2. PaySim Mobile Money Financial Fraud & AML Dataset (Kaggle / HuggingFace) - P2P & Mobile Wallets
3. Cryptocurrency & Blockchain AML Stream (Elliptic/Mixer Style) - Tumblers, Exchanges & Unhosted Wallets
4. Retail Card Payment & Merchant Fraud Stream - CNP, Micro-Structuring & Rapid Velocity Testing
5. Trade-Based Money Laundering (TBML) & Sanctions Stream - Phantom Shipments, Over-Invoicing & High-Risk OFAC
6. Securities, Investment Fraud & Ponzi Scheme AML Stream - Broker-Dealer Wash Trades & Capital Redistribution
7. Real Estate Title, Escrow & FinCEN GTO AML Stream - Luxury Cash Purchases & Shell Escrow Layering
8. Digital Fintech P2P Lending & Micro-Finance Structuring - Multi-Platform Loan Stacking & Rapid Payouts
9. Enterprise Multi-Typology High-Throughput Synthetic AML Engine - FinCEN/FATF Structuring, Layering & Mules

Unifies diverse schemas into the canonical SAR detection schema for enterprise surveillance,
feature engineering, continuous model training, and FinCEN/FCA compliance checking.
"""

import os
import sys
import random
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Safe cache configuration on secondary drive if available
if os.path.exists("D:\\"):
    os.environ.setdefault('HF_HOME', r'D:\hf_cache')
    DATA_CACHE_DIR = r'D:\aml_data_cache'
else:
    DATA_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache')

os.makedirs(DATA_CACHE_DIR, exist_ok=True)

HIGH_RISK_JURISDICTIONS = ['KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG']
NORMAL_JURISDICTIONS = ['GB', 'US', 'DE', 'FR', 'NL', 'CA', 'AU', 'SG', 'CH', 'JP']

class MultiSourceAMLDataset:
    def __init__(self, data_cache_dir=None, local_kaggle_dir=None):
        self.cache_dir = data_cache_dir or DATA_CACHE_DIR
        self.local_kaggle_dir = local_kaggle_dir
        self.ibm_stream_iter = None
        self.paysim_stream_iter = None
        self.customer_cache = {}
        self._ibm_cached_df = None
        self._paysim_cached_df = None
        
    def _init_ibm_stream(self):
        """Initializes streaming iterator for IBM AML dataset."""
        try:
            from datasets import load_dataset
            print("  [+] Connecting to IBM AML dataset stream (eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans)...")
            ds = load_dataset('eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans', split='train', streaming=True)
            self.ibm_stream_iter = iter(ds)
            return True
        except Exception as e:
            print(f"  [!] Note: Online IBM streaming initialization notice: {e}")
            return False

    def _init_paysim_stream(self):
        """Initializes streaming iterator for PaySim dataset."""
        try:
            from datasets import load_dataset
            print("  [+] Connecting to PaySim AML dataset stream (theman10/paysim)...")
            ds = load_dataset('theman10/paysim', split='train', streaming=True)
            self.paysim_stream_iter = iter(ds)
            return True
        except Exception as e:
            print(f"  [!] Note: Online PaySim streaming initialization notice: {e}")
            return False

    # -------------------------------------------------------------------------
    # 1. IBM TRANSACTIONS FOR AML (INTERBANK WIRES)
    # -------------------------------------------------------------------------
    def fetch_ibm_batch(self, batch_size=50000):
        cache_path = os.path.join(self.cache_dir, 'ibm_kaggle_cache.parquet')
        if not os.path.exists(cache_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache', 'ibm_kaggle_cache.parquet')
            if os.path.exists(alt_path):
                cache_path = alt_path
        if os.path.exists(cache_path):
            try:
                if self._ibm_cached_df is None:
                    self._ibm_cached_df = pd.read_parquet(cache_path)
                replace = (batch_size > len(self._ibm_cached_df))
                return self._ibm_cached_df.sample(n=batch_size, replace=replace).reset_index(drop=True)
            except Exception:
                pass

        if self.local_kaggle_dir and os.path.exists(self.local_kaggle_dir):
            for fname in os.listdir(self.local_kaggle_dir):
                if ('HI-Small' in fname or 'LI-Small' in fname or 'HI-Medium' in fname or 'Trans' in fname) and fname.endswith('.csv'):
                    local_path = os.path.join(self.local_kaggle_dir, fname)
                    try:
                        df_raw = pd.read_csv(local_path, nrows=batch_size)
                        return self._transform_ibm_df(df_raw)
                    except Exception:
                        pass

        if self.ibm_stream_iter is None:
            if not self._init_ibm_stream():
                return pd.DataFrame()

        records = []
        try:
            for _ in range(batch_size):
                try:
                    raw_row = next(self.ibm_stream_iter)
                    records.append(raw_row)
                except StopIteration:
                    self._init_ibm_stream()
                    break
        except Exception as e:
            print(f"  [!] IBM streaming notice: {e}")

        if not records:
            return pd.DataFrame()

        df_raw = pd.DataFrame(records)
        return self._transform_ibm_df(df_raw)

    def _transform_ibm_df(self, df_raw):
        if df_raw.empty:
            return pd.DataFrame()

        rows = []
        now = datetime.now()
        for idx, r in df_raw.iterrows():
            amt = float(r.get('Amount Paid', r.get('Amount Received', 1000.0)))
            is_laundering = int(r.get('Is Laundering', 0))
            from_bank = str(r.get('From Bank', '10'))
            to_bank = str(r.get('To Bank', '10'))
            from_acc = str(r.get('Account', f'ACC_{idx}'))
            to_acc = str(r.get('Account.1', f'BEN_{idx}'))
            fmt = str(r.get('Payment Format', 'Wire')).upper()
            curr = str(r.get('Payment Currency', 'US Dollar'))
            
            if 'WIRE' in fmt:
                channel = 'SWIFT'
                t_type = 'WIRE_TRANSFER'
            elif 'CASH' in fmt:
                channel = 'BRANCH'
                t_type = 'CASH_DEPOSIT' if (idx % 2 == 0) else 'CASH_WITHDRAWAL'
            elif 'CHEQUE' in fmt:
                channel = 'BRANCH'
                t_type = 'CHEQUE_DEPOSIT'
            elif 'REINVEST' in fmt or 'ACH' in fmt:
                channel = 'ONLINE'
                t_type = 'INTERNAL_TRANSFER'
            else:
                channel = 'ONLINE'
                t_type = 'WIRE_TRANSFER'

            ts_str = str(r.get('Timestamp', ''))
            try:
                dt_obj = pd.to_datetime(ts_str)
                ts_iso = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts_iso = (now - timedelta(minutes=int(idx) * 2)).strftime('%Y-%m-%d %H:%M:%S')

            is_cross_border = (from_bank != to_bank)
            if is_cross_border and is_laundering == 1:
                orig_country = 'US'
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
            else:
                orig_country = 'US'
                dest_country = random.choice(NORMAL_JURISDICTIONS) if is_cross_border else 'US'

            if is_laundering == 1:
                if 9000.0 <= amt <= 9999.0:
                    typology = 'STRUCTURING'
                elif amt >= 50000.0 and (amt % 10000.0 == 0):
                    typology = 'ROUND_AMOUNT'
                elif dest_country in HIGH_RISK_JURISDICTIONS:
                    typology = 'HIGH_RISK_JURISDICTION'
                else:
                    typology = 'RAPID_LAYERING'
            else:
                typology = 'BENIGN'

            rows.append({
                'transaction_id': f"TXN-IBM-{idx:08d}-{abs(hash(from_acc)) % 1000000:06d}",
                'timestamp': ts_iso,
                'customer_id': f"IBM_B{from_bank}_{from_acc}",
                'counterparty_id': f"IBM_B{to_bank}_{to_acc}",
                'amount': round(max(1.0, amt), 2),
                'currency': 'USD' if 'Dollar' in curr else 'EUR',
                'transaction_type': t_type,
                'channel': channel,
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': is_laundering,
                'aml_typology': typology,
                'dataset_source': 'IBM_AML_WORLD',
                'customer_risk_rating': 'HIGH' if is_laundering == 1 else ('MEDIUM' if is_cross_border else 'LOW')
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 2. PAYSIM MOBILE MONEY & P2P FINTECH FRAUD/AML
    # -------------------------------------------------------------------------
    def fetch_paysim_batch(self, batch_size=50000):
        cache_path = os.path.join(self.cache_dir, 'paysim_kaggle_cache.parquet')
        if not os.path.exists(cache_path):
            alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'cache', 'paysim_kaggle_cache.parquet')
            if os.path.exists(alt_path):
                cache_path = alt_path
        if os.path.exists(cache_path):
            try:
                if self._paysim_cached_df is None:
                    self._paysim_cached_df = pd.read_parquet(cache_path)
                replace = (batch_size > len(self._paysim_cached_df))
                return self._paysim_cached_df.sample(n=batch_size, replace=replace).reset_index(drop=True)
            except Exception:
                pass

        if self.paysim_stream_iter is None:
            if not self._init_paysim_stream():
                return pd.DataFrame()

        records = []
        try:
            for _ in range(batch_size):
                try:
                    raw_row = next(self.paysim_stream_iter)
                    records.append(raw_row)
                except StopIteration:
                    self._init_paysim_stream()
                    break
        except Exception as e:
            print(f"  [!] PaySim streaming notice: {e}")

        if not records:
            return pd.DataFrame()

        df_raw = pd.DataFrame(records)
        return self._transform_paysim_df(df_raw)

    def _transform_paysim_df(self, df_raw):
        if df_raw.empty:
            return pd.DataFrame()

        rows = []
        base_time = datetime(2026, 1, 1, 0, 0, 0)
        for idx, r in df_raw.iterrows():
            amt = float(r.get('amount', 500.0))
            is_fraud = int(r.get('isFraud', 0) or r.get('isFlaggedFraud', 0))
            step = int(r.get('step', 1))
            raw_type = str(r.get('type', 'TRANSFER')).upper()
            orig = str(r.get('nameOrig', f'C{idx}'))
            dest = str(r.get('nameDest', f'M{idx}'))

            if raw_type == 'TRANSFER':
                t_type = 'WIRE_TRANSFER'
                channel = 'ONLINE'
            elif raw_type == 'CASH_OUT':
                t_type = 'CASH_WITHDRAWAL'
                channel = 'ATM' if amt < 2000 else 'BRANCH'
            elif raw_type == 'CASH_IN':
                t_type = 'CASH_DEPOSIT'
                channel = 'BRANCH'
            elif raw_type == 'PAYMENT':
                t_type = 'BILL_PAYMENT'
                channel = 'MOBILE'
            else:
                t_type = 'INTERNAL_TRANSFER'
                channel = 'ONLINE'

            txn_time = base_time + timedelta(hours=step, minutes=int(idx % 60), seconds=int((idx * 7) % 60))
            ts_iso = txn_time.strftime('%Y-%m-%d %H:%M:%S')

            if is_fraud == 1:
                orig_country = 'US'
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                typology = 'RAPID_LAYERING' if raw_type == 'TRANSFER' else 'MULE_FAN_IN'
            else:
                orig_country = 'US'
                dest_country = 'US' if (raw_type != 'TRANSFER' or idx % 3 != 0) else random.choice(NORMAL_JURISDICTIONS)
                typology = 'BENIGN'

            rows.append({
                'transaction_id': f"TXN-PSM-{step:04d}-{idx:06d}",
                'timestamp': ts_iso,
                'customer_id': f"PSM_{orig}",
                'counterparty_id': f"PSM_{dest}",
                'amount': round(max(1.0, amt), 2),
                'currency': 'USD',
                'transaction_type': t_type,
                'channel': channel,
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': is_fraud,
                'aml_typology': typology,
                'dataset_source': 'PAYSIM_FINANCIAL',
                'customer_risk_rating': 'HIGH' if is_fraud == 1 else 'LOW'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 3. CRYPTOCURRENCY & BLOCKCHAIN AML (ELLIPTIC / MIXER STYLE)
    # -------------------------------------------------------------------------
    def fetch_crypto_aml_batch(self, batch_size=20000):
        """
        Generates realistic cryptocurrency AML transactions based on Elliptic / Chainalysis patterns:
        - Tumbler / Mixer layering (Tornado-style hop splitting)
        - Peer-to-peer crypto OTC transfers
        - Unhosted private wallet cash-outs
        - Sanctioned crypto exchange interactions
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 4 == 0) # 25% illicit cryptocurrency activity
            wallet_orig = f"0x{hashlib.sha256(f'orig_{idx}'.encode()).hexdigest()[:16]}"
            wallet_dest = f"0x{hashlib.sha256(f'dest_{idx}'.encode()).hexdigest()[:16]}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(9100.0, 9950.0),      # Structuring into crypto
                    random.uniform(25000.0, 150000.0),   # Mixer lump-sum cashout
                    random.uniform(5000.0, 8000.0)       # Mule dispersal
                ]))
                typology = random.choice(['CRYPTO_MIXER_LAYERING', 'UNHOSTED_WALLET_SPLIT', 'SANCTIONS_DEX_ROUTING'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = random.choice(['RU', 'IR', 'KP', 'US', 'GB'])
            else:
                amt = round(float(np.random.lognormal(mean=6.5, sigma=1.0)), 2)
                amt = max(25.0, min(amt, 15000.0))
                typology = 'BENIGN_CRYPTO_PURCHASE'
                orig_country = 'US'
                dest_country = 'US'

            ts_iso = (now - timedelta(seconds=idx * 3)).strftime('%Y-%m-%d %H:%M:%S')
            
            rows.append({
                'transaction_id': f"TXN-CRYPTO-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': f"WALLET_{wallet_orig}",
                'counterparty_id': f"EXCHANGE_{wallet_dest}",
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'CRYPTO_PURCHASE' if not is_suspicious else 'CRYPTO_WITHDRAWAL',
                'channel': 'CRYPTO_BLOCKCHAIN',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'CRYPTO_BLOCKCHAIN_AML',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'MEDIUM'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 4. RETAIL CARD PAYMENT & MERCHANT FRAUD STREAM
    # -------------------------------------------------------------------------
    def fetch_card_fraud_batch(self, batch_size=20000):
        """
        Generates payment card fraud & merchant laundering streams:
        - Micro-deposit smurfing & velocity card testing
        - Shell merchant factoring & chargeback laundering
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 5 == 0) # 20% card fraud / AML
            card_token = f"CARD_****{random.randint(1000, 9999)}"
            merchant_id = f"MERCH_{random.randint(100, 999)}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(9200.0, 9900.0),    # Prepaid card structuring
                    random.uniform(500.0, 1500.0),     # Velocity card bursts
                    random.uniform(20000.0, 50000.0)   # Shell merchant cash back
                ]))
                typology = random.choice(['MICRO_CHARGE_TESTING', 'PREPAID_CARD_SMURFING', 'MERCHANT_FACTORING'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = 'US'
            else:
                amt = round(float(np.random.lognormal(mean=4.2, sigma=0.8)), 2)
                amt = max(5.0, min(amt, 4500.0))
                typology = 'BENIGN_RETAIL'
                orig_country = 'US'
                dest_country = 'US'

            ts_iso = (now - timedelta(seconds=idx * 2)).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'transaction_id': f"TXN-CARD-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': f"HOLDER_{card_token}",
                'counterparty_id': merchant_id,
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'DEBIT_CARD_PURCHASE',
                'channel': 'ECOMMERCE',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'CARD_MERCHANT_FRAUD',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'LOW'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 5. TRADE-BASED MONEY LAUNDERING (TBML) & SANCTIONS STREAM
    # -------------------------------------------------------------------------
    def fetch_tbml_trade_batch(self, batch_size=20000):
        """
        Generates Trade-Based Money Laundering (TBML) & Sanctions Corridors:
        - Over-invoicing / under-invoicing of commercial goods
        - Phantom shipping & double invoicing
        - Direct interaction with FATF grey/black listed jurisdictions
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 3 == 0) # 33% TBML schemes
            importer = f"CORP_IMP_{random.randint(100, 999)}"
            exporter = f"CORP_EXP_{random.randint(100, 999)}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(50000.0, 500000.0), # Trade over-invoicing
                    random.uniform(100000.0, 1000000.0)# Phantom shipment wire
                ]))
                typology = random.choice(['TRADE_OVER_INVOICING', 'PHANTOM_SHIPMENT_LAYERING', 'SANCTIONS_CORRIDOR_EVASION'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = random.choice(['US', 'GB', 'DE', 'CH'])
            else:
                amt = float(random.uniform(10000.0, 80000.0))
                typology = 'BENIGN_COMMERCIAL_TRADE'
                orig_country = random.choice(['US', 'GB', 'DE'])
                dest_country = random.choice(NORMAL_JURISDICTIONS)

            ts_iso = (now - timedelta(minutes=idx * 5)).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'transaction_id': f"TXN-TBML-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': importer,
                'counterparty_id': exporter,
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'WIRE_TRANSFER',
                'channel': 'SWIFT',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'TRADE_BASED_TBML',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'MEDIUM'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 6. SECURITIES, INVESTMENT FRAUD & PONZI SCHEME STREAM
    # -------------------------------------------------------------------------
    def fetch_securities_investment_batch(self, batch_size=20000):
        """
        Generates Securities AML & Investment Fraud Streams (SEC / FINRA typologies):
        - Rapid capital recycling & Ponzi scheme distributions
        - Broker-dealer layering & wash trading cashouts
        - Microcap / OTC pump-and-dump account drainage
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 4 == 0) # 25% investment fraud / layering
            broker_orig = f"BROKER_ACC_{random.randint(1000, 9999)}"
            counterparty = f"FUND_ENTITY_{random.randint(100, 999)}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(45000.0, 350000.0), # Ponzi liquidity redemption
                    random.uniform(9200.0, 9950.0),    # Structured brokerage deposits
                    random.uniform(120000.0, 750000.0) # Wash trade liquidation
                ]))
                typology = random.choice(['PONZI_CAPITAL_RECYCLING', 'SECURITIES_PUMP_AND_DUMP', 'BROKER_DEALER_LAYERING'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = random.choice(['US', 'GB', 'SG', 'CH'])
            else:
                amt = round(float(np.random.lognormal(mean=6.8, sigma=1.1)), 2)
                amt = max(50.0, min(amt, 25000.0))
                typology = 'BENIGN_PORTFOLIO_DCA'
                orig_country = 'US'
                dest_country = 'US'

            ts_iso = (now - timedelta(minutes=idx * 2)).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'transaction_id': f"TXN-SEC-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': broker_orig,
                'counterparty_id': counterparty,
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'INVESTMENT_WITHDRAWAL' if is_suspicious else 'INVESTMENT_DEPOSIT',
                'channel': 'BROKERAGE_API',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'SECURITIES_INVESTMENT_FRAUD',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'LOW'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 7. REAL ESTATE TITLE, ESCROW & FINCEN GTO STREAM
    # -------------------------------------------------------------------------
    def fetch_real_estate_batch(self, batch_size=20000):
        """
        Generates Real Estate Money Laundering Streams (FinCEN GTO Typologies):
        - All-cash residential & commercial shell LLC purchases
        - Title escrow routing & round-trip purchase cancellation refunds
        - Beneficial ownership concealment via nominee buyers
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 5 == 0) # 20% real estate illicit flows
            buyer = f"SHELL_LLC_{random.randint(100, 999)}"
            escrow = f"TITLE_ESCROW_{random.randint(10, 99)}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(350000.0, 2500000.0), # All-cash GTO shell property purchase
                    random.uniform(150000.0, 800000.0),  # Escrow layer refund diversion
                    random.uniform(9300.0, 9900.0)       # Cash structuring towards closing fee
                ]))
                typology = random.choice(['REAL_ESTATE_GTO_CASH_PURCHASE', 'ESCROW_LAYER_DIVERSION', 'SHELL_DEED_FLIPPING'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = random.choice(['US', 'VG', 'PA', 'AE'])
            else:
                amt = round(float(np.random.lognormal(mean=8.2, sigma=0.9)), 2)
                amt = max(1000.0, min(amt, 150000.0))
                typology = 'BENIGN_MORTGAGE_ESCROW'
                orig_country = 'US'
                dest_country = 'US'

            ts_iso = (now - timedelta(hours=idx * 1)).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'transaction_id': f"TXN-RE-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': buyer,
                'counterparty_id': escrow,
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'ESCROW_DISBURSEMENT',
                'channel': 'ESCROW_SETTLEMENT',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'REAL_ESTATE_ESCROW_AML',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'MEDIUM'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 8. DIGITAL FINTECH P2P LENDING & MICRO-FINANCE STRUCTURING
    # -------------------------------------------------------------------------
    def fetch_fintech_p2p_lending_batch(self, batch_size=20000):
        """
        Generates Digital Fintech P2P Lending & Micro-Finance Structuring Streams:
        - Multi-app loan stacking and synthetic identity cash extraction
        - Immediate loan disbursement sweeps to offshore virtual accounts
        - Straw borrower micro-loan smurfing networks
        """
        rows = []
        now = datetime.now()
        for idx in range(batch_size):
            is_suspicious = (idx % 4 == 0) # 25% fintech loan fraud
            borrower = f"P2P_USER_{random.randint(10000, 99999)}"
            lending_app = f"FINTECH_LENDER_{random.randint(1, 20)}"
            
            if is_suspicious:
                amt = float(random.choice([
                    random.uniform(9200.0, 9950.0),    # Structuring loan requests
                    random.uniform(15000.0, 45000.0),  # Loan stacking simultaneous burst
                    random.uniform(5000.0, 12000.0)    # Instant offshore sweep
                ]))
                typology = random.choice(['LOAN_STACKING_DISPERSAL', 'STRAW_BORROWER_SMURFING', 'INSTANT_OFFSHORE_SWEEP'])
                dest_country = random.choice(HIGH_RISK_JURISDICTIONS)
                orig_country = 'US'
            else:
                amt = round(float(np.random.lognormal(mean=5.5, sigma=0.7)), 2)
                amt = max(100.0, min(amt, 8000.0))
                typology = 'BENIGN_PEER_REPAYMENT'
                orig_country = 'US'
                dest_country = 'US'

            ts_iso = (now - timedelta(seconds=idx * 4)).strftime('%Y-%m-%d %H:%M:%S')

            rows.append({
                'transaction_id': f"TXN-LEND-{idx:07d}",
                'timestamp': ts_iso,
                'customer_id': borrower,
                'counterparty_id': lending_app,
                'amount': round(amt, 2),
                'currency': 'USD',
                'transaction_type': 'LOAN_DISBURSEMENT' if is_suspicious else 'P2P_TRANSFER',
                'channel': 'MOBILE_FINTECH',
                'originator_country': orig_country,
                'beneficiary_country': dest_country,
                'is_sar_reported': 1 if is_suspicious else 0,
                'aml_typology': typology,
                'dataset_source': 'FINTECH_P2P_LENDING',
                'customer_risk_rating': 'HIGH' if is_suspicious else 'LOW'
            })

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 9. UNIFIED HETEROGENEOUS MULTI-DATASET STREAM SYNTHESIZER
    # -------------------------------------------------------------------------
    def generate_unified_stream_batch(
        self,
        batch_size=100000,
        ibm_ratio=0.10,
        paysim_ratio=0.10,
        crypto_ratio=0.10,
        card_ratio=0.10,
        tbml_ratio=0.10,
        securities_ratio=0.10,
        real_estate_ratio=0.10,
        fintech_ratio=0.10,
        synthetic_gen_func=None,
        customer_df=None,
        max_online_per_source=2500
    ):
        """
        Synthesizes a combined heterogeneous multi-domain stream across 9 distinct crime domains:
        [1] IBM AML Transactions (Cross-Bank Wires)
        [2] PaySim Financial (Mobile Wallets & P2P)
        [3] Crypto & Blockchain AML (Mixers & Tumblers)
        [4] Card & Retail Fraud (Micro-Charges & Velocity)
        [5] Trade-Based Money Laundering (TBML & Sanctions)
        [6] Securities & Investment Fraud (Ponzi & Wash Trades)
        [7] Real Estate Title & Escrow AML (FinCEN GTO Shell Purchases)
        [8] Digital Fintech P2P Lending & Loan Stacking
        [9] Enterprise Synthetic Engine (FinCEN/FATF Core SAR Typologies)
        """
        batches = []
        
        # 1. IBM AML
        if self.local_kaggle_dir and os.path.exists(self.local_kaggle_dir):
            ibm_n = int(batch_size * ibm_ratio)
        else:
            ibm_n = min(int(batch_size * ibm_ratio), max_online_per_source)
            
        paysim_n = min(int(batch_size * paysim_ratio), max_online_per_source)
        crypto_n = int(batch_size * crypto_ratio)
        card_n = int(batch_size * card_ratio)
        tbml_n = int(batch_size * tbml_ratio)
        sec_n = int(batch_size * securities_ratio)
        re_n = int(batch_size * real_estate_ratio)
        fintech_n = int(batch_size * fintech_ratio)
        
        # 1. Fetch IBM AML
        if ibm_n > 0:
            df_ibm = self.fetch_ibm_batch(batch_size=ibm_n)
            if not df_ibm.empty:
                batches.append(df_ibm)

        # 2. Fetch PaySim
        if paysim_n > 0:
            df_psm = self.fetch_paysim_batch(batch_size=paysim_n)
            if not df_psm.empty:
                batches.append(df_psm)

        # 3. Fetch Crypto AML
        if crypto_n > 0:
            df_crypto = self.fetch_crypto_aml_batch(batch_size=crypto_n)
            if not df_crypto.empty:
                batches.append(df_crypto)

        # 4. Fetch Card Fraud
        if card_n > 0:
            df_card = self.fetch_card_fraud_batch(batch_size=card_n)
            if not df_card.empty:
                batches.append(df_card)

        # 5. Fetch TBML
        if tbml_n > 0:
            df_tbml = self.fetch_tbml_trade_batch(batch_size=tbml_n)
            if not df_tbml.empty:
                batches.append(df_tbml)

        # 6. Fetch Securities & Ponzi
        if sec_n > 0:
            df_sec = self.fetch_securities_investment_batch(batch_size=sec_n)
            if not df_sec.empty:
                batches.append(df_sec)

        # 7. Fetch Real Estate Escrow
        if re_n > 0:
            df_re = self.fetch_real_estate_batch(batch_size=re_n)
            if not df_re.empty:
                batches.append(df_re)

        # 8. Fetch Fintech P2P Lending
        if fintech_n > 0:
            df_fintech = self.fetch_fintech_p2p_lending_batch(batch_size=fintech_n)
            if not df_fintech.empty:
                batches.append(df_fintech)

        # 9. Core Enterprise Synthetic Engine
        current_len = sum(len(b) for b in batches)
        synthetic_n = max(1000, batch_size - current_len)
        if synthetic_gen_func is not None and customer_df is not None:
            df_synth = synthetic_gen_func(
                customer_df=customer_df,
                existing_df=None,
                batch_size=synthetic_n,
                suspicious_ratio=0.45,
                add_noise=True,
                noise_ratio=0.08
            )
            df_synth['dataset_source'] = 'ENTERPRISE_SYNTHETIC_AML'
            batches.append(df_synth)

        if not batches:
            return pd.DataFrame()

        combined = pd.concat(batches, ignore_index=True)
        return combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
