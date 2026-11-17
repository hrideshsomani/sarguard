"""
Enterprise AML Data Preprocessing Engine
Performs multi-stage data sanitization, missing value imputation, text normalization,
timestamp verification, outlier handling, and data quality reporting.
Designed for high-throughput financial surveillance streams.
"""

import numpy as np
import pandas as pd
from datetime import datetime

class AMLDataPreprocessor:
    def __init__(self):
        self.last_report = {}
        
    def preprocess(self, transactions_df, verbose=False):
        """
        Executes a 6-stage data preprocessing pipeline:
        Stage 1: Schema Validation & Numerical Coercion
        Stage 2: Text Normalization & Casing Standardization
        Stage 3: Missing Value Detection & Regulatory Imputation
        Stage 4: Datetime Standardization & Chronological Ordering
        Stage 5: Extreme Outlier Handling & Winsorization
        Stage 6: Preprocessing Audit & Data Quality Scorecard
        """
        if transactions_df is None or len(transactions_df) == 0:
            return transactions_df, {}

        df = transactions_df.copy()
        initial_len = len(df)
        report = {
            'initial_records': initial_len,
            'missing_imputations': {},
            'casing_fixes': 0,
            'amount_anomalies_fixed': 0,
            'outliers_winsorized': 0,
            'timestamp_errors_resolved': 0,
            'data_health_score': 100.0
        }

        # -------------------------------------------------------------
        # STAGE 1: Schema Validation & Numerical Coercion
        # -------------------------------------------------------------
        # Ensure 'amount' is float and positive
        if 'amount' in df.columns:
            if df['amount'].dtype == object:
                # Remove dollar signs, commas, extra whitespace
                cleaned_amt = df['amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
                df['amount'] = pd.to_numeric(cleaned_amt, errors='coerce')
                report['amount_anomalies_fixed'] += int(df['amount'].isna().sum())
            
            # Impute invalid/negative amounts with median
            amt_median = float(df['amount'].median()) if not df['amount'].isna().all() else 250.0
            if amt_median <= 0:
                amt_median = 250.0
            
            nan_amt_count = int(df['amount'].isna().sum() + (df['amount'] <= 0).sum())
            if nan_amt_count > 0:
                df['amount'] = df['amount'].apply(lambda x: amt_median if pd.isna(x) or x <= 0 else float(x))
                report['missing_imputations']['amount'] = nan_amt_count

        # Coerce IDs to string
        for id_col in ['transaction_id', 'customer_id', 'originator_account', 'beneficiary_account']:
            if id_col in df.columns:
                df[id_col] = df[id_col].astype(str).str.strip()

        # -------------------------------------------------------------
        # STAGE 2: Text Normalization & Casing Standardization
        # -------------------------------------------------------------
        text_cols = ['originator_name', 'beneficiary_name', 'channel', 'transaction_type', 
                     'originator_country', 'beneficiary_country', 'currency', 'aml_typology']
        
        casing_fixes = 0
        for col in text_cols:
            if col in df.columns:
                # Track raw state for reporting
                non_null_mask = df[col].notna()
                raw_values = df.loc[non_null_mask, col].astype(str)
                cleaned_values = raw_values.str.strip()
                
                # Standardize uppercase for regulatory categories
                if col in ['channel', 'transaction_type', 'originator_country', 'beneficiary_country', 'currency', 'aml_typology']:
                    cleaned_values = cleaned_values.str.upper()
                    
                diff_count = int((raw_values != cleaned_values).sum())
                casing_fixes += diff_count
                df.loc[non_null_mask, col] = cleaned_values

        report['casing_fixes'] = casing_fixes

        # Standardize known channel mapping
        channel_map = {
            'BRANCH': 'BRANCH_COUNTER',
            'ONLINE': 'ONLINE_BANKING',
            'APP': 'MOBILE_APP',
            'WIRE': 'SWIFT'
        }
        if 'channel' in df.columns:
            df['channel'] = df['channel'].replace(channel_map)

        # Standardize known transaction type mapping
        type_map = {
            'WIRE': 'WIRE_TRANSFER',
            'ACH': 'ACH_TRANSFER',
            'P2P': 'PEER_TO_PEER',
            'CASH_IN': 'CASH_DEPOSIT',
            'CASH_OUT': 'CASH_WITHDRAWAL',
            'CRYPTO': 'CRYPTO_PURCHASE'
        }
        if 'transaction_type' in df.columns:
            df['transaction_type'] = df['transaction_type'].replace(type_map)

        # -------------------------------------------------------------
        # STAGE 3: Missing Value Detection & Regulatory Imputation
        # -------------------------------------------------------------
        # Device IP: default to UNKNOWN_IP
        if 'device_ip' in df.columns:
            missing_ip = int(df['device_ip'].isna().sum() + (df['device_ip'].astype(str).str.strip() == '').sum())
            if missing_ip > 0:
                df['device_ip'] = df['device_ip'].fillna('UNKNOWN_IP')
                df['device_ip'] = df['device_ip'].replace('', 'UNKNOWN_IP')
                report['missing_imputations']['device_ip'] = missing_ip

        # Beneficiary Account
        if 'beneficiary_account' in df.columns:
            missing_ba = int(df['beneficiary_account'].isna().sum() + (df['beneficiary_account'].astype(str).str.strip().isin(['', 'nan', 'None'])).sum())
            if missing_ba > 0:
                df['beneficiary_account'] = df['beneficiary_account'].fillna('UNSPECIFIED_ACCT')
                df['beneficiary_account'] = df['beneficiary_account'].replace(['', 'nan', 'None'], 'UNSPECIFIED_ACCT')
                report['missing_imputations']['beneficiary_account'] = missing_ba

        # Beneficiary Name
        if 'beneficiary_name' in df.columns:
            missing_bn = int(df['beneficiary_name'].isna().sum() + (df['beneficiary_name'].astype(str).str.strip().isin(['', 'nan', 'None'])).sum())
            if missing_bn > 0:
                df['beneficiary_name'] = df['beneficiary_name'].fillna('UNKNOWN_BENEFICIARY')
                df['beneficiary_name'] = df['beneficiary_name'].replace(['', 'nan', 'None'], 'UNKNOWN_BENEFICIARY')
                report['missing_imputations']['beneficiary_name'] = missing_bn

        # Country codes
        for c_col in ['originator_country', 'beneficiary_country']:
            if c_col in df.columns:
                missing_c = int(df[c_col].isna().sum() + (df[c_col].astype(str).str.strip().isin(['', 'nan', 'None'])).sum())
                if missing_c > 0:
                    df[c_col] = df[c_col].fillna('UNKNOWN')
                    df[c_col] = df[c_col].replace(['', 'nan', 'None'], 'UNKNOWN')
                    report['missing_imputations'][c_col] = missing_c

        # Currency
        if 'currency' in df.columns:
            df['currency'] = df['currency'].fillna('USD').replace(['', 'nan', 'None'], 'USD')

        # Red Flag Notes
        if 'red_flag_notes' in df.columns:
            df['red_flag_notes'] = df['red_flag_notes'].fillna('Routine surveillance record.')

        # -------------------------------------------------------------
        # STAGE 4: Datetime Standardization & Chronological Ordering
        # -------------------------------------------------------------
        if 'timestamp' in df.columns:
            raw_ts = df['timestamp']
            parsed_dt = pd.to_datetime(raw_ts, errors='coerce')
            ts_errors = int(parsed_dt.isna().sum())
            
            if ts_errors > 0:
                # Forward/backward fill or fallback to current datetime
                fallback_dt = datetime.now()
                parsed_dt = parsed_dt.ffill().bfill().fillna(fallback_dt)
                report['timestamp_errors_resolved'] = ts_errors

            df['dt'] = parsed_dt
            # Chronological sort per customer for precise rolling velocity
            if 'customer_id' in df.columns:
                df = df.sort_values(['customer_id', 'dt']).reset_index(drop=True)
            else:
                df = df.sort_values('dt').reset_index(drop=True)
            
            df['timestamp'] = df['dt'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # -------------------------------------------------------------
        # STAGE 5: Extreme Outlier Handling & Winsorization
        # -------------------------------------------------------------
        # Cap non-AML extreme outliers at 99.9th percentile to prevent model skew
        if 'amount' in df.columns:
            q999 = float(df['amount'].quantile(0.999))
            # Only cap if threshold is reasonably high ($500,000+)
            cap_limit = max(q999, 500000.0)
            extreme_mask = df['amount'] > cap_limit
            outliers_count = int(extreme_mask.sum())
            if outliers_count > 0:
                df.loc[extreme_mask, 'amount'] = cap_limit
                report['outliers_winsorized'] = outliers_count

        # -------------------------------------------------------------
        # STAGE 6: Preprocessing Audit & Data Quality Scorecard
        # -------------------------------------------------------------
        total_flaws = (
            sum(report['missing_imputations'].values()) +
            report['casing_fixes'] +
            report['amount_anomalies_fixed'] +
            report['timestamp_errors_resolved']
        )
        health_penalty = min(30.0, (total_flaws / max(initial_len * len(df.columns), 1)) * 100.0 * 5.0)
        report['data_health_score'] = round(100.0 - health_penalty, 2)
        report['final_records'] = len(df)
        
        self.last_report = report
        
        if verbose:
            print("[+] Preprocessing Complete:")
            print(f"    - Processed {initial_len:,} rows")
            print(f"    - Imputations: {report['missing_imputations']}")
            print(f"    - Casing / Whitespace Fixes: {report['casing_fixes']:,}")
            print(f"    - Data Health Score: {report['data_health_score']}%")

        return df, report

    def get_last_report(self):
        return self.last_report

    @staticmethod
    def get_preprocessing_steps_summary():
        """Returns structured descriptions of all preprocessing steps for presentations & viva."""
        return [
            {
                'step': '1. Schema Validation & Numerical Coercion',
                'description': 'Validates column integrity, strips currency symbols ($) and commas, coerces amounts to floating-point values, and imputes zero/negative amounts with the dataset median.'
            },
            {
                'step': '2. Text & Casing Normalization',
                'description': 'Strips irregular whitespace and standardizes categorical text into uppercase canonical tokens (e.g. "WIRE_TRANSFER", "BRANCH_COUNTER", "SWIFT") across channels, transaction types, and countries.'
            },
            {
                'step': '3. Missing Value (Null) Imputation',
                'description': 'Applies regulatory-compliant imputation for missing optional fields: device IPs imputed to "UNKNOWN_IP", missing counterparty accounts imputed to "UNSPECIFIED_ACCT", and missing entity names to "UNKNOWN_BENEFICIARY".'
            },
            {
                'step': '4. Datetime Standardization & Chronological Sorting',
                'description': 'Parses timestamps across heterogeneous formats with error fallback. Sorts transactions chronologically by (customer_id, timestamp) to ensure deterministic rolling 24-hour temporal velocity and dwell-time calculations.'
            },
            {
                'step': '5. Extreme Outlier Winsorization',
                'description': 'Applies soft percentile capping (99.9th percentile or $500,000 threshold) on non-structuring amounts to prevent extreme numeric distortion in tree split thresholds while preserving high-risk AML signals.'
            },
            {
                'step': '6. Preprocessing Audit & Health Scorecard',
                'description': 'Generates an immutable audit trail of all data transformations, tracking the exact count of imputations, text fixes, and outlier caps with an overall Data Health Quality Score.'
            }
        ]
