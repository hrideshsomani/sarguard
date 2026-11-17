"""
AML Feature Engineering & Extraction Engine
Extracts quantitative, behavioral, graph, and compliance features from raw transaction streams
and customer metadata.
"""

import numpy as np
import pandas as pd
from datetime import datetime

HIGH_RISK_JURISDICTIONS = {'KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG'}

try:
    from ..preprocessing.data_preprocessor import AMLDataPreprocessor
except Exception:
    from src.preprocessing.data_preprocessor import AMLDataPreprocessor

class AMLFeatureExtractor:
    def __init__(self, customer_df=None):
        self.customer_metadata = {}
        self.preprocessor = AMLDataPreprocessor()
        self.last_preprocessing_report = {}
        if customer_df is not None:
            self.set_customers(customer_df)
            
    def set_customers(self, customer_df):
        for _, row in customer_df.iterrows():
            cid = row['customer_id']
            self.customer_metadata[cid] = {
                'expected_turnover': float(row.get('expected_monthly_turnover', 5000.0)),
                'risk_rating': str(row.get('risk_rating', 'LOW')),
                'customer_type': str(row.get('customer_type', 'INDIVIDUAL')),
                'primary_country': str(row.get('primary_country', 'GB')),
                'kyc_status': str(row.get('kyc_status', 'VERIFIED'))
            }

    def get_preprocessing_report(self):
        """Returns report of data quality and sanitizations performed during last extraction."""
        return self.last_preprocessing_report
            
    def extract_features(self, transactions_df):
        """
        Extracts multi-dimensional behavioral, regulatory, and anomaly features
        for ML models and rule verification. Automatically applies data preprocessing
        and noise sanitization first.
        """
        # Stage 0: Data Sanitization, Imputation, Text Normalization & Outlier Capping
        cleaned_df, report = self.preprocessor.preprocess(transactions_df)
        self.last_preprocessing_report = report
        df = cleaned_df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['dt'] = df['timestamp']
            
        df = df.sort_values('dt').reset_index(drop=True)
        
        # 1. Customer metadata joins
        risk_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        turnover_vals = []
        cust_risk_vals = []
        is_corp_vals = []
        
        for cid in df['customer_id']:
            meta = self.customer_metadata.get(cid, {})
            turnover_vals.append(meta.get('expected_turnover', 5000.0))
            cust_risk_vals.append(risk_map.get(meta.get('risk_rating', 'LOW'), 1))
            is_corp_vals.append(1 if meta.get('customer_type', 'INDIVIDUAL') != 'INDIVIDUAL' else 0)
            
        df['expected_turnover'] = turnover_vals
        df['customer_risk_level'] = cust_risk_vals
        df['is_corporate_account'] = is_corp_vals
        
        # 2. Amount & Structuring features
        df['amount_log'] = np.log1p(df['amount'].astype(float))
        df['turnover_ratio'] = df['amount'] / np.maximum(df['expected_turnover'], 100.0)
        
        # Structuring proximity: FinCEN $10,000 threshold avoidance
        is_struct = ((df['amount'] >= 9000.0) & (df['amount'] <= 9999.0)).astype(int)
        df['is_structuring_range'] = is_struct
        # Closeness to $9,500 within structuring range
        df['structuring_proximity'] = np.where(
            df['is_structuring_range'] == 1,
            1.0 - (np.abs(df['amount'] - 9500.0) / 500.0),
            0.0
        )
        
        # Exact round numbers flag (e.g., $50,000, $100,000)
        is_round = (
            ((df['amount'] >= 10000.0) & (df['amount'] % 10000.0 == 0.0)) |
            ((df['amount'] >= 5000.0) & (df['amount'] % 5000.0 == 0.0))
        ).astype(int)
        df['is_round_amount'] = is_round
        
        # 3. Jurisdictional & Cross-Border features
        df['is_high_risk_origin'] = df['originator_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_high_risk_dest'] = df['beneficiary_country'].isin(HIGH_RISK_JURISDICTIONS).astype(int)
        df['is_cross_border'] = (df['originator_country'] != df['beneficiary_country']).astype(int)
        df['jurisdiction_risk_score'] = df['is_high_risk_origin'] * 2 + df['is_high_risk_dest'] * 2 + df['is_cross_border']
        
        # 4. Payment Method & Channel Risk
        cash_crypto_types = {'CASH_DEPOSIT', 'CASH_WITHDRAWAL', 'CRYPTO_PURCHASE'}
        df['is_cash_or_crypto'] = df['transaction_type'].isin(cash_crypto_types).astype(int)
        df['is_swift_wire'] = df['channel'].isin({'SWIFT'}).astype(int)
        
        # 5. Temporal & Velocity Features
        df['hour'] = df['dt'].dt.hour
        df['day_of_week'] = df['dt'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night_hours'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Vectorized Velocity in last 24h & flow-through per customer (O(N log N) vs O(N^2))
        df['orig_idx'] = np.arange(len(df))
        df_sorted = df.sort_values(['customer_id', 'dt'])
        
        # 24h rolling count and sum
        roll_stats = df_sorted.set_index('dt').groupby('customer_id')['amount'].rolling('24h').agg(['count', 'sum']).reset_index()
        df_sorted['velocity_24h_count'] = roll_stats['count'].values
        df_sorted['velocity_24h_sum'] = roll_stats['sum'].values
        df_sorted['velocity_24h_sum_log'] = np.log1p(df_sorted['velocity_24h_sum'])
        
        # Rapid flow-through within 6 hours
        prev_dt = df_sorted.groupby('customer_id')['dt'].shift(1)
        prev_amt = df_sorted.groupby('customer_id')['amount'].shift(1)
        time_diff = (df_sorted['dt'] - prev_dt).dt.total_seconds()
        
        is_flow = (
            (time_diff <= 21600) & (time_diff > 0) & 
            (df_sorted['amount'] > 10000) & 
            ((df_sorted['amount'] - prev_amt).abs() / np.maximum(df_sorted['amount'], 1.0) < 0.20)
        ).astype(int)
        df_sorted['rapid_flow_through_flag'] = is_flow
        
        # Restore original order
        df = df_sorted.sort_values('orig_idx').drop(columns=['orig_idx'])
        
        # Drop temporary dt column if created
        if 'dt' in df.columns and 'timestamp' in df.columns:
            df = df.drop(columns=['dt'])
            
        return df

    @staticmethod
    def get_feature_columns():
        return [
            'amount_log',
            'turnover_ratio',
            'is_structuring_range',
            'structuring_proximity',
            'is_round_amount',
            'is_high_risk_origin',
            'is_high_risk_dest',
            'is_cross_border',
            'jurisdiction_risk_score',
            'is_cash_or_crypto',
            'is_swift_wire',
            'customer_risk_level',
            'is_corporate_account',
            'velocity_24h_count',
            'velocity_24h_sum_log',
            'rapid_flow_through_flag',
            'is_weekend',
            'is_night_hours'
        ]
