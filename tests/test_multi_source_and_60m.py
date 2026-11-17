"""
Unit Tests for Multi-Source Dataset Ingestion and 60M Verification Engine
Tests IBM AML mapping, PaySim mapping, unified streaming, and model calibration.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.multi_source_dataset import MultiSourceAMLDataset
from src.features.feature_extractor import AMLFeatureExtractor
from src.models.detector import AMLDetectionSuite
from src.data.synthetic_generator import generate_customer_profiles, generate_incremental_transactions

class TestMultiSourceAnd60M(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers = generate_customer_profiles(num_customers=30, seed=42)
        cls.multi_ds = MultiSourceAMLDataset()
        cls.extractor = AMLFeatureExtractor(cls.customers)

    def test_01_ibm_schema_transformation(self):
        """Verify raw IBM AML format maps cleanly to canonical schema."""
        raw_mock = pd.DataFrame([{
            'Timestamp': '2026/01/15 14:30',
            'From Bank': 12,
            'Account': '88721A',
            'To Bank': 15,
            'Account.1': '99432B',
            'Amount Paid': 9500.0,
            'Amount Received': 9500.0,
            'Payment Currency': 'US Dollar',
            'Payment Format': 'Wire',
            'Is Laundering': 1
        }])
        df = self.multi_ds._transform_ibm_df(raw_mock)
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]['amount'], 9500.0)
        self.assertEqual(df.iloc[0]['is_sar_reported'], 1)
        self.assertEqual(df.iloc[0]['aml_typology'], 'STRUCTURING')
        self.assertEqual(df.iloc[0]['dataset_source'], 'IBM_AML_WORLD')

    def test_02_paysim_schema_transformation(self):
        """Verify PaySim mobile money schema maps cleanly."""
        raw_mock = pd.DataFrame([{
            'step': 24,
            'type': 'TRANSFER',
            'amount': 120000.0,
            'nameOrig': 'C1000000001',
            'nameDest': 'C2000000002',
            'oldbalanceOrg': 150000.0,
            'newbalanceOrig': 30000.0,
            'oldbalanceDest': 1000.0,
            'newbalanceDest': 121000.0,
            'isFraud': 1,
            'isFlaggedFraud': 1
        }])
        df = self.multi_ds._transform_paysim_df(raw_mock)
        self.assertFalse(df.empty)
        self.assertEqual(df.iloc[0]['amount'], 120000.0)
        self.assertEqual(df.iloc[0]['is_sar_reported'], 1)
        self.assertEqual(df.iloc[0]['aml_typology'], 'RAPID_LAYERING')
        self.assertEqual(df.iloc[0]['dataset_source'], 'PAYSIM_FINANCIAL')

    def test_03_feature_extraction_on_multi_source(self):
        """Ensure all 18 quantitative features extract without NaNs on multi-source data."""
        raw_ibm = pd.DataFrame([{
            'Timestamp': '2026/01/15 14:30',
            'From Bank': 10,
            'Account': 'ACC1',
            'To Bank': 20,
            'Account.1': 'ACC2',
            'Amount Paid': 9500.0,
            'Payment Format': 'Wire',
            'Is Laundering': 1
        }])
        df_ibm = self.multi_ds._transform_ibm_df(raw_ibm)
        feats = self.extractor.extract_features(df_ibm)
        for col in AMLFeatureExtractor.get_feature_columns():
            self.assertIn(col, feats.columns)
            self.assertEqual(feats[col].isna().sum(), 0)

    def test_04_detection_suite_scoring(self):
        """Test hybrid calibrated ensemble prediction on transformed multi-source stream."""
        raw_ibm = pd.DataFrame([{
            'Timestamp': '2026/01/15 14:30',
            'From Bank': 10,
            'Account': 'ACC1',
            'To Bank': 20,
            'Account.1': 'ACC2',
            'Amount Paid': 9500.0,
            'Payment Format': 'Wire',
            'Is Laundering': 1
        }])
        df = self.multi_ds._transform_ibm_df(raw_ibm)
        feats = self.extractor.extract_features(df)
        detector = AMLDetectionSuite()
        
        # Quick train on small synthetic baseline
        baseline = generate_incremental_transactions(self.customers, batch_size=100)
        base_feats = self.extractor.extract_features(baseline)
        detector.train(base_feats)
        
        preds, ensemble_scores, risk_scores_100, triggers = detector.predict_hybrid_ensemble(feats, threshold=0.35)
        self.assertEqual(len(preds), 1)
        self.assertTrue(0 <= ensemble_scores[0] <= 1.0)
        self.assertTrue(0 <= risk_scores_100[0] <= 100.0)

    def test_05_experience_replay_memory_bank(self):
        """Verify AMLExemplarMemoryBank preserves past training exemplars without forgetting."""
        from src.models.detector import AMLExemplarMemoryBank
        bank = AMLExemplarMemoryBank(capacity=500)
        
        batch1 = pd.DataFrame({
            'transaction_id': [f'TX1_{i}' for i in range(50)],
            'amount': [100.0 * i for i in range(50)],
            'is_sar_reported': [i % 2 for i in range(50)],
            'dataset_source': ['IBM_AML_WORLD'] * 50
        })
        bank.add_exemplars(batch1)
        self.assertGreaterEqual(len(bank.reservoir_df), 25)
        
        # Add diverse second batch from a different dataset
        batch2 = pd.DataFrame({
            'transaction_id': [f'TX2_{i}' for i in range(50)],
            'amount': [200.0 * i for i in range(50)],
            'is_sar_reported': [i % 2 for i in range(50)],
            'dataset_source': ['CRYPTO_BLOCKCHAIN_AML'] * 50
        })
        bank.add_exemplars(batch2)
        
        replay = bank.get_replay_sample(n=20)
        self.assertEqual(len(replay), 20)
        # Verify both historical datasets exist in memory bank replay
        sources = set(bank.reservoir_df['dataset_source'].unique())
        self.assertIn('IBM_AML_WORLD', sources)
        self.assertIn('CRYPTO_BLOCKCHAIN_AML', sources)

    def test_06_new_heterogeneous_datasets(self):
        """Verify Crypto AML, Card Fraud, and TBML dataset batch generators."""
        df_crypto = self.multi_ds.fetch_crypto_aml_batch(batch_size=10)
        self.assertEqual(len(df_crypto), 10)
        self.assertEqual(df_crypto.iloc[0]['dataset_source'], 'CRYPTO_BLOCKCHAIN_AML')
        
        df_card = self.multi_ds.fetch_card_fraud_batch(batch_size=10)
        self.assertEqual(len(df_card), 10)
        self.assertEqual(df_card.iloc[0]['dataset_source'], 'CARD_MERCHANT_FRAUD')
        
        df_tbml = self.multi_ds.fetch_tbml_trade_batch(batch_size=10)
        self.assertEqual(len(df_tbml), 10)
        self.assertEqual(df_tbml.iloc[0]['dataset_source'], 'TRADE_BASED_TBML')

if __name__ == '__main__':
    unittest.main()
