"""
Comprehensive Test Suite for Enterprise AML SAR Narrative Generator
Tests data generation, feature engineering, multi-model evaluation (Accuracy, FPR, FNR),
PII obfuscation, SAR narrative synthesis, and cryptographic audit seal.
"""

import unittest
import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.synthetic_generator import generate_customer_profiles, generate_synthetic_transactions
from src.features.feature_extractor import AMLFeatureExtractor
from src.privacy.pii_masker import PIIMasker
from src.models.rules import AMLRuleEngine
from src.models.detector import AMLDetectionSuite
from src.models.evaluator import ModelEvaluator
from src.generator.sar_narrative import SARNarrativeGenerator
from src.audit.audit_trail import AuditTrailManager

class TestAMLSARSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers = generate_customer_profiles(num_customers=50, seed=42)
        cls.txns = generate_synthetic_transactions(cls.customers, total_records=400, suspicious_ratio=0.25, seed=42)
        cls.extractor = AMLFeatureExtractor(cls.customers)
        cls.features_df = cls.extractor.extract_features(cls.txns)
        cls.masker = PIIMasker()
        cls.detector = AMLDetectionSuite()
        cls.detector.train(cls.features_df)
        cls.evaluator = ModelEvaluator(cls.detector)
        cls.narrative_gen = SARNarrativeGenerator(cls.masker)
        cls.audit_mgr = AuditTrailManager()

    def test_01_synthetic_data_schema_and_typologies(self):
        """Verify synthetic data contains all typologies and required schema."""
        expected_cols = {'transaction_id', 'timestamp', 'customer_id', 'originator_name', 
                         'beneficiary_name', 'amount', 'is_sar_reported', 'aml_typology'}
        self.assertTrue(expected_cols.issubset(set(self.txns.columns)))
        
        typologies = set(self.txns['aml_typology'].unique())
        self.assertIn('NORMAL', typologies)
        self.assertIn('STRUCTURING', typologies)
        self.assertIn('RAPID_FLOW_THROUGH', typologies)
        self.assertIn('MULE_FAN_IN', typologies)
        self.assertIn('ROUND_TRIPPING_SHELL', typologies)
        self.assertIn('DORMANT_SPIKE', typologies)

    def test_02_feature_extraction(self):
        """Verify AML features are non-empty and have no unexpected NaNs."""
        feature_cols = AMLFeatureExtractor.get_feature_columns()
        for col in feature_cols:
            self.assertIn(col, self.features_df.columns)
            self.assertEqual(self.features_df[col].isna().sum(), 0, f"NaNs found in feature: {col}")

    def test_03_pii_masking_and_reversibility(self):
        """Verify PII is masked and can be completely recovered from vault."""
        sample_name = "Jonathan Sterling"
        sample_acct = "ACCT-98124567"
        sample_ip = "192.168.1.105"
        
        m_name = self.masker.mask_entity_name(sample_name)
        m_acct = self.masker.mask_account_number(sample_acct)
        m_ip = self.masker.mask_ip_address(sample_ip)
        
        self.assertTrue(m_name.startswith("[ENTITY_"))
        self.assertTrue(m_acct.startswith("[ACCT_****"))
        self.assertTrue(m_ip.startswith("[IP_192.168."))
        
        # Test text unmasking
        text = f"Notice: Account {m_acct} owned by {m_name} from IP {m_ip}."
        unmasked = self.masker.unmask_text(text)
        self.assertIn(sample_name, unmasked)
        self.assertIn(sample_acct, unmasked)
        self.assertIn(sample_ip, unmasked)

    def test_04_compliance_rule_engine(self):
        """Verify deterministic rule engine triggers on structuring and velocity."""
        engine = AMLRuleEngine()
        row_struct = {
            'amount': 9500.0,
            'is_structuring_range': 1,
            'rapid_flow_through_flag': 0,
            'originator_country': 'US',
            'beneficiary_country': 'GB',
            'expected_turnover': 5000.0
        }
        res = engine.evaluate_transaction(row_struct)
        self.assertEqual(res['is_rule_alert'], 1)
        self.assertTrue(any(r['id'] == 'AML-RULE-101' for r in res['triggered_rules']))

    def test_05_multi_model_accuracy_fpr_fnr_metrics(self):
        """Verify all 4 models calculate valid Accuracy, FPR, FNR, and ROC-AUC."""
        eval_res = self.evaluator.evaluate_all_models(self.features_df, threshold=0.45)
        metrics_list = eval_res['metrics_list']
        self.assertEqual(len(metrics_list), 4)
        
        for m in metrics_list:
            acc = m['accuracy']
            fpr = m['false_positive_rate']
            fnr = m['false_negative_rate']
            auc = m['roc_auc']
            
            self.assertGreaterEqual(acc, 0.70, f"{m['model_name']} accuracy too low: {acc}")
            self.assertGreaterEqual(fpr, 0.0)
            self.assertLessEqual(fpr, 0.20, f"{m['model_name']} FPR too high: {fpr}")
            self.assertGreaterEqual(fnr, 0.0)
            self.assertLessEqual(fnr, 0.40, f"{m['model_name']} FNR too high: {fnr}")
            self.assertGreaterEqual(auc, 0.75, f"{m['model_name']} AUC too low: {auc}")

    def test_06_sar_narrative_generation(self):
        """Verify 5-part FinCEN/FCA narrative synthesis and evidence citations."""
        cust_row = self.customers.iloc[0].to_dict()
        flagged_txns = self.features_df[self.features_df['customer_id'] == cust_row['customer_id']]
        if len(flagged_txns) == 0:
            flagged_txns = self.features_df.head(4)
            
        summary = {
            'ensemble_risk_score': 88.5,
            'triggered_rules': [{'id': 'AML-RULE-101', 'name': 'CTR Structuring', 'severity': 'HIGH'}]
        }
        
        res = self.narrative_gen.generate_narrative_package(
            case_id="SAR-TEST-001",
            customer_profile=cust_row,
            transactions_df=flagged_txns,
            detection_summary=summary,
            is_masked=False
        )
        
        narrative = res['narrative_text']
        self.assertIn("PART I: EXECUTIVE SUMMARY", narrative)
        self.assertIn("PART II: SUBJECT IDENTIFICATION", narrative)
        self.assertIn("PART III: CHRONOLOGICAL SUMMARY", narrative)
        self.assertIn("PART IV: AML TYPOLOGY ANALYSIS", narrative)
        self.assertIn("PART V: CONCLUSION & RECOMMENDED REGULATORY DISPOSITION", narrative)
        self.assertEqual(len(res['lineage']), 5)

    def test_07_audit_trail_and_cryptographic_seal(self):
        """Verify SHA-256 seal is generated, verifies authentic data, and detects tampering."""
        cust_row = self.customers.iloc[0].to_dict()
        flagged_txns = self.features_df.head(3)
        summary = {'ensemble_risk_score': 90.0, 'triggered_rules': []}
        narrative_pkg = {'narrative_text': "Legitimate Compliance Narrative Factual Evidence.", 'lineage': []}
        
        audit_rec = self.audit_mgr.create_audit_record(
            case_id="AUDIT-TEST-01",
            customer_profile=cust_row,
            flagged_transactions=flagged_txns,
            detection_summary=summary,
            narrative_package=narrative_pkg
        )
        
        seal = audit_rec['cryptographic_seal_sha256']
        self.assertEqual(len(seal), 64) # Valid SHA-256 hex string
        
        txns_list = flagged_txns.to_dict(orient='records')
        
        # Valid test
        is_valid, _ = self.audit_mgr.verify_record_integrity(audit_rec, narrative_pkg['narrative_text'], txns_list)
        self.assertTrue(is_valid)
        
        # Tampered test
        tampered_text = narrative_pkg['narrative_text'] + " MODIFIED!"
        is_tampered_valid, _ = self.audit_mgr.verify_record_integrity(audit_rec, tampered_text, txns_list)
        self.assertFalse(is_tampered_valid)

    def test_08_data_preprocessing_and_noise_sanitization(self):
        """Verify AMLDataPreprocessor sanitizes synthetic noise, imputes nulls, and normalizes casing."""
        from src.preprocessing.data_preprocessor import AMLDataPreprocessor
        from src.data.synthetic_generator import inject_synthetic_noise
        
        preprocessor = AMLDataPreprocessor()
        noisy_df = inject_synthetic_noise(self.txns.head(50), noise_ratio=0.30, seed=42)
        
        # Verify noise was injected
        self.assertTrue(noisy_df.isna().sum().sum() > 0 or (noisy_df['channel'].str.contains(' ')).any())
        
        # Run preprocessing
        cleaned_df, report = preprocessor.preprocess(noisy_df)
        
        # Assert no NaNs remain in critical fields
        self.assertEqual(cleaned_df['amount'].isna().sum(), 0)
        self.assertEqual(cleaned_df['device_ip'].isna().sum(), 0)
        self.assertEqual(cleaned_df['beneficiary_account'].isna().sum(), 0)
        self.assertIn('data_health_score', report)
        self.assertGreaterEqual(report['data_health_score'], 70.0)

if __name__ == '__main__':
    unittest.main()
