"""
Unit Tests for FastAPI MLOps Endpoints
Tests /health, /models/info, /predict, /metrics, and /generate-sar
"""

import unittest
from fastapi.testclient import TestClient
from app_api import app

class TestMLOpsAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['models_loaded'])

    def test_models_info_endpoint(self):
        response = self.client.get('/models/info')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('architecture', data)
        self.assertIn('operating_parameters', data)

    def test_predict_endpoint(self):
        payload = {
            'transactions': [
                {
                    'transaction_id': 'TXN-UNITTEST-01',
                    'customer_id': 'CUST-0001',
                    'amount': 9750.00,
                    'transaction_type': 'CASH_DEPOSIT',
                    'channel': 'BRANCH_COUNTER',
                    'originator_country': 'US',
                    'beneficiary_country': 'US'
                }
            ],
            'decision_threshold': 0.35
        }
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        items = response.json()
        self.assertEqual(len(items), 1)
        self.assertIn('sar_risk_score', items[0])
        self.assertIn('is_sar_flagged', items[0])

    def test_metrics_endpoint(self):
        response = self.client.get('/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('metrics', data)
        self.assertIn('accuracy', data['metrics'])

    def test_generate_sar_endpoint(self):
        payload = {
            'customer_id': 'CUST-0001',
            'investigator_name': 'Test_Investigator',
            'investigator_notes': 'Structuring alert',
            'mask_pii': True
        }
        response = self.client.post('/generate-sar', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('statutory_sar_narrative', data)
        self.assertIn('cryptographic_seal_sha256', data)
        self.assertTrue(data['pii_masking_applied'])

    def test_simulation_status_endpoint(self):
        response = self.client.get('/simulation/status')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'READY')
        self.assertIn('target_stream_rate', data)
        self.assertIn('model_performance', data)

    def test_simulation_tick_endpoint(self):
        payload = {
            'batch_size': 50,
            'customer_id': 'ALL',
            'threshold': 0.35,
            'auto_train': True
        }
        response = self.client.post('/simulation/tick', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['batch_size'], 50)
        self.assertIn('sar_flagged_count', data)
        self.assertIn('mlops_training_status', data)
        self.assertIn('exemplars_in_memory', data['mlops_training_status'])

if __name__ == '__main__':
    unittest.main()
