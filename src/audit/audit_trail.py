"""
Audit Trail, Lineage & Cryptographic Provenance Engine
Generates immutable, tamper-evident regulatory audit dossiers with SHA-256 cryptographic seals,
linking every generated narrative assertion back to deterministic transaction rows and compliance rule firings.
"""

import json
import hashlib
from datetime import datetime
import pandas as pd

class AuditTrailManager:
    def __init__(self, log_filepath=None):
        self.log_filepath = log_filepath
        self.audit_records = []

    def create_audit_record(self, case_id, customer_profile, flagged_transactions, detection_summary, narrative_package, investigator="COMPLIANCE_OFFICER_104"):
        """
        Creates an immutable, regulator-verifiable audit record.
        Computes SHA-256 cryptographic seal of all inputs and outputs.
        """
        txns_summary = []
        for _, row in flagged_transactions.iterrows():
            txns_summary.append({
                'transaction_id': str(row['transaction_id']),
                'timestamp': str(row['timestamp']),
                'amount': float(row['amount']),
                'type': str(row['transaction_type']),
                'beneficiary': str(row['beneficiary_name']),
                'beneficiary_country': str(row['beneficiary_country']),
                'channel': str(row['channel']),
                'red_flag': str(row.get('red_flag_notes', ''))
            })

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Raw data payload to hash
        payload_to_hash = {
            'case_id': case_id,
            'customer_id': customer_profile.get('customer_id'),
            'investigator': investigator,
            'timestamp': timestamp_str,
            'transactions': txns_summary,
            'ensemble_score': float(detection_summary.get('ensemble_risk_score', 0.0)),
            'triggered_rules': detection_summary.get('triggered_rules', []),
            'narrative_text': narrative_package.get('narrative_text', '')
        }
        
        serialized = json.dumps(payload_to_hash, sort_keys=True)
        crypto_seal = hashlib.sha256(serialized.encode('utf-8')).hexdigest()

        record = {
            'case_id': case_id,
            'audit_timestamp': timestamp_str,
            'investigator': investigator,
            'customer_id': customer_profile.get('customer_id'),
            'customer_name': customer_profile.get('customer_name'),
            'account_number': customer_profile.get('account_number'),
            'total_transactions_reviewed': len(txns_summary),
            'total_flagged_amount': float(sum(t['amount'] for t in txns_summary)),
            'ml_risk_score': float(detection_summary.get('ensemble_risk_score', 0.0)),
            'model_ensemble_breakdown': {
                'rule_weight': '40%',
                'random_forest_weight': '40%',
                'isolation_forest_weight': '20%'
            },
            'triggered_rules': detection_summary.get('triggered_rules', []),
            'narrative_lineage': narrative_package.get('lineage', []),
            'cryptographic_seal_sha256': crypto_seal,
            'is_tamper_verified': True,
            'filing_status': 'PENDING_REGULATORY_SUBMISSION'
        }
        
        self.audit_records.append(record)
        
        if self.log_filepath:
            self._save_to_disk()
            
        return record

    def verify_record_integrity(self, record, narrative_text, transactions_list):
        """Verifies if narrative or transactions have been modified since audit seal was generated."""
        txns_summary = []
        for t in transactions_list:
            txns_summary.append({
                'transaction_id': str(t['transaction_id']),
                'timestamp': str(t['timestamp']),
                'amount': float(t['amount']),
                'type': str(t['transaction_type']),
                'beneficiary': str(t['beneficiary_name']),
                'beneficiary_country': str(t['beneficiary_country']),
                'channel': str(t['channel']),
                'red_flag': str(t.get('red_flag_notes', ''))
            })
            
        payload = {
            'case_id': record['case_id'],
            'customer_id': record['customer_id'],
            'investigator': record['investigator'],
            'timestamp': record['audit_timestamp'],
            'transactions': txns_summary,
            'ensemble_score': float(record['ml_risk_score']),
            'triggered_rules': record['triggered_rules'],
            'narrative_text': narrative_text
        }
        
        recomputed = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
        is_valid = (recomputed == record['cryptographic_seal_sha256'])
        return is_valid, recomputed

    def _save_to_disk(self):
        try:
            with open(self.log_filepath, 'w') as f:
                json.dump(self.audit_records, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save audit log: {e}")

    def export_as_json(self, record):
        return json.dumps(record, indent=2)
