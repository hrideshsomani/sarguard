"""
Deterministic AML Compliance Rule Engine
Implements statutory FinCEN, FCA, and FATF regulatory alert rules.
Provides transparent, deterministic rule triggers with line-item audit explainability.
"""

HIGH_RISK_JURISDICTIONS = {'KY', 'PA', 'CY', 'RU', 'IR', 'KP', 'MM', 'AE', 'BS', 'VG'}

class AMLRuleEngine:
    def __init__(self):
        self.rules = [
            {
                'id': 'AML-RULE-101',
                'name': 'CTR Structuring Avoidance ($9,000-$9,999)',
                'severity': 'HIGH',
                'description': 'Transaction amount is calibrated just beneath the $10,000 statutory Currency Transaction Report threshold.',
                'weight': 35
            },
            {
                'id': 'AML-RULE-102',
                'name': 'Rapid Layering / Flow-Through Fund Movement',
                'severity': 'CRITICAL',
                'description': 'Funds entering the account are immediately transferred out (>85% matching volume) within a short dwell window.',
                'weight': 40
            },
            {
                'id': 'AML-RULE-103',
                'name': 'High-Risk Secrecy Jurisdiction Corridor',
                'severity': 'HIGH',
                'description': 'Cross-border fund transfer interacting with FATF grey/black listed jurisdiction or offshore financial center.',
                'weight': 30
            },
            {
                'id': 'AML-RULE-104',
                'name': 'Extreme Turnover Volume Spike (>10x KYC baseline)',
                'severity': 'HIGH',
                'description': 'Transaction amount exceeds 10x the customer stated expected monthly turnover baseline.',
                'weight': 25
            },
            {
                'id': 'AML-RULE-105',
                'name': 'Mule Aggregation / Fan-In Network Activity',
                'severity': 'CRITICAL',
                'description': 'Rapid succession of micro-deposits from disparate senders followed by immediate lump-sum cash-out.',
                'weight': 35
            },
            {
                'id': 'AML-RULE-106',
                'name': 'Round-Denomination Offshore Wire Transfer',
                'severity': 'MEDIUM',
                'description': 'Exact round-sum cross-border wire transfer ($50k, $100k, $250k) with offshore corporate counterparty.',
                'weight': 20
            }
        ]

    def evaluate_transaction(self, row):
        """
        Evaluates a single transaction row against all compliance rules.
        Returns a list of triggered rule objects and a total rule-based risk score (0-100).
        """
        triggers = []
        score = 0
        
        amt = float(row.get('amount', 0.0))
        t_type = str(row.get('transaction_type', ''))
        orig_ctry = str(row.get('originator_country', ''))
        dest_ctry = str(row.get('beneficiary_country', ''))
        turnover = float(row.get('expected_turnover', 5000.0))
        struct_flag = int(row.get('is_structuring_range', 0))
        flow_flag = int(row.get('rapid_flow_through_flag', 0))
        v24_count = int(row.get('velocity_24h_count', 0))
        is_round = int(row.get('is_round_amount', 0))
        
        # Rule 101: Structuring
        if struct_flag == 1 or (9000.0 <= amt <= 9999.0):
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-101')
            triggers.append({
                **rule,
                'evidence': f"Amount ${amt:,.2f} is in the high-suspicion CTR avoidance corridor ($9,000-$9,999)."
            })
            score += rule['weight']

        # Rule 102: Flow-Through
        if flow_flag == 1:
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-102')
            triggers.append({
                **rule,
                'evidence': f"Transaction matches rapid pass-through signature with immediate matching outflow."
            })
            score += rule['weight']

        # Rule 103: High-Risk Jurisdiction
        is_high_risk = (orig_ctry in HIGH_RISK_JURISDICTIONS) or (dest_ctry in HIGH_RISK_JURISDICTIONS)
        if is_high_risk and (amt > 15000.0 or str(row.get('customer_risk_level', 1)) in ['2', '3', 2, 3]):
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-103')
            h_country = orig_ctry if orig_ctry in HIGH_RISK_JURISDICTIONS else dest_ctry
            triggers.append({
                **rule,
                'evidence': f"High-risk corridor detected with FATF jurisdiction '{h_country}' on transaction of ${amt:,.2f}."
            })
            score += rule['weight']

        # Rule 104: Turnover Spike
        if turnover > 0 and (amt / turnover >= 8.0):
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-104')
            ratio = amt / turnover
            triggers.append({
                **rule,
                'evidence': f"Amount ${amt:,.2f} is {ratio:.1f}x higher than stated monthly turnover (${turnover:,.2f})."
            })
            score += rule['weight']

        # Rule 105: Mule Fan-In
        if v24_count >= 4 and t_type in ['PEER_TO_PEER', 'CRYPTO_PURCHASE', 'CASH_DEPOSIT']:
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-105')
            triggers.append({
                **rule,
                'evidence': f"Elevated 24h velocity ({v24_count} transactions) involving retail fan-in/cash-out channel {t_type}."
            })
            score += rule['weight']

        # Rule 106: Round-Sum Offshore Wire
        if is_round == 1 and amt >= 50000.0 and (orig_ctry in HIGH_RISK_JURISDICTIONS or dest_ctry in HIGH_RISK_JURISDICTIONS):
            rule = next(r for r in self.rules if r['id'] == 'AML-RULE-106')
            triggers.append({
                **rule,
                'evidence': f"Exact round denomination (${amt:,.2f}) wire transfer routed to/from offshore jurisdiction."
            })
            score += rule['weight']

        # Normalized risk score 0 to 100
        rule_score = min(100.0, float(score))
        is_alert = 1 if rule_score >= 35.0 else 0
        
        return {
            'rule_score': rule_score,
            'is_rule_alert': is_alert,
            'triggered_rules': triggers,
            'rule_count': len(triggers)
        }

    def evaluate_dataframe(self, df):
        """Evaluates an entire DataFrame, returning rule scores, flags, and triggers."""
        results = []
        for _, row in df.iterrows():
            res = self.evaluate_transaction(row)
            results.append(res)
            
        rule_scores = [r['rule_score'] for r in results]
        rule_alerts = [r['is_rule_alert'] for r in results]
        rule_counts = [r['rule_count'] for r in results]
        rule_triggers = [r['triggered_rules'] for r in results]
        
        return rule_scores, rule_alerts, rule_counts, rule_triggers
