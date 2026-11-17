"""
PII Obfuscation & Reversible Token Vault Engine
Complies with GDPR, GLBA, and banking privacy regulations by obfuscating
Personally Identifiable Information (names, account numbers, tax IDs, IP addresses)
prior to processing, with cryptographic reversible mapping for authorized official filings.
"""

import re
import hashlib

class PIIMasker:
    def __init__(self, salt="ComplianceSAR_2026"):
        self.salt = salt
        self.vault = {}        # token -> real_value
        self.reverse_vault = {}# real_value -> token
        self.counter = 0

    def reset_vault(self):
        self.vault.clear()
        self.reverse_vault.clear()
        self.counter = 0

    def _generate_token(self, prefix, raw_value):
        if raw_value in self.reverse_vault:
            return self.reverse_vault[raw_value]
        
        self.counter += 1
        h = hashlib.sha256(f"{raw_value}_{self.salt}_{self.counter}".encode()).hexdigest()[:6].upper()
        token = f"[{prefix}_{h}]"
        
        self.vault[token] = raw_value
        self.reverse_vault[raw_value] = token
        return token

    def mask_account_number(self, account_no):
        if not account_no or not isinstance(account_no, str):
            return account_no
        # e.g., ACCT-9821453 -> [ACCT_****1453]
        digits = re.sub(r'\D', '', account_no)
        last4 = digits[-4:] if len(digits) >= 4 else digits
        token = f"[ACCT_****{last4}]"
        self.vault[token] = account_no
        self.reverse_vault[account_no] = token
        return token

    def mask_ip_address(self, ip_str):
        if not ip_str or not isinstance(ip_str, str):
            return ip_str
        parts = ip_str.split('.')
        if len(parts) == 4:
            masked = f"[IP_{parts[0]}.{parts[1]}.XX.XX]"
            self.vault[masked] = ip_str
            self.reverse_vault[ip_str] = masked
            return masked
        return self._generate_token("IP", ip_str)

    def mask_entity_name(self, name, entity_type="ENTITY"):
        if not name or not isinstance(name, str):
            return name
        return self._generate_token(entity_type, name)

    def mask_tax_id(self, tax_id):
        if not tax_id or not isinstance(tax_id, str):
            return tax_id
        # e.g. TAX-IND-992144 -> [TAX_****2144]
        digits = re.sub(r'\D', '', tax_id)
        last4 = digits[-4:] if len(digits) >= 4 else digits
        token = f"[TAX_****{last4}]"
        self.vault[token] = tax_id
        self.reverse_vault[tax_id] = token
        return token

    def mask_transaction_record(self, record_dict):
        """Returns a copy of the transaction record with all PII fields masked."""
        rec = dict(record_dict)
        if 'originator_name' in rec:
            rec['originator_name'] = self.mask_entity_name(rec['originator_name'], "ORIG_NAME")
        if 'beneficiary_name' in rec:
            rec['beneficiary_name'] = self.mask_entity_name(rec['beneficiary_name'], "BENEF_NAME")
        if 'originator_account' in rec:
            rec['originator_account'] = self.mask_account_number(rec['originator_account'])
        if 'beneficiary_account' in rec:
            rec['beneficiary_account'] = self.mask_account_number(rec['beneficiary_account'])
        if 'device_ip' in rec:
            rec['device_ip'] = self.mask_ip_address(rec['device_ip'])
        if 'tax_id' in rec:
            rec['tax_id'] = self.mask_tax_id(rec['tax_id'])
        return rec

    def mask_dataframe(self, df):
        """Masks a pandas DataFrame in-place or copy."""
        masked_df = df.copy()
        if 'originator_name' in masked_df.columns:
            masked_df['originator_name'] = masked_df['originator_name'].apply(lambda x: self.mask_entity_name(x, "ORIG_NAME"))
        if 'beneficiary_name' in masked_df.columns:
            masked_df['beneficiary_name'] = masked_df['beneficiary_name'].apply(lambda x: self.mask_entity_name(x, "BENEF_NAME"))
        if 'originator_account' in masked_df.columns:
            masked_df['originator_account'] = masked_df['originator_account'].apply(self.mask_account_number)
        if 'beneficiary_account' in masked_df.columns:
            masked_df['beneficiary_account'] = masked_df['beneficiary_account'].apply(self.mask_account_number)
        if 'device_ip' in masked_df.columns:
            masked_df['device_ip'] = masked_df['device_ip'].apply(self.mask_ip_address)
        if 'tax_id' in masked_df.columns:
            masked_df['tax_id'] = masked_df['tax_id'].apply(self.mask_tax_id)
        return masked_df

    def mask_text(self, text):
        """Masks known entities inside raw text using the reverse vault."""
        out = text
        for real_val, token in sorted(self.reverse_vault.items(), key=lambda x: len(x[0]), reverse=True):
            if real_val in out:
                out = out.replace(real_val, token)
        return out

    def unmask_text(self, text):
        """Re-hydrates masked tokens back to true identities for official authorized filing."""
        out = text
        for token, real_val in sorted(self.vault.items(), key=lambda x: len(x[0]), reverse=True):
            if token in out:
                out = out.replace(token, real_val)
        return out
