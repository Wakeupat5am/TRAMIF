from copy import deepcopy
import hashlib
import json
import unittest
from scripts.freeze_retention_references import validate_inventory
from tests.test_d060_integration import retention_row


class TestRetentionReferenceBinding(unittest.TestCase):
    def fixtures(self):
        config={'frozen':'synthetic'}
        h=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        row=retention_row(1,'2020-02-01T00:00:00+00:00');row.update(config_sha256=h,file_size='10')
        summary=dict(config_sha256=h,errors=0,processed_samples=1,complete_validation_inventory=True)
        return [row],config,summary,[{k:row[k] for k in ('sha','split','timestamp','family')}]

    def test_valid_full_inventory_includes_excluded_records(self):
        rows,c,s,canonical=self.fixtures();rows[0]['retained']='False'
        result=validate_inventory(rows,c,s,canonical,1,'validation')
        self.assertEqual(len(result),1)

    def test_duplicate_or_missing_id_fails(self):
        rows,c,s,canonical=self.fixtures()
        with self.assertRaises(ValueError):validate_inventory(rows+rows,c,s,canonical,1,'validation')
        canonical[0]['sha']='missing'
        with self.assertRaises(ValueError):validate_inventory(rows,c,s,canonical,1,'validation')

    def test_future_or_wrong_split_rejected(self):
        rows,c,s,canonical=self.fixtures();rows[0]['timestamp']='2020-04-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'cutoff'):validate_inventory(rows,c,s,canonical,1,'validation')

    def test_config_or_label_identity_mismatch_rejected(self):
        rows,c,s,canonical=self.fixtures();c['changed']=True
        with self.assertRaisesRegex(ValueError,'config'):validate_inventory(rows,c,s,canonical,1,'validation')
        rows,c,s,canonical=self.fixtures();rows[0]['family']='changed'
        with self.assertRaisesRegex(ValueError,'identity'):validate_inventory(rows,c,s,canonical,1,'validation')

    def test_incomplete_receipt_rejected(self):
        rows,c,s,canonical=self.fixtures();s['complete_validation_inventory']=False
        with self.assertRaises(ValueError):validate_inventory(rows,c,s,canonical,1,'validation')


if __name__=='__main__':unittest.main()
