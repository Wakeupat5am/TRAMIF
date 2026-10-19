import unittest
from scripts.backup_research_snapshot import allowed

class TestSnapshotAllowlist(unittest.TestCase):
    def test_exclusions(self):
        for name in ('../secret.json','/etc/a.json','C:/a.json','x\\y.json','data/raw/sample.exe',
                     'data/derived/future_inputs_v1/receipt.json','data/derived/sbsmi_dataset_v1/train/sha.png',
                     'data/derived/entropy_dataset_v1/train/sha.npy'):
            self.assertFalse(allowed(name),name)

    def test_curated_sources_and_state(self):
        for name in ('scripts/evaluate_historical_episode.py','data/derived/pre_future_package/run/package.json',
                     'data/derived/entropy_dataset_v1/config.json','data/derived/prefix_training_v1/E1/sbsmi/run/epoch_100.pt'):
            self.assertTrue(allowed(name),name)

if __name__=='__main__':unittest.main()
