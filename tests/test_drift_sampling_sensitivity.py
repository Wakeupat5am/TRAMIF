import unittest
from scripts.check_drift_sampling_sensitivity import selection_indices

class TestSelection(unittest.TestCase):
    def test_identity_alignment_not_position(self):
        rows=[dict(sha='a',family='x',group_id='shared'),dict(sha='b',family='y',group_id='shared')]
        subset=[dict(case='E1',**rows[1]),dict(case='E1',**rows[0])]
        self.assertEqual(selection_indices(rows,subset),[1,0])
        self.assertEqual(rows[0]['group_id'],rows[1]['group_id'])

    def test_fail_closed(self):
        rows=[dict(sha='a',family='x',group_id='g')]
        for subset in ([],rows*2,[dict(sha='b')],[dict(sha='a',family='y')]):
            with self.assertRaises((ValueError,RuntimeError)): selection_indices(rows,subset)

if __name__=='__main__': unittest.main()
