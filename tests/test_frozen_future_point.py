import unittest
import numpy as np
from unittest.mock import patch
from scripts import evaluate_frozen_future as future
from scripts.evaluate_frozen_future import monthly_point_metrics,MONTHS


class TestFuturePointMath(unittest.TestCase):
    def bundles(self):
        # Entirely synthetic: one example per month, no actual future samples.
        labels=np.array([0,1,0,1,0,1]);scores=np.zeros((6,51));scores[np.arange(6),labels]=4
        return [dict(ids=np.array([str(i) for i in range(6)]),classes=np.array([str(i) for i in range(51)]),
                     labels=labels.copy(),logits=scores.copy(),temperature=1.) for _ in range(3)]

    def dates(self):return [m+'-01 00:00:00+00:00' for m in MONTHS]

    def test_six_month_mean_worst_and_absent_classes(self):
        names,logs,monthly,result=monthly_point_metrics(self.bundles(),self.dates(),[1/3]*3,[.5,.2,.3])
        self.assertEqual(logs.shape,(6,6,51));self.assertEqual(len(monthly),6)
        for name in names:
            self.assertEqual(result[name]['mean_monthly_macro_f1'],1.)
            self.assertEqual(result[name]['worst_month_macro_f1'],1.)
            self.assertAlmostEqual(monthly[MONTHS[0]][name]['macro_f1_fixed'],1/51)

    def test_missing_month_and_outside_month_fail(self):
        dates=self.dates();dates[-1]=dates[0]
        with self.assertRaises(ValueError):monthly_point_metrics(self.bundles(),dates,[1/3]*3,[1/3]*3)
        dates=self.dates();dates[-1]='2020-10-01 00:00:00+00:00'
        with self.assertRaises(ValueError):monthly_point_metrics(self.bundles(),dates,[1/3]*3,[1/3]*3)

    def test_alignment_fail(self):
        b=self.bundles();b[1]['ids']=b[1]['ids'][::-1]
        with self.assertRaises(ValueError):monthly_point_metrics(b,self.dates(),[1/3]*3,[1/3]*3)

    def test_cli_guard_precedes_any_file_read(self):
        args=['eval','--package','nonexistent','--package-sha256','none','--inputs','nonexistent','--input-receipt-sha256','none']
        with patch('sys.argv',args),patch.object(future,'read_hashed',side_effect=AssertionError('Must not read')) as read:
            with self.assertRaises(SystemExit):future.main()
            read.assert_not_called()


if __name__=='__main__':unittest.main()
