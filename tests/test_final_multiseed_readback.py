import copy
import math
import statistics
import unittest
from scripts.readback_final_multiseed import statistics_readback,SEEDS,VIEWS


def fixture():
    rows=[]
    for seed in SEEDS:
        loss=[(seed-40)/10+i*.1 for i in range(3)];drift=[.01*(i+1) for i in range(3)]
        q=[math.exp(-l) for l in loss];qd=[math.exp(-l-d) for l,d in zip(loss,drift)]
        rows.extend(dict(seed=seed,view=view,temperature=2.,L=loss[i],mmd2=2*drift[i],d=drift[i],Q_weight=q[i]/sum(q),QD_weight=qd[i]/sum(qd)) for i,view in enumerate(VIEWS))
    summary=[]
    for view in VIEWS:
        for metric in ('temperature','L','mmd2','d','Q_weight','QD_weight'):
            v=[r[metric] for r in rows if r['view']==view];summary.append(dict(view=view,metric=metric,N=5,mean=statistics.mean(v),SD=statistics.stdev(v),min=min(v),max=max(v)))
    return rows,summary


class TestFinalReadback(unittest.TestCase):
    def test_independent_summary_and_weights(self):statistics_readback(*fixture())

    def test_wrong_ddof_and_missing_seed_stop(self):
        rows,s=fixture();s[1]['SD']*=math.sqrt(4/5)
        with self.assertRaises(ValueError):statistics_readback(rows,s)
        with self.assertRaises(ValueError):statistics_readback(rows[:-1],fixture()[1])

    def test_wrong_weights_stop(self):
        rows,s=fixture();rows[0]['Q_weight']+=.01
        with self.assertRaises(ValueError):statistics_readback(rows,s)


if __name__=='__main__':unittest.main()
