import unittest
from unittest.mock import patch
import numpy as np
from scripts import fit_final_learned_static as f


class TestFitFinalLearnedStatic(unittest.TestCase):
    def fixture(self):
        rows=[dict(sha='a',family='a'),dict(sha='b',family='b'),dict(sha='c',family='c')]
        bundles={v:dict(logits=np.array([[2.,0,0],[0,2,0],[0,0,2]]),labels=np.array([0,1,2]),sample_identifiers=np.array(['a','b','c']),class_names=np.array(['a','b','c'])) for v in f.VIEWS}
        return bundles,rows,{'a':0,'b':1,'c':2},['a','b'],{v:2. for v in f.VIEWS}

    def test_full_K_target_logs_and_frozen_support(self):
        logs,families,mask=f.targets(*self.fixture())
        expected=1-np.log(np.exp(1)+2)
        np.testing.assert_allclose(logs,np.full((2,3),expected));self.assertEqual(families,['a','b']);self.assertEqual(mask.tolist(),[True,True,False])

    def test_independent_mixture_objective_and_boundary(self):
        logs=np.log([[.9,.1,.2],[.1,.9,.2]])
        self.assertAlmostEqual(f.objective_readback(logs,['a','b'],['a','b'],[.5,.5,0]),-np.log(.5))
        self.assertAlmostEqual(f.objective_readback(logs,['a','b'],['a','b'],[0,0,1]),-np.log(.2))

    def test_ID_alignment_and_class_order_guard(self):
        args=self.fixture();b=args[0]['sbsmi']
        for k in ('logits','labels','sample_identifiers'):b[k]=b[k][::-1]
        f.targets(*args)
        b['class_names']=b['class_names'][::-1]
        with self.assertRaises(ValueError):f.targets(*args)

    def test_approval_guard_before_artifact_reads(self):
        with patch.object(f,'Audit',side_effect=AssertionError('No reads')):
            with self.assertRaises(PermissionError):f.execute(None)

    def test_failure_diagnostic_does_not_register_weights(self):
        try:f.solver.solve(np.log([[.55,.3,.1],[.25,.7,.1]]),['a','b'],['a','b'],max_iterations=1)
        except RuntimeError as exc:
            diagnostic=f.failure_diagnostics(exc);self.assertEqual(diagnostic['iteration'],1);self.assertIn('residual',diagnostic)
        else:self.fail('Expected fixed-budget nonconvergence')


if __name__=='__main__':unittest.main()
