import unittest
from unittest.mock import patch
import numpy as np
from scripts import fit_final_learned_static_v2 as f


class TestProductionV2(unittest.TestCase):
    def fixture(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])
        return logs,['a','b'],['a','b']

    def test_approval_before_reads(self):
        with patch.object(f,'source_binding',side_effect=AssertionError('No reads')):
            with self.assertRaises(PermissionError):f.execute(None)

    def test_verified_boundary_and_full_K_log_semantics(self):
        args=self.fixture();fit=f.solver.solve_diagnostic(*args)
        result=f.verify_fit(*args,fit)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['independent_KKT']['PASS'])
        self.assertAlmostEqual(result['independent_objective'],-np.log(.5))

    def test_nonstationary_candidate_rejected_independently(self):
        args=self.fixture()
        fake=dict(status='CONVERGED_CANDIDATE_NOT_APPROVED',diagnostic_weights=[1.,0,0],
                  KKT_projected_residual=0.,iterations=1,
                  objective=f.previous.objective_readback(*args,[1.,0,0]))
        with self.assertRaisesRegex(ValueError,'Independent KKT'):f.verify_fit(*args,fake)

    def test_residual_and_objective_fail_closed(self):
        args=self.fixture();fit=f.solver.solve_diagnostic(*args)
        for changes in ({'KKT_projected_residual':1.1e-10},{'objective':fit['objective']+.01},{'iterations':10001}):
            with self.assertRaises(ValueError):f.verify_fit(*args,{**fit,**changes})

    def test_simplex_fail_closed(self):
        args=self.fixture();fit=f.solver.solve_diagnostic(*args)
        for weights in ([.3,.3,.3],[1.1,-.1,0],[float('nan'),0,1]):
            with self.assertRaises(ValueError):f.verify_fit(*args,{**fit,'diagnostic_weights':weights})

    def test_exact_family_weights_not_sample_frequency(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])
        a=f.independent_kkt(logs,['a','b'],['a','b'],[.5,.5,0])
        b=f.independent_kkt(np.vstack([logs[0:1]]*9+[logs[1:2]]),['a']*9+['b'],['a','b'],[.5,.5,0])
        self.assertAlmostEqual(float(a['objective']),float(b['objective']))
        self.assertTrue(a['PASS'] and b['PASS'])

    def test_solver_binding_is_exact_D057_source(self):
        self.assertEqual(f.source_binding()['scripts/learned_static_solver_v2.py'],f.SOLVER_SHA)
        self.assertFalse(f.solver.SPEC['empirical_approval'])


if __name__=='__main__':unittest.main()
