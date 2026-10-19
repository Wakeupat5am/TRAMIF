import unittest
import numpy as np
from scripts import learned_static_solver as s


class TestLearnedStatic(unittest.TestCase):
    def test_dominant_view_boundary_solution(self):
        result=s.solve(np.log([[.9,.3,.1],[.8,.2,.1]]),['a','b'],['a','b'])
        np.testing.assert_allclose(result['weights'],[1,0,0],atol=1e-12)

    def test_known_balanced_mixture_and_determinism(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])
        result=s.solve(logs,['a','b'],['a','b']);self.assertEqual(result,s.solve(logs,['a','b'],['a','b']))
        np.testing.assert_allclose(result['weights'],[.5,.5,0],atol=1e-10)
        self.assertAlmostEqual(result['objective'],-np.log(.5))

    def test_flat_tie_preserves_uniform(self):
        result=s.solve(np.log([[.2,.2,.2],[.5,.5,.5]]),['a','a'],['a'])
        np.testing.assert_array_equal(result['weights'],[1/3]*3)

    def test_class_balance_invariant_to_repeating_family(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])
        a=s.solve(logs,['a','b'],['a','b']);b=s.solve(np.vstack([logs[0:1]]*9+[logs[1:2]]),['a']*9+['b'],['a','b'])
        np.testing.assert_allclose(a['weights'],b['weights'],atol=1e-12)

    def test_gradient_finite_difference_and_extreme_logdomain(self):
        args=s.prepare([[-1000,-1001,-1002],[-3,-2,-1]],['a','b'],['a','b']);w=np.array([.3,.4,.3])
        _,g=s.objective_gradient(w,*args)
        direction=np.array([1.,-1,0]);eps=1e-6
        numerical=(s.objective_gradient(w+eps*direction,*args)[0]-s.objective_gradient(w-eps*direction,*args)[0])/(2*eps)
        self.assertAlmostEqual(numerical,float(g@direction),places=7)
        self.assertTrue(np.isfinite(s.solve([[-1000,-1001,-1002]],['a'],['a'])['objective']))

    def test_invalid_inputs_and_no_convergence_fail_closed(self):
        for logs,families,support in (([[np.nan,0,0]],['a'],['a']),([[1,0,0]],['a'],['a']),([[-1,-2,-3]],['a'],['b'])):
            with self.assertRaises(ValueError):s.solve(logs,families,support)
        with self.assertRaises(RuntimeError):s.solve(np.log([[.55,.3,.1],[.25,.7,.1]]),['a','b'],['a','b'],max_iterations=1)


if __name__=='__main__':unittest.main()
