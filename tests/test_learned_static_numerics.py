"""D057 synthetic only: independent scalar Decimal/KKT and closed-form checks."""
import math
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch

import numpy as np
from scripts import learned_static_solver as old
from scripts import learned_static_solver_v2 as new
from scripts.diagnose_learned_static import DecimalReference, decimal_projection, replay


def represented(w):
    values = [Decimal.from_float(float(x)) for x in w]
    pivot = int(np.argmax(w))
    values[pivot] = 1-sum(x for j, x in enumerate(values) if j != pivot)
    return values


class TestLearnedStaticNumerics(unittest.TestCase):
    def assert_kkt(self, logs, families, support, result):
        with localcontext() as ctx:
            ctx.prec = 60
            _, alpha, _ = old.prepare(logs, families, support)
            ref = DecimalReference(logs, alpha, exact_logs=True)
            w = represented(result['diagnostic_weights'])
            g = ref.gradient(w)
            active = [j for j, x in enumerate(w) if x > Decimal('1e-12')]
            multiplier = sum(g[j] for j in active)/len(active)
            self.assertLess(max(abs(g[j]-multiplier) for j in active), Decimal('5e-10'))
            for j in set(range(3))-set(active):
                self.assertGreaterEqual(g[j]-multiplier, Decimal('-5e-10'))
            self.assertLessEqual(result['KKT_projected_residual'], 1e-10)
            self.assertFalse(result['weights_registered'])
            self.assertNotIn('weights', result)

    def test_protocol_parameters_unchanged_and_candidate_isolated(self):
        for key in old.SPEC:
            self.assertEqual(old.SPEC[key], new.SPEC[key])
        self.assertFalse(new.SPEC['empirical_approval'])

    def test_projection_vs_decimal_all_active_sets(self):
        cases = [[.2,.6,.1], [3,-2,1], [-100,-99,-101], [1,1,1], [1e-9,.5,.5]]
        with localcontext() as ctx:
            ctx.prec = 60
            for x in cases:
                expected = decimal_projection([Decimal.from_float(float(t)) for t in x])
                actual = new.project(x)
                np.testing.assert_allclose(actual, [float(t) for t in expected], rtol=0, atol=3e-15)
                self.assertTrue((actual >= 0).all())
                self.assertLessEqual(abs(math.fsum(actual)-1), 2*np.finfo(float).eps)

    def test_gradient_multiple_finite_difference_scales(self):
        logs = np.log([[.8,.2,.1], [.1,.7,.4], [.2,.3,.6]])
        a, alpha, offset = old.prepare(logs, ['a','a','b'], ['a','b'])
        w = np.array([.3,.4,.3])
        g = new.centered_gradient(w,a,alpha,0)
        for d in (np.array([1.,-1,0]), np.array([0.,1,-1])):
            for h in (1e-3,1e-4,1e-5,1e-6):
                derivative=(new.objective(w+h*d,a,alpha,offset)-new.objective(w-h*d,a,alpha,offset))/(2*h)
                self.assertLess(abs(derivative-g@d), 3*h*h+1e-9)

    def test_gradient_independent_raw_log_exp_reference(self):
        logs = np.array([[-1000,-1001,-1002],[-5,-2,-3]], dtype=float)
        a, alpha, _ = old.prepare(logs,['a','b'],['a','b'])
        w = np.array([.3,.4,.3])
        with localcontext() as ctx:
            ctx.prec=60
            g=DecimalReference(logs,alpha,exact_logs=True).gradient(represented(w))
            np.testing.assert_allclose(new.centered_gradient(w,a,alpha,1),[float(x-g[1]) for x in g],rtol=0,atol=3e-15)

    def test_centered_projection_preserves_stationarity_definition(self):
        a,alpha,offset=old.prepare(np.log([[.8,.2,.1],[.1,.7,.4]]),['a','b'],['a','b'])
        for w in (np.array([.3,.4,.3]),np.array([1.,0,0]),np.array([0.,.6,.4])):
            _,g=old.objective_gradient(w,a,alpha,offset)
            original=float(max(abs(w-old.simplex_projection(w-g))))
            corrected,_,_=new.stationarity(w,a,alpha)
            self.assertLessEqual(abs(original-corrected),3e-15)

    def test_armijo_gap_with_changing_largest_coordinate(self):
        logs=np.log([[.8,.2,.1],[.1,.7,.4],[.2,.3,.6]])
        a,alpha,_=old.prepare(logs,['a','a','b'],['a','b'])
        with localcontext() as ctx:
            ctx.prec=70
            ref=DecimalReference(a,alpha)
            for w,v in (([.1,.8,.1],[.8,.1,.1]),([0.,.6,.4],[1.,0.,0.]),([.2,.3,.5],[.2,.5,.3])):
                dw,dv=represented(w),represented(v)
                g=ref.gradient(dw);gd=sum(x*(b-c) for x,b,c in zip(g,dv,dw))
                exact=ref.difference(dw,dv)-Decimal('0.0001')*gd
                actual=new.armijo_gap(np.array(w),np.array(v),a,alpha,int(np.argmax(w)))
                self.assertLess(abs(actual['gap']-float(exact)),2e-15)
                self.assertEqual(actual['accepted'],bool(exact<=0 and gd<0))

    def test_log_remainder_decimal_oracle_and_switch(self):
        z=np.array([-0.999999,-.1,-.00100001,-.001, -.00099999,-1e-9,0,1e-12,.001,.00100001,1,100])
        with localcontext() as ctx:
            ctx.prec=70
            expected=[]
            for value in z:
                x=Decimal.from_float(float(value));expected.append(float(x-(1+x).ln()))
        np.testing.assert_allclose(new.log_remainder(z),expected,rtol=5e-13,atol=0)
        with self.assertRaises(FloatingPointError):new.log_remainder([-1])

    def test_small_reduction_hidden_by_absolute_objective_is_resolved(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])-1000
        a,alpha,offset=old.prepare(logs,['a','b'],['a','b'])
        w=np.array([.5+1e-9,.5-1e-9,0.]);v=np.array([.5,.5,0.])
        self.assertEqual(new.objective(w,a,alpha,offset),new.objective(v,a,alpha,offset))
        gap=new.armijo_gap(w,v,a,alpha,0)
        self.assertTrue(gap['accepted']);self.assertLess(gap['gap'],0)
        with localcontext() as ctx:
            ctx.prec=70
            ref=DecimalReference(a,alpha);dw=represented(w);dv=represented(v)
            g=ref.gradient(dw);gd=sum(x*(b-c) for x,b,c in zip(g,dv,dw))
            expected=ref.difference(dw,dv)-Decimal('0.0001')*gd
        self.assertAlmostEqual(gap['gap']/float(expected),1.,places=6)

    def test_storage_mass_does_not_create_spurious_descent(self):
        logs=np.log([[.8,.2,.3],[.2,.8,.3]])
        a,alpha,_=old.prepare(logs,['a','b'],['a','b'])
        w=np.array([.6,.25,.15]);v=w.copy();v[0]=np.nextafter(v[0],np.inf)
        # Same implicit simplex point, different rounded stored largest coordinate.
        result=new.armijo_gap(w,v,a,alpha,0)
        self.assertEqual(result['gap'],0);self.assertFalse(result['accepted'])
        self.assertEqual(result['directional_derivative'],0)

    def test_reject_true_ascent_even_when_absolute_objective_rounds_equal(self):
        logs=np.log([[.9,.1,.05],[.1,.9,.05]])-1000
        a,alpha,offset=old.prepare(logs,['a','b'],['a','b'])
        w=np.array([.5,.5,0.]);v=np.array([.5+1e-9,.5-1e-9,0.])
        self.assertEqual(new.objective(w,a,alpha,offset),new.objective(v,a,alpha,offset))
        gap=new.armijo_gap(w,v,a,alpha,0)
        self.assertFalse(gap['accepted']);self.assertGreater(gap['gap'],0)

    def test_known_interior_optimum_and_kkt(self):
        rows=np.repeat(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),[60,25,15],axis=0)
        logs=np.log(rows);families=['a']*100
        result=new.solve_diagnostic(logs,families,['a'])
        np.testing.assert_allclose(result['diagnostic_weights'],(np.array([.6,.25,.15])-.1)/.7,rtol=0,atol=2e-10)
        self.assert_kkt(logs,families,['a'],result)

    def test_known_boundary_optima(self):
        for rows,expected in (([[.9,.3,.1],[.8,.2,.1]],[1,0,0]),([[.9,.1,.05],[.1,.9,.05]],[.5,.5,0])):
            logs=np.log(rows);result=new.solve_diagnostic(logs,['a','b'],['a','b'])
            np.testing.assert_allclose(result['diagnostic_weights'],expected,rtol=0,atol=1e-10)
            self.assert_kkt(logs,['a','b'],['a','b'],result)

    def test_nearly_identical_views_and_exact_flat_tie(self):
        for rows in ([[.3,.3,.3],[.2,.2,.2]],[[.3+1e-8,.3,.3-1e-8],[.3-1e-8,.3,.3+1e-8]]):
            logs=np.log(rows);result=new.solve_diagnostic(logs,['a','b'],['a','b'])
            np.testing.assert_array_equal(result['diagnostic_weights'],[1/3]*3)
            self.assert_kkt(logs,['a','b'],['a','b'],result)

    def test_ill_conditioned_case_is_not_false_convergence(self):
        # Very weak discrimination: a small iteration budget cannot identify optimum.
        logs=np.log([[.301,.3,.299],[.301,.3,.299]])
        with self.assertRaisesRegex(RuntimeError,'Convergence tolerance not reached'):
            new.solve_diagnostic(logs,['a','b'],['a','b'],max_iterations=5)

    def test_tiny_target_probabilities_and_offset_invariance(self):
        base=np.log([[.8,.2,.1],[.1,.7,.3],[.3,.2,.6]])
        a=new.solve_diagnostic(base,['a','b','c'],['a','b','c'])
        b=new.solve_diagnostic(base-1000,['a','b','c'],['a','b','c'])
        np.testing.assert_allclose(a['diagnostic_weights'],b['diagnostic_weights'],rtol=0,atol=1e-10)
        self.assertAlmostEqual(b['objective']-a['objective'],1000,places=10)
        self.assert_kkt(base-1000,['a','b','c'],['a','b','c'],b)

    def test_class_balance_under_repeated_family(self):
        rows=np.log([[.9,.1,.05],[.1,.9,.05]])
        a=new.solve_diagnostic(rows,['a','b'],['a','b'])
        b=new.solve_diagnostic(np.vstack([rows[0:1]]*9+[rows[1:2]]),['a']*9+['b'],['a','b'])
        np.testing.assert_allclose(a['diagnostic_weights'],b['diagnostic_weights'],rtol=0,atol=1e-12)

    def test_determinism(self):
        logs=np.log([[.8,.2,.1],[.1,.7,.3],[.3,.2,.6]])
        self.assertEqual(new.solve_diagnostic(logs,['a','b','c'],['a','b','c']),new.solve_diagnostic(logs,['a','b','c'],['a','b','c']))

    def test_fail_closed_nonfinite_and_invalid_input(self):
        for logs in ([[float('nan'),0,0]], [[1,0,0]], [[-1,-2]]):
            with self.assertRaises(ValueError):new.solve_diagnostic(logs,['a'],['a'])
        with self.assertRaises(ValueError):new.point([.3,.3,.3])
        with self.assertRaises(ValueError):new.solve_diagnostic([[-1,-2,-3]],['a'],['a'],max_iterations=10001)
        with self.assertRaises(FloatingPointError):new.mixture([1,0,0],np.array([[0.,1,1]]))

    def test_armijo_and_nonfinite_failure_are_explicit(self):
        logs=np.log([[.8,.2,.1]])
        with patch.object(new,'armijo_gap',return_value={'accepted':False}):
            with self.assertRaisesRegex(RuntimeError,'Armijo failed'):
                new.solve_diagnostic(logs,['a'],['a'])
        with patch.object(new,'armijo_gap',side_effect=FloatingPointError('nonfinite')):
            with self.assertRaises(FloatingPointError):new.solve_diagnostic(logs,['a'],['a'])

    def test_diagnostic_replay_matches_original_on_control(self):
        logs=np.log([[.8,.2,.1]])
        result=replay(logs,['a'],['a']);expected=old.solve(logs,['a'],['a'])
        self.assertEqual(result['trace'][-1]['weights'],expected['weights'])
        self.assertEqual(result['trace'][-1]['residual'],expected['KKT_projected_residual'])


if __name__=='__main__':unittest.main()
