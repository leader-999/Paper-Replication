"""Numerical and behavioral checks against independent calculations."""
import unittest,json
from pathlib import Path
from dataclasses import replace
import numpy as np,numba
from scipy.optimize import minimize
from model import Config,make_environment,benchmarks,demand
from learning import train_batch
from analyze import cycle,paper_losses,discounted_value
ROOT=Path(__file__).resolve().parent

class Checks(unittest.TestCase):
    def test_static_nash_and_joint_monopoly(self):
        pn,pm,rn,rm,q=benchmarks((1,1),(2,2),.25)
        self.assertAlmostEqual(pn[0],1.47292666,places=7)
        self.assertAlmostEqual(pm[0],1.92498092,places=7)
        for i in range(2):
            fit=minimize(lambda x:-(x[0]-1)*demand(np.array([x[0],pn[1-i]]),(2,2),.25)[0],[pn[i]])
            self.assertAlmostEqual(fit.x[0],pn[i],places=5)
        fit=minimize(lambda p:-np.dot(p-1,demand(p,(2,2),.25)),[1.8,1.8])
        np.testing.assert_allclose(fit.x,pm,atol=1e-5)

    def test_cost_asymmetry_shares(self):
        expected=[.500,.545,.588,.627,.662,.722]
        for c,target in zip([1,.875,.75,.625,.5,.25],expected):
            q=benchmarks((1,c),(2,2),.25)[-1]
            self.assertAlmostEqual(q[1]/q.sum(),target,delta=.00051)

    def test_homogeneous_ties_and_outside_option(self):
        np.testing.assert_equal(demand([1.5,1.5],[2,2],0),[.5,.5])
        np.testing.assert_equal(demand([1.4,1.5],[2,2],0),[1,0])
        np.testing.assert_equal(demand([2.1,2.2],[2,2],0),[0,0])

    def test_bellman_linear_system_vs_finite_discounted_sum(self):
        e=make_environment(Config(m=3));p=np.array([[0,1,2,2,0,1,1,2,0],[1,2,0,0,1,2,2,0,1]],dtype=np.int16);s=4;delta=.95
        exact=discounted_value(p,s,e['rewards'][0],delta,3);total=0
        for t in range(1000):
            j=int(p[0,s]*3+p[1,s]);total+=delta**t*e['rewards'][0,j,0];s=j
        self.assertAlmostEqual(exact,total,places=10)

    def test_one_state_myopic_best_response(self):
        e=make_environment(Config(m=5,memory=0,delta=0));r=e['rewards'][0];p=np.array([[[2],[2]]],dtype=np.int16)
        on,off,eq=paper_losses(p,np.array([0]),r,5,0,0)
        q0=r[np.arange(5)*5+2,0];q1=r[2*5+np.arange(5),1]
        expected=((q0.max()-q0[2])/q0.max()+(q1.max()-q1[2])/q1.max())/2
        self.assertAlmostEqual(on[0],expected);self.assertAlmostEqual(off[0],expected)

    def test_compiled_kernel_matches_plain_python_updates(self):
        c=Config(m=3,exploration=False,init='optimistic',stable_periods=500,max_steps=300,history_end=0,history_every=0);e=make_environment(c);seed=73
        got=train_batch(e['q_initial'],e['rewards'],3,3,1,c.alpha,c.beta,c.delta,500,300,np.array([seed],np.uint64),True,0,0,False,0.)
        # Independent reference loop: full np.argmax each step, no cached actions.
        mask=(1<<64)-1;rng=(seed+0x9E3779B97F4A7C15)&mask
        def uniform():
            nonlocal rng
            rng=(rng^(rng<<13))&mask;rng^=rng>>7;rng=(rng^(rng<<17))&mask
            return (rng>>11)/9007199254740992.
        s=int(uniform()*9);q=e['q_initial'].copy()
        for t in range(300):
            actions=np.argmax(q[:,s],axis=1)
            for i in range(2):uniform()
            nxt=int(actions[0]*3+actions[1])
            for i,a in enumerate(actions):q[i,s,a]=(1-c.alpha)*q[i,s,a]+c.alpha*(e['rewards'][0,nxt,i]+c.delta*q[i,nxt].max())
            s=nxt
        np.testing.assert_allclose(got[5][0],q,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(got[0][0],np.argmax(q,axis=2));self.assertEqual(got[1][0],s)

    def test_saved_baseline_policy_is_argmax_q(self):
        paths=list((ROOT/'results/runs').glob('baseline_*/batch_00000.npz'))
        if not paths:self.skipTest('Run baseline first')
        d=np.load(paths[0]);np.testing.assert_array_equal(d['policies'],np.argmax(d['qs'],axis=-1))
        self.assertTrue(d['converged'].all())

if __name__=='__main__':
    numba.set_num_threads(2);suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks);result=unittest.TextTestRunner(verbosity=2).run(suite)
    (ROOT/'results/validation.json').write_text(json.dumps(dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),success=result.wasSuccessful()),indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
