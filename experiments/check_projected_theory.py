"""Numerical checks of the projected-stage proof; not a substitute for proof."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.linalg import solve
from scale_benchmark import Problem,optimize
import scale_benchmark as module

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('runs/projected_theory'));a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    H=np.array([[2.,1.],[1.,2.]]);b=np.array([-1.,1.]);star=np.array([0.,.5]);rows=[]
    for alpha in [1.,.1,.01,.001,.0001,.00001]:
        w=solve(H+np.diag([1/alpha,0]),b,assume_a='pos');z=w.copy();z[0]=max(0,z[0]);u=(w-z)/alpha
        residual=np.linalg.norm(H@w-b+u);certificate=np.linalg.norm(H@z-b+u)**2/2
        # Stable direct restricted objective difference and exact analytic expression.
        error=(z[1]-.5)**2;exact=9*alpha**2/(4*(2+3*alpha)**2)
        assert abs(error-exact)<1e-13 and error<=certificate+1e-13
        assert residual<1e-12 and np.linalg.norm(u)<=1.5+1e-12
        rows.append(dict(alpha=alpha,w=w.tolist(),z=z.tolist(),gap=error,analytic_gap=exact,certificate=certificate,gap_over_alpha_squared=error/alpha**2,smoothed_gradient_norm=residual))
    # Deliberately fail every Newton trial: the reference step must still progress.
    old=module.smooth_inner
    def failed(*args,**kwargs):raise ArithmeticError('Deliberate trial failure for safeguard test')
    module.smooth_inner=failed
    p=Problem(n=20,N=80)
    r=optimize(p,'P-CONT',tol=1e-4,budget=5)
    module.smooth_inner=old
    assert r['counts'].get('reference_fallbacks',0)>0
    assert r['counts']['reference_fallbacks']==r['counts']['reference_gradient_steps']
    initial=r['history'][0]['gap'];assert r['gap']<initial
    data=dict(sharp_halfspace_example=rows,forced_failure_safeguard=dict(status=r['status'],initial_gap=initial,final_gap=r['gap'],updates=r['outer_iterations'],counts=r['counts']),note='Analytic example uses exact half-space proximal map; time/iteration-capped safeguard test reports its actual status.')
    (a.output/'theory_checks.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS: sharp half-space error, local dual bound, projected certificate and forced trial-failure safeguard')
if __name__=='__main__':main()
