"""Expanded scale campaign and a safeguarded projected continuation method.
Archived v2 experiments remain immutable. See docs/SCALE_BENCHMARKS.md.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,time,json,argparse,subprocess,hashlib,platform
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.linalg import solve
from scipy.special import expit
from threadpoolctl import threadpool_info
from value_benchmark import Problem,Metric,encoded,peak_memory,mv

def smooth_inner(p,x,g,B,alpha,eta,mode='ssn'):
    s=np.zeros(p.n);c=p.c
    def q(s):return g@s+.5*s@mv(B,s,c)+p.hsm(x+s,alpha)
    old=q(s)
    for j in range(1000):
        if time.perf_counter()>p.deadline:raise ArithmeticError("Inner time budget")
        R=g+mv(B,s,c)+p.gh(x+s,alpha);rn=np.linalg.norm(R)
        if rn<=eta*np.linalg.norm(s):return s
        if rn<1e-14:raise ArithmeticError('Relative model criterion below floating-point resolution')
        if mode=='gradient':
            # Tight current metric bound, not an artificially inflated fixed bound.
            c['metric_checks']+=1;d=-R/(B.bounds()[1]+1/alpha);c['inner_gradient_steps']+=1
        else:
            c['ssn_steps']+=1
            if B.dense is not None:
                H=p.block_action(x+s,alpha,np.eye(p.n),kind='hessian')
                d=solve(B.dense+H,-R,assume_a='pos');c['dense_solves']+=1
            else:
                z=p.block_action(x+s,alpha,-R,kind='inverse',gamma=B.gamma)
                W=p.block_action(x+s,alpha,B.U,kind='inverse',gamma=B.gamma)
                d=z-W@solve(np.diag(B.c)+B.U.T@W,B.U.T@z,assume_a='sym') if len(B.c) else z
                c['reduced_solves']+=bool(len(B.c))
            vd=mv(B,d,c)+p.block_action(x+s,alpha,d,kind='hessian');er=vd+R
            lr=np.linalg.norm(er)/rn
            directional_ratio=float(er@d/max(d@vd,1e-300))
            p.max_directional=max(p.max_directional,directional_ratio)
            p.max_linear=max(p.max_linear,lr)
            # Sufficient directional test uses the computed direction directly.
            if not np.isfinite(lr) or lr>1e-7 or directional_ratio>.5 or R@d>=0:raise ArithmeticError('Uncertified SSN direction; no dense fallback')
        slope=R@d;t=1.
        for _ in range(50):
            sn=s+t*d;qn=q(sn)
            if qn<=old+1e-4*t*slope+4e-16*max(1,abs(old)):break
            t*=.5;c['inner_backtracks']+=1
        else:raise ArithmeticError('Inner line search failed')
        s,old=sn,qn
    raise ArithmeticError('Inner iteration budget')

def proximal_inner(p,x,g,B,eta):
    """Deterministic PQN/PN baseline: proximal-residual SSN with PG safeguard."""
    w=x.copy();c=p.c;gamma=B.gamma
    c['metric_checks']+=1;upper=B.bounds()[1]
    def q(w):s=w-x;return g@s+.5*s@mv(B,s,c)+p.h(w)
    old=q(w)
    for j in range(1000):
        if time.perf_counter()>p.deadline:raise ArithmeticError("Inner time budget")
        a=g+mv(B,w-x,c);z=w-a/gamma;prox=p.prox(z,1/gamma);R=w-prox
        # Explicit subgradient certificate at prox, rather than a mapping-only test.
        res=gamma*R+mv(B,prox-w,c)
        if np.linalg.norm(res)<=eta*np.linalg.norm(prox-x):return prox-x
        c['ssn_steps']+=1
        if B.dense is None:
            W=p.block_action(z,1/gamma,B.U)/gamma
            d=-R+W@solve(np.diag(B.c)+B.U.T@W,B.U.T@R,assume_a='gen') if len(B.c) else -R
            c['reduced_solves']+=bool(len(B.c))
        else:
            P=p.block_action(z,1/gamma,np.eye(p.n));J=np.eye(p.n)-P+P@B.dense/gamma
            d=solve(J,-R,assume_a='gen');c['dense_solves']+=1
        slope=a@d+p.h(w+d)-p.h(w)
        if not np.isfinite(slope) or slope>=-1e-12*(d@d):
            d=p.prox(w-a/upper,1/upper)-w;slope=a@d+p.h(w+d)-p.h(w);c['proximal_safeguards']+=1
        t=1.
        for _ in range(50):
            wn=w+t*d;qn=q(wn)
            if qn<=old+1e-4*t*slope+4e-16*max(1,abs(old)):break
            t*=.5;c['inner_backtracks']+=1
        else:raise ArithmeticError('Proximal SSN line search failed')
        w,old=wn,qn
    raise ArithmeticError('Proximal inner iteration budget')


class HessianOperator:
    def __init__(self,p,x):
        self.p=p;v=expit(-p.y*p.av(x));self.weights=v*(1-v)/p.N
        p.c['hessian_operator_builds']+=1
    def matvec(self,v):
        self.p.c['hessian_vector_products']+=1
        return self.p.at(self.weights*self.p.av(v))+self.p.tau*v

def matrix_free_inner(p,x,g,B,eta):
    w=x.copy();v=w.copy();a=1.;upper=p.L
    for j in range(1000):
        if time.perf_counter()>p.deadline:raise ArithmeticError('Inner time budget')
        grad=g+mv(B,v-x,p.c);z=p.prox(v-grad/upper,1/upper)
        residual=upper*(v-z)+mv(B,z-v,p.c)
        p.c['inner_proximal_steps']+=1
        if np.linalg.norm(residual)<=eta*np.linalg.norm(z-x):return z-x
        an=(1+np.sqrt(1+4*a*a))/2
        if (v-z)@(z-w)>0:an=1.;vn=z.copy();p.c['inner_restarts']+=1
        else:vn=z+(a-1)/an*(z-w)
        w,v,a=z,vn,an
    raise ArithmeticError('Matrix-free inner iteration budget')

def optimize(p,method,tol=1e-6,budget=45,memory=10):
    p.c=Counter();p.max_linear=0.;p.max_directional=0.;hist=[];w=np.zeros(p.n);pairs=[]
    start=time.perf_counter();p.deadline=start+budget
    lo=p.tau/10;hi=10*p.L;gamma=(p.tau+p.L)/2
    target=tol/p.Lh2;alpha=target if method=='MEL' else max(.01,target)
    stage=0;status='iteration_limit';v=w.copy();momentum=1.;answer=w.copy()
    projected=method in ['CONT-P','P-CONT'];safe=method=='P-CONT';nonsmooth=method in ['PQN','PN-MF']
    certificate=None;stage_anchor=p.value(w,alpha) if safe else None
    for k in range(15001):
        z=p.prox(w,alpha) if projected else w
        answer=z.copy();gap=p.gap(answer)
        if safe:
            u=p.gh(w,alpha);g=p.grad(w);r=g+u
            certificate=float(np.linalg.norm(p.grad(z)+u)**2/(2*p.tau));p.c['projected_residual_certificates']+=1
        elapsed=time.perf_counter()-start
        hist.append(dict(k=k,time=elapsed,gap=gap,alpha=0 if nonsmooth or method=='R-FISTA' else alpha,stage=stage,projected_certificate=certificate))
        if gap<=tol:status='converged';break
        if elapsed>budget:status='time_limit';break
        if method=='R-FISTA':
            wn=p.prox(v-p.grad(v)/p.L,1/p.L);an=(1+np.sqrt(1+4*momentum**2))/2
            if (v-wn)@(wn-w)>0:an=1.;vn=wn.copy();p.c['momentum_restarts']+=1
            else:vn=wn+(momentum-1)/an*(wn-w)
            w,v,momentum=wn,vn,an;continue
        if k>=1200:break
        if safe and np.linalg.norm(r)<=alpha:
            alpha*=.1;stage+=1;p.c['stage_changes']+=1
            if alpha<1e-16:status='Smoothing below numerical budget';break
            # Computable fixed-anchor comparison supplies the theorem's uniform initial-gap bound.
            stage_anchor=p.value(np.zeros(p.n),alpha)
            if p.value(w,alpha)>stage_anchor:w=np.zeros(p.n);pairs=[];p.c['stage_resets']+=1
            continue
        if method in ['CONT','CONT-P'] and alpha>target and p.gap(w,alpha)<=.25*alpha*p.Lh2:
            alpha=max(target,alpha*.1);stage+=1;p.c['stage_changes']+=1
        if not safe:g=p.grad(w)
        if method=='PN-MF':B=HessianOperator(p,w)
        else:
            B=Metric(p.n,gamma,pairs);p.c['metric_checks']+=1;bl,bh=B.bounds()
            if bl<lo or bh>hi:B=Metric(p.n,gamma);pairs=[];p.c['metric_resets']+=1
        forcing=g+(p.prox(w-g,1)-w if nonsmooth else p.gh(w,alpha))
        eta=min(lo/4,.1*np.linalg.norm(forcing))
        if safe:
            ref=w-r/(p.L+1/alpha);ref_value=p.value(ref,alpha);p.c['reference_gradient_steps']+=1
            # Strict finite trial budget; failed trials cannot stall the certified reference step.
            saved=p.deadline;p.deadline=min(saved,time.perf_counter()+min(2.,max(.01,budget/10)))
        try:
            if method=='PN-MF':s=matrix_free_inner(p,w,g,B,eta)
            elif nonsmooth:s=proximal_inner(p,w,g,B,eta)
            else:s=smooth_inner(p,w,g,B,alpha,eta)
            hv=p.h if nonsmooth else lambda q:p.hsm(q,alpha)
            delta=g@s+hv(w+s)-hv(w);old=p.f(w)+hv(w);t=1.
            if delta>=0:raise ArithmeticError('Outer direction not descent')
            for bt in range(50):
                candidate=w+t*s;val=p.f(candidate)+hv(candidate)
                if val<=old+1e-4*t*delta+4e-16*max(1,abs(old)):break
                t*=.5;p.c['outer_backtracks']+=1
            else:raise ArithmeticError('Outer line search failed')
            if safe and val>ref_value:candidate=ref;p.c['reference_fallbacks']+=1
        except (ArithmeticError,np.linalg.LinAlgError) as err:
            if safe:candidate=ref;p.c['reference_fallbacks']+=1;p.c['trial_failures']+=1
            else:status=str(err);break
        finally:
            if safe:p.deadline=saved
        if safe:
            p.c['safeguard_checks']+=1
            # Record violation rather than allowing a fixed Armijo allowance to weaken this safeguard.
            if p.value(candidate,alpha)>ref_value:raise ArithmeticError('Reference decrease check failed')
        step=candidate-w;y=p.grad(candidate)-g
        if method!='PN-MF':
            if step@y>1e-12*np.linalg.norm(step)*np.linalg.norm(y):
                pairs=(pairs+[(step,y)])[-memory:];gamma=float(np.clip(y@y/(step@y),lo,hi))
            else:p.c['curvature_skips']+=1
        w=candidate
    elapsed=time.perf_counter()-start;counts=dict(p.c);gap=p.gap(answer)
    return dict(status=status,seconds=elapsed,gap=gap,tolerance=tol,x=answer,history=hist,counts=counts,outer_iterations=len(hist)-1,max_linear_residual=p.max_linear,max_directional_ratio=p.max_directional,dense_fallbacks=0,final_alpha=alpha,projected_certificate=certificate)

def cases():
    return [dict(id='s4000',n=4000),dict(id='s8000',n=8000),dict(id='s16000',n=16000),
            dict(id='seed1',n=4000,seed=1),dict(id='rho99',n=4000,rho=.99),
            dict(id='weak4000',n=4000,reg=.01),dict(id='tight4000',n=4000,tol=1e-8),
            dict(id='group8000',n=8000,group=5),dict(id='samples2000',n=4000,N=2000)]

def prepare(case,path):
    conf=dict(case);conf.pop('id');conf.pop('tol',None);p=Problem(**conf)
    np.savez(path,A=p.A,y=p.y,n=p.n,N=p.N,tau=p.tau,group=p.group,L=p.L,lam=p.lam,Lh2=p.Lh2)

def load(path):
    p=Problem.__new__(Problem)
    with np.load(path) as d:
        for key in d.files:setattr(p,key,d[key].copy() if key in ['A','y'] else d[key].item())
    p.c=Counter();return p

def checks():
    from scipy.optimize import minimize
    rng=np.random.default_rng(12)
    for group in [1,5]:
        p=Problem(n=20,N=80,group=group);x=rng.normal(size=20);q=rng.normal(size=20)
        B=HessianOperator(p,x);assert np.linalg.norm(B.matvec(q)-p.hess(x)@q)<1e-10
        p.deadline=time.perf_counter()+10;s=matrix_free_inner(p,x,p.grad(x),B,1e-4)
        for method in ['PN-MF','P-CONT','CONT-P']:
            r=optimize(p,method,budget=20);assert r['status']=='converged',(group,method,r['status'],r['gap'])
            assert p.gap(r['x'])<=1e-6
    # Non-Lipschitz regularizer: quadratic h and half-line indicator, exact smoothed stages.
    for kind in ['quadratic','indicator']:
        for alpha in [.1,.01,.001]:
            mu=2.;a=-1. if kind=='indicator' else 1.;beta=3.
            if kind=='quadratic':w=mu*a/(mu+beta/(1+alpha*beta));z=w/(1+alpha*beta);star=mu*a/(mu+beta);h=lambda x:beta*x*x/2
            else:w=mu*a/(mu+1/alpha);z=max(w,0);star=0.;h=lambda x:0.
            u=(w-z)/alpha;err=(mu*(z-a)+u)**2/(2*mu)
            gap=mu/2*(z-a)**2+h(z)-(mu/2*(star-a)**2+h(star))
            assert gap<=err+1e-12
    print('PASS: Hessian operator, true model residual, projected methods, non-Lipschitz certificates',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','pilot','suite','worker']);ap.add_argument('--output',type=Path,default=Path('runs/scale_review'));ap.add_argument('--job');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='check':checks();return
    if a.command=='worker':
        job=json.loads(a.job);p=load(a.output/'data'/f"{job['case']['id']}.npz");tol=job['case'].get('tol',1e-6)
        optimize(p,job['method'],tol,budget=2)
        r=optimize(p,job['method'],tol,budget=45);r.update(job);r['peak_process_bytes']=peak_memory()
        (a.output/job['file']).write_text(encoded(r),encoding='utf-8',newline='\n');return
    selected=cases() if a.command=='suite' else [dict(id='pilot_l1',n=200),dict(id='pilot_group',n=200,group=5)]
    (a.output/'data').mkdir(exist_ok=True);jobs=[]
    for case in selected:
        path=a.output/'data'/f"{case['id']}.npz"
        if not path.exists():prepare(case,path)
        methods=['CONT','CONT-P','P-CONT','PQN','R-FISTA','PN-MF']
        if case['id']=='s4000' or a.command=='pilot':methods+=['MEL']
        for rep in range(3 if a.command=='suite' else 1):
            for method in methods[rep:]+methods[:rep]:jobs.append(dict(case=case,method=method,label=method,rep=rep,file=f"{case['id']}_{method}_r{rep}.json"))
    protocol=dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=[{k:v for k,v in d.items() if k!='filepath'} for d in threadpool_info()],warmup_seconds=2,measurement_budget_seconds=45,inner_iteration_cap=1000,outer_iteration_cap=1200,repetitions=3 if a.command=='suite' else 1,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.output/'data').glob('*.npz')},jobs=jobs)
    (a.output/'protocol.json').write_text(encoded(protocol),encoding='utf-8',newline='\n')
    for i,job in enumerate(jobs):
        if (a.output/job['file']).exists():continue
        print(i+1,len(jobs),job['file'],flush=True)
        try:subprocess.run([sys.executable,__file__,'worker','--output',str(a.output.resolve()),'--job',json.dumps(job)],check=True,timeout=120)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:(a.output/job['file']).write_text(encoded(dict(job,status=type(e).__name__)),encoding='utf-8',newline='\n')
    rows=[json.loads((a.output/j['file']).read_text()) for j in jobs]
    (a.output/'summary.json').write_text(encoded(rows),encoding='utf-8',newline='\n')
if __name__=='__main__':main()
