"""Original-objective benchmarks, explicit counters and independent worker records."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,time,json,argparse,platform,hashlib,subprocess,ctypes
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit,xlogy
from scipy.linalg import solve
from threadpoolctl import threadpool_limits,threadpool_info
threadpool_limits(1)
from run_experiments import Metric

def encoded(obj):
    return json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))

class Problem:
    def __init__(self,n=80,N=600,rho=.9,reg=.1,tau=.02,group=1,seed=0):
        self.n,self.N,self.tau,self.group=n,N,tau,group
        rng=np.random.default_rng(seed);self.A=rng.normal(size=(N,n))
        for j in range(1,n):self.A[:,j]=rho*self.A[:,j-1]+np.sqrt(1-rho*rho)*self.A[:,j]
        self.A=(self.A-self.A.mean(0))/self.A.std(0)
        w=np.zeros(n);w[:10]=rng.normal(size=10)
        self.y=np.where(rng.random(N)<expit(self.A@w),1.,-1.)
        # Exact norm is data preparation, outside all solver timers.
        self.L=np.linalg.norm(self.A,2)**2/(4*N)+tau
        self.lam=reg*np.max(np.linalg.norm((self.A.T@self.y/(2*N)).reshape(-1,group),axis=1))
        self.Lh2=(n/group)*self.lam**2;self.c=Counter()
    def av(self,x):self.c['A_products']+=1;return self.A@x
    def at(self,x):self.c['AT_products']+=1;return self.A.T@x
    def f(self,x):self.c['loss_values']+=1;return np.logaddexp(0,-self.y*self.av(x)).mean()+self.tau/2*(x@x)
    def grad(self,x):self.c['gradients']+=1;return self.at(-self.y*expit(-self.y*self.av(x)))/self.N+self.tau*x
    def hess(self,x):
        self.c['dense_hessians']+=1;p=expit(-self.y*self.av(x))
        return (self.A.T*(p*(1-p)))@self.A/self.N+self.tau*np.eye(self.n)
    def h(self,x):return self.lam*np.linalg.norm(x.reshape(-1,self.group),axis=1).sum()
    def prox(self,x,t):
        self.c['prox_evaluations']+=1;v=x.reshape(-1,self.group);r=np.linalg.norm(v,axis=1,keepdims=True)
        return (v*np.maximum(1-t*self.lam/np.maximum(r,1e-300),0)).ravel()
    def dual_project(self,v):
        a=v.reshape(-1,self.group);r=np.linalg.norm(a,axis=1,keepdims=True)
        return (a*np.minimum(1,self.lam/np.maximum(r,1e-300))).ravel()
    def hsm(self,x,alpha):
        u=self.dual_project(x/alpha);return x@u-alpha/2*(u@u)
    def gh(self,x,alpha):return self.dual_project(x/alpha)
    def value(self,x,alpha=0):return self.f(x)+(self.hsm(x,alpha) if alpha else self.h(x))
    def gap(self,x,alpha=0):
        self.c['certificates']+=1;p=expit(-self.y*self.av(x));v=self.at(-self.y*p)/self.N
        u=self.dual_project(-v/(1+self.tau*alpha));ent=-(xlogy(p,p)+xlogy(1-p,1-p)).mean()
        val=self.value(x,alpha)-ent+np.linalg.norm(v+u)**2/(2*self.tau)+alpha/2*(u@u)
        if val < -1e-10:raise ArithmeticError('Negative primal-dual gap')
        return max(0.,val)
    def block_action(self,x,t,v,kind='prox',gamma=1.):
        # Proximal Jacobian, smoothed Hessian, or inverse of gamma I + Hessian.
        vector=v.ndim==1;vv=v[:,None] if vector else v
        a=x.reshape(-1,self.group);r=np.linalg.norm(a,axis=1);active=r>t*self.lam
        unit=a/np.maximum(r[:,None],1e-300)
        z=vv.reshape(self.n//self.group,self.group,vv.shape[1]);rad=unit[:,:,None]*np.sum(unit[:,:,None]*z,axis=1)[:,None,:]
        if kind=='prox':
            tan=np.where(active,1-t*self.lam/np.maximum(r,1e-300),0);rr=active.astype(float)
        else:
            ht=np.where(active,self.lam/np.maximum(r,1e-300),1/t);hr=np.where(active,0.,1/t)
            tan,rr=(1/(gamma+ht),1/(gamma+hr)) if kind=='inverse' else (ht,hr)
        ans=(tan[:,None,None]*(z-rad)+rr[:,None,None]*rad).reshape(vv.shape)
        return ans[:,0] if vector else ans

def mv(B,v,c):c['metric_products']+=1;return B.matvec(v)

def smooth_inner(p,x,g,B,alpha,eta,mode='ssn'):
    s=np.zeros(p.n);c=p.c
    def q(s):return g@s+.5*s@mv(B,s,c)+p.hsm(x+s,alpha)
    old=q(s)
    for j in range(1000):
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

def optimize(p,method,tol=1e-6,budget=60,memory=10):
    p.c=Counter();p.max_linear=0.;p.max_directional=0.;hist=[];x=np.zeros(p.n);pairs=[]
    start=time.perf_counter();lo=p.tau/10;hi=10*p.L;gamma=(p.tau+p.L)/2
    target=tol/p.Lh2;alpha=max(.01,target) if method.startswith('CONT') else target
    nonsmooth=method in ['PQN','PN'];z=x.copy();momentum=1.;stage=0;status='iteration_limit'
    for k in range(15001):
        gap=p.gap(x);elapsed=time.perf_counter()-start
        hist.append({'k':k,'time':elapsed,'gap':gap,'alpha':0 if nonsmooth or method=='R-FISTA' else alpha,'stage':stage})
        if gap<=tol:status='converged';break
        if elapsed>budget:status='time_limit';break
        if method=='R-FISTA':
            xn=p.prox(z-p.grad(z)/p.L,1/p.L);tn=(1+np.sqrt(1+4*momentum**2))/2
            if (z-xn)@(xn-x)>0:tn=1.;zn=xn.copy();p.c['momentum_restarts']+=1
            else:zn=xn+(momentum-1)/tn*(xn-x)
            x,z,momentum=xn,zn,tn;continue
        if k>=600:break
        if method.startswith('CONT') and alpha>target and p.gap(x,alpha)<=.25*alpha*p.Lh2:
            alpha=max(target,alpha*.1);stage+=1;p.c['stage_changes']+=1
            # Reuse curvature pairs: they model f, which is unchanged across stages.
        g=p.grad(x)
        if method in ['PN','MEL-PN']:B=Metric(p.n,gamma,dense=p.hess(x))
        elif method=='MEL-I':B=Metric(p.n,gamma)
        else:B=Metric(p.n,gamma,pairs)
        p.c['metric_checks']+=1;bl,bh=B.bounds()
        if bl<lo or bh>hi:B=Metric(p.n,gamma);pairs=[];p.c['metric_resets']+=1
        eta=min(lo/4,.1*np.linalg.norm(g+(p.gh(x,alpha) if not nonsmooth else p.prox(x-g,1)-x)))
        try:
            s=proximal_inner(p,x,g,B,eta) if nonsmooth else smooth_inner(p,x,g,B,alpha,eta,'gradient' if method=='MEL-GD' else 'ssn')
        except (ArithmeticError,np.linalg.LinAlgError) as err:status=str(err);break
        hv=p.h if nonsmooth else lambda w:p.hsm(w,alpha)
        delta=g@s+hv(x+s)-hv(x);old=p.f(x)+hv(x);t=1.
        if delta>=0:status='Outer direction not descent';break
        for _ in range(50):
            xn=x+t*s
            if p.f(xn)+hv(xn)<=old+1e-4*t*delta+4e-16*max(1,abs(old)):break
            t*=.5;p.c['outer_backtracks']+=1
        else:status='Outer line search failed';break
        u=xn-x;y=p.grad(xn)-g
        if u@y>1e-12*np.linalg.norm(u)*np.linalg.norm(y):
            pairs=(pairs+[(u,y)])[-memory:];gamma=float(np.clip(y@y/(u@y),lo,hi))
        else:p.c['curvature_skips']+=1
        x=xn
    elapsed=time.perf_counter()-start;counts=dict(p.c);checked=p.gap(x)
    return dict(status=status,seconds=elapsed,gap=checked,tolerance=tol,x=x,history=hist,counts=counts,
                outer_iterations=len(hist)-1,max_linear_residual=p.max_linear,max_directional_ratio=p.max_directional,dense_fallbacks=0)

def peak_memory():
    if os.name=='nt':
        class PMC(ctypes.Structure):
            _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(s,ctypes.c_size_t) for s in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        pm=PMC();pm.cb=ctypes.sizeof(pm)
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(ctypes.windll.kernel32.GetCurrentProcess()),ctypes.byref(pm),pm.cb)
        return pm.PeakWorkingSetSize
    import resource
    v=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v if sys.platform=='darwin' else v*1024

def checks():
    rng=np.random.default_rng(123)
    for group in [1,5]:
        p=Problem(n=20,N=80,group=group);p.max_linear=0.;p.max_directional=0.
        x=rng.normal(size=20);v=rng.normal(size=20);eps=1e-6
        for alpha in [.01,.3]:
            fd=(p.gh(x+eps*v,alpha)-p.gh(x-eps*v,alpha))/(2*eps)
            assert np.linalg.norm(fd-p.block_action(x,alpha,v,kind='hessian'))<1e-6
        B=Metric(20,.5,[(v,p.hess(x)@v)])
        for smooth in [False,True]:
            s=smooth_inner(p,x,p.grad(x),B,.03,1e-4) if smooth else proximal_inner(p,x,p.grad(x),B,1e-4)
            from scipy.optimize import minimize
            if smooth:
                q=lambda s:p.grad(x)@s+.5*s@B.matvec(s)+p.hsm(x+s,.03)
                rr=minimize(q,np.zeros(20),jac=lambda s:p.grad(x)+B.matvec(s)+p.gh(x+s,.03),method='BFGS',options={'gtol':1e-10})
                assert abs(q(s)-rr.fun)<1e-6
                # Auxiliary-variable elimination identity, numerically checked.
                z=p.prox(x+s,.03);M=np.linalg.inv(np.linalg.inv(B.matrix())+.03*np.eye(20));center=x-np.linalg.solve(B.matrix(),p.grad(x))
                residual=M@(z-center)+(x+s-z)/.03
                assert np.linalg.norm(residual)<1e-3
        for method in ['R-FISTA','PQN','PN','MEL','CONT']:
            r=optimize(p,method,tol=1e-6,budget=20)
            assert r['status']=='converged',(group,method,r['status'],r['gap'])
    print('PASS: block derivatives, model solution, elimination identity, two regularizers and five solvers',flush=True)

def cases():
    return [dict(id='C0',n=80,rho=.9),dict(id='U0',n=80,rho=0),
            dict(id='n400',n=400,rho=.9),dict(id='n1200',n=1200,rho=.9),dict(id='n2400',n=2400,rho=.9),
            dict(id='ill',n=400,rho=.99,tau=.002),dict(id='weak',n=400,rho=.9,reg=.01),
            dict(id='strong',n=400,rho=.9,reg=.5),dict(id='tight',n=400,rho=.9,tol=1e-8),
            dict(id='group400',n=400,rho=.9,group=5),dict(id='group1200',n=1200,rho=.9,group=5)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','pilot','suite','worker','plots']);ap.add_argument('--output',type=Path,default=Path('runs/value_review'));ap.add_argument('--job');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='check':checks();return
    if a.command=='plots':make_plots(a.output);return
    if a.command=='worker':
        job=json.loads(a.job);conf=dict(job['case']);conf.pop('id');tol=conf.pop('tol',1e-6)
        p=Problem(**conf);optimize(p,job['method'],tol=tol,budget=20,memory=job.get('memory',10))
        r=optimize(p,job['method'],tol=tol,budget=45,memory=job.get('memory',10));r.update(job);r['peak_process_bytes']=peak_memory()
        (a.output/job['file']).write_text(encoded(r),encoding='utf-8');return
    jobs=[];selected=cases() if a.command=='suite' else [cases()[0],cases()[9]]
    for case in selected:
        methods=['MEL','CONT','PQN','R-FISTA']+(['PN','MEL-PN'] if case['n']<=400 else [])
        if case['id']=='n400':methods+=['MEL-I','MEL-GD','MEL-m5','MEL-m20']
        for rep in range(3 if a.command=='suite' else 1):
            for method in methods[rep:]+methods[:rep]:
                jobs.append(dict(case=case,method='MEL' if '-m' in method else method,memory=int(method.split('-m')[1]) if '-m' in method else 10,label=method,rep=rep,file=f"{case['id']}_{method}_r{rep}.json"))
    for i,job in enumerate(jobs):
        if (a.output/job['file']).exists():continue
        print(i+1,len(jobs),job['file'],flush=True)
        try:subprocess.run([sys.executable,__file__,'worker','--output',str(a.output.resolve()),'--job',json.dumps(job)],check=True,timeout=150)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:
            (a.output/job['file']).write_text(encoded(dict(job,status=type(e).__name__)),encoding='utf-8')
    rows=[json.loads((a.output/j['file']).read_text()) for j in jobs]
    (a.output/'summary.json').write_text(encoded(rows),encoding='utf-8')
    (a.output/'protocol.json').write_text(encoded(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),threads=[{k:v for k,v in d.items() if k!='filepath'} for d in threadpool_info()],warmup='one same-setting warm-up per measured worker, 20 s budget',repetitions=3 if a.command=='suite' else 1,measurement_budget_seconds=45,process_timeout_seconds=150,peak_memory='OS peak resident working set per fresh worker, includes imports, data preparation and warm-up',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),skips='dense PN and MEL-PN excluded above n=400 by declared dense resource budget')),encoding='utf-8')
    make_plots(a.output)

def make_plots(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=json.loads((out/'summary.json').read_text());plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    colors={'MEL':'#0072B2','CONT':'#D55E00','PQN':'#009E73','R-FISTA':'#CC79A7','PN':'#777777','MEL-PN':'#000000'}
    for filename,ids in [('value_curves',['C0','n400']),('structure_curves',['ill','group400'])]:
        fig,axs=plt.subplots(1,2,figsize=(7,3.2))
        for ax,cid in zip(axs,ids):
            for method,color in colors.items():
                rr=[r for r in rows if r['case']['id']==cid and r['label']==method and 'seconds' in r]
                if not rr:continue
                r=sorted(rr,key=lambda z:z['seconds'])[len(rr)//2]
                ax.plot([v['time'] for v in r['history']],[max(v['gap'],1e-16) for v in r['history']],label=method,color=color)
                if r['status']!='converged':ax.plot(r['history'][-1]['time'],max(r['gap'],1e-16),'x',color=color)
            ax.set(xscale='symlog',yscale='log',xlabel='Total solve time (s)',title=cid);ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(left=0);ax.grid(alpha=.2)
        axs[0].set_ylabel('Original primal-dual gap');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=6,fontsize=8)
        fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/(filename+'.pdf'));fig.savefig(out/(filename+'.png'),dpi=160);plt.close(fig)
    summary=[]
    for cid in dict.fromkeys(r['case']['id'] for r in rows):
        for method in dict.fromkeys(r['label'] for r in rows if r['case']['id']==cid):
            rs=[r for r in rows if r['case']['id']==cid and r['label']==method];valid=[r for r in rs if 'seconds' in r]
            summary.append(dict(case=cid,method=method,success=sum(r['status']=='converged' for r in rs),runs=len(rs),median_seconds=float(np.median([r['seconds'] for r in valid])) if valid else None,min_seconds=min([r['seconds'] for r in valid],default=None),max_seconds=max([r['seconds'] for r in valid],default=None),median_peak_mb=float(np.median([r['peak_process_bytes']/2**20 for r in valid])) if valid else None,median_outer=float(np.median([r['outer_iterations'] for r in valid])) if valid else None,median_ssn=float(np.median([r['counts'].get('ssn_steps',0) for r in valid])) if valid else None,failures=[r['status'] for r in rs if r['status']!='converged']))
    (out/'aggregate.json').write_text(encoded(summary),encoding='utf-8')

if __name__=='__main__':main()
