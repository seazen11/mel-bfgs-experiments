"""Nested MEL-BFGS and the supporting numerical experiments."""
import sys, os
from pathlib import Path
ROOT=Path(os.environ.get('MEL_OUTPUT_DIR', Path(__file__).resolve().parents[1]/'runs'/'latest')).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib,platform,urllib.request,argparse
import numpy as np
import scipy
from scipy.special import expit,xlogy
from scipy.linalg import solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits,threadpool_info
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
threadpool_limits(1)
OUT=ROOT/'results'; OUT.mkdir(parents=True,exist_ok=True)
def dump(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)),encoding='utf-8')

class Logistic:
    def __init__(self,A,y,name,tau=.02):
        self.A,self.y,self.name,self.tau=A,y,name,tau
        self.N,self.n=A.shape
        self.lam=.1*np.max(np.abs(A.T@y/(2*self.N)))
        self.L=np.linalg.norm(A,2)**2/(4*self.N)+tau
    def f(self,x):
        return np.logaddexp(0,-self.y*(self.A@x)).mean()+self.tau/2*(x@x)
    def grad(self,x):
        p=expit(-self.y*(self.A@x))
        return self.A.T@(-self.y*p)/self.N+self.tau*x
    def hess(self,x):
        p=expit(-self.y*(self.A@x))
        return (self.A.T*(p*(1-p)))@self.A/self.N+self.tau*np.eye(self.n)
    def h(self,x,alpha):
        if alpha==0:return self.lam*np.abs(x).sum()
        u=np.clip(x/alpha,-self.lam,self.lam)
        return x@u-alpha/2*(u@u)
    def gh(self,x,alpha):return np.clip(x/alpha,-self.lam,self.lam)
    def diag(self,x,alpha):return (np.abs(x)<alpha*self.lam).astype(float)/alpha
    def value(self,x,alpha):return self.f(x)+self.h(x,alpha)
    def gap(self,x,alpha=0):
        p=expit(-self.y*(self.A@x)); v=self.A.T@(-self.y*p)/self.N
        u=np.clip(-v/(1+self.tau*alpha),-self.lam,self.lam)
        entropy=-(xlogy(p,p)+xlogy(1-p,1-p)).mean()
        dual=entropy-np.linalg.norm(v+u)**2/(2*self.tau)-alpha/2*(u@u)
        gap=self.value(x,alpha)-dual
        if gap < -1e-10:raise RuntimeError('Invalid dual certificate')
        return max(0.,gap)

def datasets():
    ps=[]
    for corr in [0.,.9]:
        for seed in [0,1,2]:
            rng=np.random.default_rng(seed); A=rng.normal(size=(600,80))
            for j in range(1,80):A[:,j]=corr*A[:,j-1]+np.sqrt(1-corr*corr)*A[:,j]
            A=(A-A.mean(0))/A.std(0)
            w=np.zeros(80); w[:10]=rng.normal(size=10)
            y=np.where(rng.random(600)<expit(A@w),1.,-1.)
            ps.append(Logistic(A,y,f'synthetic_rho{corr}_seed{seed}'))
    data=ROOT/'data'/'wdbc.data'; data.parent.mkdir(exist_ok=True)
    if not data.exists():urllib.request.urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',data)
    if hashlib.sha256(data.read_bytes()).hexdigest() != 'd606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a':
        raise ValueError('WDBC checksum mismatch; see docs/DATA.md')
    raw=np.loadtxt(data,delimiter=',',dtype=str)
    assert raw.shape==(569,32)
    A=raw[:,2:].astype(float); A=(A-A.mean(0))/A.std(0)
    ps.append(Logistic(A,np.where(raw[:,1]=='M',1.,-1.),'wdbc'))
    dump('manifest',{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),'blas':threadpool_info(),
        'source':'https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic','wdbc_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'preprocessing':'All rows standardized; no intercept. Optimization benchmark, no prediction claims.',
        'problems':[{'name':p.name,'N':p.N,'n':p.n,'tau':p.tau,'lambda':p.lam,'Lf':p.L} for p in ps]})
    return ps

class Metric:
    def __init__(self,n,gamma,pairs=(),dense=None):
        self.n,self.gamma,self.dense=n,gamma,dense; U=[]; signs=[]
        for u,y in pairs:
            bu=gamma*u
            if U:
                W=np.column_stack(U); bu+=W@(np.array(signs)*(W.T@u))
            U.extend([bu/np.sqrt(u@bu),y/np.sqrt(u@y)]); signs.extend([-1.,1.])
        self.U=np.column_stack(U) if U else np.empty((n,0)); self.c=np.array(signs)
    def matvec(self,v):
        return self.dense@v if self.dense is not None else self.gamma*v+self.U@(self.c*(self.U.T@v))
    def matrix(self):
        return self.dense if self.dense is not None else self.gamma*np.eye(self.n)+(self.U*self.c)@self.U.T
    def bounds(self):
        if self.dense is not None:return np.linalg.eigvalsh(self.dense)[[0,-1]]
        if not len(self.c):return [self.gamma,self.gamma]
        _,r=np.linalg.qr(self.U,mode='reduced'); ev=np.linalg.eigvalsh((r*self.c)@r.T)+self.gamma
        return min(self.gamma,ev[0]),max(self.gamma,ev[-1])
    def linear(self,diag,rhs):
        if self.dense is not None:return solve(self.dense+np.diag(diag),rhs,assume_a='pos')
        ainv=1/(self.gamma+diag); z=ainv*rhs
        if not len(self.c):return z
        W=ainv[:,None]*self.U; K=np.diag(self.c)+self.U.T@W
        return z-W@solve(K,self.U.T@z,assume_a='sym')

def inner(p,x,B,alpha,eta,mode='ssn',force=False):
    gf=p.grad(x); s=np.zeros(p.n); hist=[]
    def q(v):return gf@v+.5*v@B.matvec(v)+p.h(x+v,alpha)
    q0=q(s)
    for j in range(2000):
        R=gf+B.matvec(s)+p.gh(x+s,alpha); rn=np.linalg.norm(R)
        hist.append({'j':j,'residual':rn,'q':q0,'s':s.copy()})
        if (not force and rn<=eta*np.linalg.norm(s)) or rn<2e-13:return s,hist
        diag=p.diag(x+s,alpha)
        d=-R/(10*p.L+1/alpha) if mode=='gradient' else B.linear(diag,-R)
        lr=np.linalg.norm(B.matvec(d)+diag*d+R)/max(rn,1e-300)
        if mode=='ssn' and lr>1e-8:
            d=solve(B.matrix()+np.diag(diag),-R,assume_a='pos'); lr=np.linalg.norm(B.matvec(d)+diag*d+R)/rn
        if mode=='ssn' and lr>1e-7:raise RuntimeError('Uncertified linear solve')
        slope=R@d; step=1.
        if slope>=0:raise RuntimeError('Inner direction not descent')
        for bt in range(60):
            sn=s+step*d; qn=q(sn)
            if qn<=q0+1e-4*step*slope+4e-16*max(1,abs(q0)):break
            step*=.5
        else:raise RuntimeError('Inner line search failed')
        hist[-1].update(step=step,linear_residual=lr); s,q0=sn,qn
    dump('inner_failure',{'history':hist,'x':x,'s':s,'B':B.matrix(),'alpha':alpha,'eta':eta,'problem':p.name})
    raise RuntimeError('Inner iteration limit')

def nested(p,alpha,tol=1e-7,x0=None,variant='lbfgs',memory=10,maxit=600,original=False):
    x=np.zeros(p.n) if x0 is None else x0.copy(); pairs=[]; gamma=(p.tau+p.L)/2
    dense=gamma*np.eye(p.n); hist=[]; start=time.perf_counter(); inner_total=0; resets=0
    lower=p.tau/10; upper=10*p.L; eta_bar=lower/4
    for k in range(maxit+1):
        gf=p.grad(x); g=gf+p.gh(x,alpha); cert=p.gap(x,0 if original else alpha)
        row={'k':k,'time':time.perf_counter()-start,'gap':cert,'original_gap':p.gap(x),'gradient':np.linalg.norm(g),'value':p.value(x,alpha),'x':x.copy(),'inner_total':inner_total}
        hist.append(row)
        if cert<=tol:return {'status':'converged','history':hist,'x':x,'resets':resets,'alpha':alpha}
        if k==maxit:break
        if variant=='newton':B=Metric(p.n,gamma,dense=p.hess(x))
        elif variant=='full':B=Metric(p.n,gamma,dense=dense)
        else:B=Metric(p.n,gamma,pairs)
        lo,hi=B.bounds()
        if lo<lower or hi>upper:B=Metric(p.n,gamma); pairs=[]; dense=gamma*np.eye(p.n); resets+=1
        eta=min(eta_bar,.1*np.linalg.norm(g))
        try:
            s,ih=inner(p,x,B,alpha,eta,mode='gradient' if variant=='inner_gradient' else 'ssn')
        except RuntimeError as exc:
            row['time']=time.perf_counter()-start
            return {'status':'inner_failed','reason':str(exc),'history':hist,'x':x,'resets':resets,'alpha':alpha}
        inner_total+=len(ih)-1
        delta=gf@s+p.h(x+s,alpha)-p.h(x,alpha); step=1.; old=p.value(x,alpha)
        for bt in range(60):
            xn=x+step*s
            if p.value(xn,alpha)<=old+1e-4*step*delta+4e-16*max(1,abs(old)):break
            step*=.5
        else:raise RuntimeError('Outer line search failed')
        row.update(outer_step=step,inner=ih,relative_model_residual=ih[-1]['residual']/max(np.linalg.norm(s),1e-300),eta=eta,Bs=B.matvec(s),s=s,metric_bounds=[lo,hi])
        u=xn-x; y=p.grad(xn)-gf
        if u@y>1e-14*np.linalg.norm(u)*np.linalg.norm(y):
            if variant=='full':
                bu=B.matvec(u); dense=B.matrix()-np.outer(bu,bu)/(u@bu)+np.outer(y,y)/(u@y)
            pairs=(pairs+[(u,y)])[-memory:]; gamma=float(np.clip(y@y/(u@y),lower,upper))
        x=xn
    return {'status':'iteration_limit','history':hist,'x':x,'resets':resets,'alpha':alpha}

def baseline(p,alpha,tol=1e-7,method='direct_lbfgs',original=False,maxit=15000):
    start=time.perf_counter(); hist=[]
    def record(x):
        gap=p.gap(x,0 if original else alpha)
        hist.append({'k':len(hist),'time':time.perf_counter()-start,'gap':gap,'original_gap':p.gap(x),'x':x.copy()})
        return gap
    x=np.zeros(p.n); record(x)
    if method=='fista':
        z=x.copy(); t=1.
        for k in range(maxit):
            v=z-p.grad(z)/p.L; xn=np.sign(v)*np.maximum(np.abs(v)-p.lam/p.L,0)
            tn=(1+np.sqrt(1+4*t*t))/2; z=xn+(t-1)/tn*(xn-x); x=xn; t=tn
            if record(x)<=tol:break
    else:
        class Stop(Exception):pass
        def cb(v):
            if record(v)<=tol:raise Stop()
        try:
            res=minimize(lambda v:(p.value(v,alpha),p.grad(v)+p.gh(v,alpha)),x,jac=True,method='L-BFGS-B',callback=cb,
                         options={'maxiter':maxit,'maxls':50,'ftol':0.,'gtol':1e-13,'maxcor':10})
            x=res.x; record(x)
        except Stop:x=hist[-1]['x']
    return {'status':'converged' if hist[-1]['gap']<=tol else 'not_certified','history':hist,'x':x,'alpha':alpha}

def check(p):
    rng=np.random.default_rng(45); x=rng.normal(size=p.n)*.001; v=rng.normal(size=p.n); alpha=.03; eps=1e-7
    fd=(p.value(x+eps*v,alpha)-p.value(x-eps*v,alpha))/(2*eps); grad=(p.grad(x)+p.gh(x,alpha))@v
    H=p.hess(x); pairs=[]
    for _ in range(5):
        u=rng.normal(size=p.n); pairs.append((u,H@u))
    B=Metric(p.n,1.,pairs); D=p.diag(x,alpha); rhs=rng.normal(size=p.n)
    d=B.linear(D,rhs); dd=solve(B.matrix()+np.diag(D),rhs,assume_a='pos'); err=np.linalg.norm(d-dd)/np.linalg.norm(dd)
    assert abs(fd-grad)<1e-6 and err<1e-10
    dump('checks',{'gradient_error':abs(fd-grad),'woodbury_error':err,'original_gap':p.gap(x),'smoothed_gap':p.gap(x,alpha)})

def compact_benchmark():
    rng=np.random.default_rng(8); rows=[]
    for n in [200,1000,2500]:
        for memory in [5,10,20]:
            hd=np.geomspace(.1,3,n); pairs=[]
            for _ in range(memory):
                u=rng.normal(size=n); pairs.append((u,hd*u))
            B=Metric(n,1.,pairs); diag=rng.choice([0.,100.],size=n); rhs=rng.normal(size=n); ts=[]; td=[]
            for rep in range(6):
                start=time.perf_counter(); d=B.linear(diag,rhs); tc=time.perf_counter()-start
                start=time.perf_counter(); dd=solve(B.matrix()+np.diag(diag),rhs,assume_a='pos'); dt=time.perf_counter()-start
                if rep:ts.append(tc); td.append(dt)
            rows.append({'n':n,'memory':memory,'compact_seconds':np.median(ts),'dense_seconds':np.median(td),'relative_error':np.linalg.norm(d-dd)/np.linalg.norm(dd),
                         'stored_metric_bytes':B.U.nbytes+B.c.nbytes,'dense_metric_bytes':n*n*8})
    dump('compact',rows)

def analytic_bias():
    lam=.2; a=np.linspace(.3,2,100); c=np.r_[np.linspace(-.15,.15,50),np.linspace(.3,1,50)]
    xs=np.sign(c)*np.maximum(np.abs(c)-lam,0)/a
    def F(x):return .5*np.sum(a*x*x)-c@x+lam*np.abs(x).sum()
    rows=[]
    for alpha in np.logspace(-1,-7,7):
        x=np.where(np.abs(c)<=lam*(1+alpha*a),c/(a+1/alpha),np.sign(c)*(np.abs(c)-lam)/a)
        gap=F(x)-F(xs); dist=np.linalg.norm(x-xs); bound=alpha*len(c)*lam*lam/2
        assert gap<=bound+1e-13 and dist<=np.sqrt(2*bound/min(a))+1e-13
        rows.append({'alpha':alpha,'original_gap':gap,'distance':dist,'gap_bound':bound,'distance_bound':np.sqrt(2*bound/min(a))})
    dump('bias',rows)

def run_suite():
    ps=datasets(); check(ps[0]); summary=[]
    for p in ps:
        print('Problem',p.name,flush=True); alpha=.01
        ref=baseline(p,alpha,1e-14)
        # Reference refinement is outside all solver timings and uses exact curvature.
        xr=ref['x'].copy()
        for _ in range(8):
            gr=p.grad(xr)+p.gh(xr,alpha)
            if np.linalg.norm(gr)<1e-13:break
            xr+=solve(p.hess(xr)+np.diag(p.diag(xr,alpha)),-gr,assume_a='pos')
        ref['x']=xr; ref['gradient_norm']=np.linalg.norm(p.grad(xr)+p.gh(xr,alpha))
        ref['distance_certificate']=ref['gradient_norm']/p.tau
        dump(p.name+'_reference',ref); Hstar=p.hess(xr)
        for method in ['lbfgs','full','newton','direct_lbfgs']:
            r=nested(p,alpha,tol=1e-13,variant=method) if method!='direct_lbfgs' else baseline(p,alpha,1e-13)
            for row in r['history']:
                row['reference_error']=np.linalg.norm(row['x']-ref['x'])
                if 's' in row:row['dm']=np.linalg.norm(row['Bs']-Hstar@row['s'])/np.linalg.norm(row['s'])
            dump(p.name+'_fixed_'+method,r)
            summary.append({'problem':p.name,'test':'fixed','method':method,'status':r['status'],'seconds':r['history'][-1]['time'],'iterations':len(r['history'])-1,'gap':r['history'][-1]['gap']})
            print(method,r['status'],len(r['history'])-1,round(r['history'][-1]['time'],3),flush=True)
        eps=1e-6; af=eps/(p.n*p.lam*p.lam)
        for method in ['lbfgs','direct_lbfgs','fista','continuation']:
            if method=='lbfgs':r=nested(p,af,tol=eps,original=True)
            elif method=='continuation':
                start=time.perf_counter(); x=np.zeros(p.n); stages=[]; allhist=[]; alphas=[]; aa=max(.1,af)
                while aa>af*(1+1e-12):alphas.append(aa); aa*=.1
                alphas.append(af)
                for aa in alphas:
                    offset=time.perf_counter()-start; rr=nested(p,aa,tol=.25*aa*p.n*p.lam*p.lam,x0=x)
                    stages.append({'alpha':aa,'status':rr['status'],'smooth_gap':p.gap(rr['x'],aa),'original_gap':p.gap(rr['x'])})
                    for row in rr['history']:row['time']+=offset; row['gap']=row['original_gap']; allhist.append(row)
                    x=rr['x']
                r={'history':allhist,'x':x,'stages':stages,'status':'converged' if p.gap(x)<=eps else 'not_certified'}
            else:r=baseline(p,af,eps,method,original=True)
            dump(p.name+'_original_'+method,r)
            summary.append({'problem':p.name,'test':'original','method':method,'status':r['status'],'seconds':r['history'][-1]['time'],'iterations':len(r['history'])-1,'gap':r['history'][-1]['gap']})
        rng=np.random.default_rng(77); pairs=[]; H=p.hess(np.zeros(p.n))
        for _ in range(10):
            u=rng.normal(size=p.n); pairs.append((u,H@u))
        B=Metric(p.n,(p.tau+p.L)/2,pairs)
        s,ih=inner(p,np.zeros(p.n),B,.01,1e-15,force=True)
        for row in ih:row['error_to_final']=np.linalg.norm(row['s']-s)
        dump(p.name+'_inner',{'history':ih,'solution':s,'reference_residual':ih[-1]['residual'],'lower_metric_bound':B.bounds()[0]})
        dump('summary',summary)
    print('Compact solve benchmark',flush=True); compact_benchmark(); analytic_bias()
    try:dump('inner_gradient_ablation',nested(ps[0],.1,tol=1e-6,variant='inner_gradient',maxit=100))
    except RuntimeError as e:dump('inner_gradient_ablation',{'status':'failed','reason':str(e)})
    make_plots(); print('All experiments completed',flush=True)

def make_plots():
    fig,axs=plt.subplots(1,3,figsize=(14,4)); p='synthetic_rho0.9_seed0'
    for method in ['lbfgs','full','newton','direct_lbfgs']:
        r=json.loads((OUT/(p+'_fixed_'+method+'.json')).read_text())['history']
        axs[0].semilogy([a['k'] for a in r],[max(a['gap'],1e-16) for a in r],label=method)
        if method!='direct_lbfgs':axs[1].semilogy([a['k'] for a in r[:-1]],[max(a['dm'],1e-16) for a in r[:-1]],label=method)
    for method in ['lbfgs','direct_lbfgs','fista','continuation']:
        r=json.loads((OUT/('wdbc_original_'+method+'.json')).read_text())['history']
        axs[2].semilogy([a['time'] for a in r],[max(a['gap'],1e-16) for a in r],label=method)
    for ax,title,xlab in zip(axs,['Fixed smoothing: certified gap','Directional consistency diagnostic','WDBC: original certified gap'],['Outer iteration','Outer iteration','Seconds']):
        ax.set_title(title); ax.set_xlabel(xlab); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'convergence.png',dpi=180); plt.close(fig)
    rows=json.loads((OUT/'bias.json').read_text()); fig,axs=plt.subplots(1,2,figsize=(9,3.6))
    for key in ['original_gap','gap_bound']:axs[0].loglog([r['alpha'] for r in rows],[r[key] for r in rows],'-o',label=key)
    rows=json.loads((OUT/'compact.json').read_text())
    for mem in [5,10,20]:
        rs=[r for r in rows if r['memory']==mem]; axs[1].plot([r['n'] for r in rs],[r['dense_seconds']/r['compact_seconds'] for r in rs],'-o',label=f'memory={mem}')
    axs[0].set_xlabel('alpha'); axs[0].set_title('Analytic smoothing bias'); axs[1].set_xlabel('Dimension'); axs[1].set_ylabel('Dense / compact solve time')
    for ax in axs:ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'bias_and_cost.png',dpi=180); plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    if args.smoke:
        ps=datasets(); check(ps[0]); r=nested(ps[0],.01,tol=1e-8); print(r['status'],len(r['history']),r['history'][-1]['gap']); dump('smoke',r)
    else:run_suite()
