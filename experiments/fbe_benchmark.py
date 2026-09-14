"""Independent implementation of Stella--Themelis--Patrinos Algorithm 2.
Fixed gamma=.95/L, envelope-gradient L-BFGS, pairs at x and line-search w,
then return the forward-backward point. Not the authors' software.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,time,json,argparse,subprocess,hashlib,platform
from pathlib import Path
from collections import Counter
import numpy as np
from scale_benchmark import load,prepare,encoded,peak_memory,HessianOperator
from guard_benchmark import run as guard_run

def envelope(p,x,gamma,gradient=False):
    g=p.grad(x);z=p.prox(x-gamma*g,gamma);s=z-x
    value=p.f(x)+p.h(z)+g@s+(s@s)/(2*gamma);p.c['envelope_values']+=1
    if not gradient:return value,z,None
    residual=-s/gamma;grad=residual-gamma*HessianOperator(p,x).matvec(residual)
    p.c['envelope_gradients']+=1
    return value,z,grad

def optimize(p,tol=1e-6,budget=45,memory=10):
    p.c=Counter();p.max_linear=0.;p.max_directional=0.;x=np.zeros(p.n);pairs=[];hist=[];gamma=.95/p.L
    start=time.perf_counter();deadline=start+budget;status='iteration_limit'
    for k in range(1201):
        gap=p.gap(x);elapsed=time.perf_counter()-start;hist.append(dict(k=k,time=elapsed,gap=gap,alpha=0,stage=0))
        if gap<=tol:status='converged';break
        if elapsed>budget:status='time_limit';break
        if k==1200:break
        old,z,g=envelope(p,x,gamma,True);q=g.copy();aa=[]
        for s,y in reversed(pairs):a=float(s@q/(s@y));aa.append(a);q-=a*y
        scale=float(pairs[-1][0]@pairs[-1][1]/(pairs[-1][1]@pairs[-1][1])) if pairs else 1.
        d=scale*q
        for (s,y),a in zip(pairs,reversed(aa)):d+=s*(a-y@d/(s@y))
        d=-d;p.c['lbfgs_two_loop']+=1
        if not np.all(np.isfinite(d)) or g@d>=0:d=-g;pairs=[];p.c['direction_resets']+=1
        t=1.;w=x.copy();zn=z.copy();accepted=False
        for bt in range(50):
            if time.perf_counter()>deadline:break
            w=x+t*d;val,zn,_=envelope(p,w,gamma)
            if val<=old:accepted=True;break
            t*=.5;p.c['outer_backtracks']+=1
        if accepted:
            _,_,gw=envelope(p,w,gamma,True);s=w-x;y=gw-g
            if s@y>1e-12*np.linalg.norm(s)*np.linalg.norm(y):pairs=(pairs+[(s,y)])[-memory:]
            else:p.c['curvature_skips']+=1
        else:zn=z;p.c['forward_backward_fallbacks']+=1
        x=zn
    seconds=time.perf_counter()-start;counts=dict(p.c);gap=p.gap(x)
    return dict(status=status,seconds=seconds,gap=gap,tolerance=tol,x=x,history=hist,counts=counts,outer_iterations=len(hist)-1,max_linear_residual=0.,max_directional_ratio=0.,dense_fallbacks=0,final_alpha=0.)

def run(p,m,tol,budget):return optimize(p,tol,budget) if m=='FBE-LBFGS' else guard_run(p,m,tol,budget)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['check','pilot','suite','worker']);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data-dir',type=Path,required=True);ap.add_argument('--job');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);a.data_dir.mkdir(parents=True,exist_ok=True)
    if a.command=='check':
        from value_benchmark import Problem
        records=[]
        for group in [1,5]:
            p=Problem(n=20,N=80,group=group);rng=np.random.default_rng(15);x=rng.normal(size=20);d=rng.normal(size=20);gamma=.95/p.L
            val,z,g=envelope(p,x,gamma,True);e=1e-6;fd=(envelope(p,x+e*d,gamma)[0]-envelope(p,x-e*d,gamma)[0])/(2*e)
            assert abs(fd-g@d)<1e-7
            assert p.value(z)<=val+1e-12 and val<=p.value(x)+1e-12
            r=optimize(p,budget=10);assert r['status']=='converged' and p.gap(r['x'])<=1e-6
            records.append(dict(group=group,gradient_error=abs(float(fd-g@d)),gap=r['gap'],status=r['status']))
        (a.output/'checks.json').write_text(encoded(records),encoding='utf-8',newline='\n');print('PASS FBE gradient, envelope inequalities, two converged solves');return
    if a.command=='worker':
        j=json.loads(a.job);p=load(a.data_dir/(j['case']['id']+'.npz'));tol=j['case'].get('tol',1e-6);run(p,j['method'],tol,2);r=run(p,j['method'],tol,45);r.update(j);r['peak_process_bytes']=peak_memory();(a.output/j['file']).write_text(encoded(r),encoding='utf-8',newline='\n');return
    selected=[dict(id='s4000',n=4000),dict(id='s16000',n=16000),dict(id='weak4000',n=4000,reg=.01),dict(id='group8000',n=8000,group=5)] if a.command=='suite' else [dict(id='guard_pilot_l1',n=200),dict(id='guard_pilot_group',n=200,group=5)]
    jobs=[];methods=['FBE-LBFGS','P-CONT-D','R-FISTA']
    for c in selected:
        path=a.data_dir/(c['id']+'.npz')
        if not path.exists():prepare(c,path)
        for rep in range(3 if a.command=='suite' else 1):
            for m in methods[rep:]+methods[:rep]:jobs.append(dict(case=c,method=m,label=m,rep=rep,file=f"{c['id']}_{m}_r{rep}.json"))
    deps=['fbe_benchmark.py','guard_benchmark.py','scale_benchmark.py','value_benchmark.py','run_experiments.py']
    protocol=dict(source_sha256={f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in deps},data_sha256={c['id']:hashlib.sha256((a.data_dir/(c['id']+'.npz')).read_bytes()).hexdigest() for c in selected},python=sys.version,numpy=np.__version__,platform=platform.platform(),warmup_seconds=2,measurement_budget_seconds=45,repetitions=3 if a.command=='suite' else 1,jobs=jobs)
    (a.output/'protocol.json').write_text(encoded(protocol),encoding='utf-8',newline='\n')
    for i,j in enumerate(jobs):
        if (a.output/j['file']).exists():raise RuntimeError('Use fresh output directory')
        print(i+1,len(jobs),j['file'],flush=True)
        try:subprocess.run([sys.executable,__file__,'worker','--output',str(a.output.resolve()),'--data-dir',str(a.data_dir.resolve()),'--job',json.dumps(j)],check=True,timeout=120)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:(a.output/j['file']).write_text(encoded(dict(j,status=type(e).__name__)),encoding='utf-8',newline='\n')
    (a.output/'summary.json').write_text(encoded([json.loads((a.output/j['file']).read_text()) for j in jobs]),encoding='utf-8',newline='\n')
if __name__=='__main__':main()
