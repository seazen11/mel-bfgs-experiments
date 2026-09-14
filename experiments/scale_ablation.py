"""Ablation: residual-stage projected continuation without the reference safeguard.
This method does not have the safeguarded-work theorem. Keep it separate from
published main runs; same data, warm-up, overall budgets and three repetitions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,time,json,argparse,subprocess,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scale_benchmark import Metric,HessianOperator,matrix_free_inner,proximal_inner,smooth_inner,load,encoded,peak_memory,cases
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
        except (ArithmeticError,np.linalg.LinAlgError) as err:
            status=str(err);break
        step=candidate-w;y=p.grad(candidate)-g
        if method!='PN-MF':
            if step@y>1e-12*np.linalg.norm(step)*np.linalg.norm(y):
                pairs=(pairs+[(step,y)])[-memory:];gamma=float(np.clip(y@y/(step@y),lo,hi))
            else:p.c['curvature_skips']+=1
        w=candidate
    elapsed=time.perf_counter()-start;counts=dict(p.c);gap=p.gap(answer)
    return dict(status=status,seconds=elapsed,gap=gap,tolerance=tol,x=answer,history=hist,counts=counts,outer_iterations=len(hist)-1,max_linear_residual=p.max_linear,max_directional_ratio=p.max_directional,dense_fallbacks=0,final_alpha=alpha,projected_certificate=certificate)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['suite','worker']);ap.add_argument('--output',type=Path,default=Path('runs/scale_ablation'));ap.add_argument('--data-dir',type=Path,default=Path('runs/scale_review/data'));ap.add_argument('--job');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='worker':
        job=json.loads(a.job);p=load(a.data_dir/f"{job['case']['id']}.npz")
        optimize(p,'P-CONT',budget=2);r=optimize(p,'P-CONT',budget=45);r.update(job);r['peak_process_bytes']=peak_memory()
        (a.output/job['file']).write_text(encoded(r),encoding='utf-8',newline='\n');return
    selected=[c for c in cases() if c['id'] in ['s4000','s16000','weak4000','group8000']];jobs=[]
    for c in selected:
        for rep in range(3):jobs.append(dict(case=c,method='P-CONT-NG',label='P-CONT-NG',rep=rep,file=f"{c['id']}_P-CONT-NG_r{rep}.json"))
    protocol=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),main_source_sha256=hashlib.sha256(Path(__file__).with_name('scale_benchmark.py').read_bytes()).hexdigest(),repetitions=3,warmup_seconds=2,measurement_budget_seconds=45,note='Follow-up ablation batch; do not interpret small cross-batch time differences as statistically significant.',jobs=jobs)
    (a.output/'protocol.json').write_text(encoded(protocol),encoding='utf-8',newline='\n')
    for i,j in enumerate(jobs):
        if (a.output/j['file']).exists():continue
        print(i+1,len(jobs),j['file'],flush=True)
        try:subprocess.run([sys.executable,__file__,'worker','--output',str(a.output.resolve()),'--data-dir',str(a.data_dir.resolve()),'--job',json.dumps(j)],check=True,timeout=120)
        except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as e:(a.output/j['file']).write_text(encoded(dict(j,status=type(e).__name__)),encoding='utf-8',newline='\n')
    (a.output/'summary.json').write_text(encoded([json.loads((a.output/j['file']).read_text()) for j in jobs]),encoding='utf-8',newline='\n')
if __name__=='__main__':main()
