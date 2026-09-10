"""Expanded comparison, separate from the original archived experiments."""
from run_experiments import *
import run_experiments as core
EXP=ROOT/'expanded_results';EXP.mkdir(exist_ok=True)
FIXED=['lbfgs','newton','direct_lbfgs','direct_bfgs','agd']
ORIGINAL=['lbfgs','newton','direct_lbfgs','ista','fista','rfista']
LABEL={'lbfgs':'MEL','newton':'PN','direct_lbfgs':'L-BFGS','direct_bfgs':'BFGS','agd':'AGD','ista':'ISTA','fista':'FISTA','rfista':'R-FISTA'}
COLORS={'lbfgs':'#0072B2','newton':'#D55E00','direct_lbfgs':'#009E73','direct_bfgs':'#CC79A7','agd':'#666666','ista':'#E69F00','fista':'#CC79A7','rfista':'#202020'}
STYLES={'lbfgs':'-','newton':'--','direct_lbfgs':'-.','direct_bfgs':':','agd':(0,(5,1,1,1)),'ista':':','fista':'--','rfista':'-.'}
def save(name,obj):
    (EXP/(name+'.json')).write_text(json.dumps(obj,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)),encoding='utf-8')
def load(name):return json.loads((EXP/(name+'.json')).read_text(encoding='utf-8'))

def additional(p,alpha,tol,method,original,maxit=15000):
    x=np.zeros(p.n);z=x.copy();t=1.;hist=[];start=time.perf_counter();restarts=0
    def record(v):
        gap=p.gap(v,0 if original else alpha)
        hist.append({'k':len(hist),'time':time.perf_counter()-start,'gap':gap,'original_gap':p.gap(v),'x':v.copy()})
        return gap
    record(x)
    if method=='direct_bfgs':
        class TargetReached(Exception):pass
        def cb(v):
            if record(v)<=tol:raise TargetReached()
        try:
            rr=minimize(lambda v:(p.value(v,alpha),p.grad(v)+p.gh(v,alpha)),x,jac=True,method='BFGS',callback=cb,
                options={'maxiter':maxit,'gtol':1e-13})
            x=rr.x;record(x);reason=str(rr.message)
        except TargetReached:
            # Callback retains its final point for independent verification.
            x=hist[-1]['x'];reason='external_gap_test'
    else:
        L=p.L+1/alpha if method=='agd' else p.L
        momentum=(np.sqrt(L)-np.sqrt(p.tau))/(np.sqrt(L)+np.sqrt(p.tau)) if method=='agd' else 0.
        reason='iteration_limit'
        for k in range(maxit):
            if method=='agd':
                xn=z-(p.grad(z)+p.gh(z,alpha))/L;zn=xn+momentum*(xn-x)
            else:
                v=z-p.grad(z)/L;xn=np.sign(v)*np.maximum(np.abs(v)-p.lam/L,0)
                if method=='ista':zn=xn.copy()
                else:
                    tn=(1+np.sqrt(1+4*t*t))/2
                    if (z-xn)@(xn-x)>0:
                        tn=1.;zn=xn.copy();restarts+=1
                    else:zn=xn+(t-1)/tn*(xn-x)
                    t=tn
            x,z=xn,zn
            if record(x)<=tol:reason='external_gap_test';break
    return {'history':hist,'x':x,'status':'converged' if hist[-1]['gap']<=tol else 'not_certified','reason':reason,'restarts':restarts}

def run_one(p,test,method):
    alpha=.01 if test=='fixed' else 1e-6/(p.n*p.lam*p.lam)
    tol=1e-13 if test=='fixed' else 1e-6
    start=time.perf_counter()
    if method in ['lbfgs','newton']:
        r=nested(p,alpha,tol=tol,variant=method,original=test=='original')
    elif method in ['direct_lbfgs','fista']:
        r=baseline(p,alpha,tol=tol,method=method,original=test=='original')
    else:r=additional(p,alpha,tol,method,test=='original')
    elapsed=time.perf_counter()-start
    independently_checked=p.gap(r['x'],0 if test=='original' else alpha)
    assert abs(independently_checked-r['history'][-1]['gap'])<1e-10
    if r['status']=='converged':assert independently_checked<=tol
    # Preserve all plotted observations, but avoid duplicating large state arrays.
    slim=[]
    for row in r['history']:
        v={k:row[k] for k in ['k','time','gap','outer_step','eta','relative_model_residual'] if k in row}
        if 'inner' in row:
            v['inner_iterations']=len(row['inner'])-1
            v['max_linear_residual']=max(q.get('linear_residual',0) for q in row['inner'])
            assert row['relative_model_residual']<=1.001*row['eta']
        slim.append(v)
    slim[-1]['time']=elapsed
    return {'history':slim,'x':r['x'],'status':r['status'],'reason':r.get('reason',''),
            'seconds':elapsed,'alpha':alpha,'tolerance':tol,'checked_gap':independently_checked,'restarts':r.get('restarts',0)}

def self_check(ps):
    # Check the proximal first step, gradient mapping, and independent certificates.
    p=ps[0];x=np.zeros(p.n);v=x-p.grad(x)/p.L
    expected=np.sign(v)*np.maximum(np.abs(v)-p.lam/p.L,0)
    rr=additional(p,.01,0.,'ista',True,maxit=1)
    assert np.allclose(rr['x'],expected,rtol=0,atol=1e-15)
    assert p.value(expected,0)<=p.value(x,0)
    for m in ['ista','rfista','agd','direct_bfgs']:
        r=additional(p,.01,1e-8,m,m in ['ista','rfista'])
        assert r['status']=='converged',(m,r['status'])
        assert p.gap(r['x'],0 if m in ['ista','rfista'] else .01)<=1e-8
    save('checks',{'proximal_first_step':'passed','descent_first_step':'passed','four_new_solvers':'passed'})

def suite():
    ps=datasets();self_check(ps);summary=[]
    for p in ps:
        print('Expanded:',p.name,flush=True)
        for test,methods in [('fixed',FIXED),('original',ORIGINAL)]:
            # One warm-up per method. Rotate measured order to reduce order effects.
            for m in methods:run_one(p,test,m)
            records={m:[] for m in methods}
            for rep in range(3):
                order=methods[rep:]+methods[:rep]
                for m in order:
                    r=run_one(p,test,m);save(f'{p.name}_{test}_{m}_r{rep}',r);records[m].append((rep,r))
            for m in methods:
                runs=records[m];ordered=sorted(runs,key=lambda z:z[1]['seconds']);midrep,mid=ordered[1]
                row={'problem':p.name,'test':test,'method':m,'median':mid['seconds'],'median_run':midrep,
                     'times':[r['seconds'] for _,r in runs],'iterations':[len(r['history'])-1 for _,r in runs],
                     'statuses':[r['status'] for _,r in runs],'gaps':[r['checked_gap'] for _,r in runs]}
                summary.append(row);print(test,m,row['statuses'][0],row['iterations'][0],round(row['median'],4),flush=True)
            save('summary',summary)
    save('protocol',{'fixed_methods':FIXED,'original_methods':ORIGINAL,'warmup':1,'measured_runs':3,
        'curve_instances':['synthetic_rho0.9_seed0','wdbc'],'representative':'median-time run for each method and problem',
        'fixed_alpha':.01,'fixed_gap':1e-13,'original_gap':1e-6,'original_alpha':'1e-6/(n*lambda^2)',
        'order':'method order rotated over repetitions','threads':threadpool_info(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    plots()

def plots():
    rows=load('summary');plt.rcParams.update({'font.family':'serif','font.size':12,'axes.labelsize':12,'legend.fontsize':11,'pdf.fonttype':42})
    def curves(test,axis,name):
        fig,axs=plt.subplots(1,2,figsize=(6.8,3.4),sharey=True)
        methods=FIXED if test=='fixed' else ORIGINAL
        for ax,problem,title in zip(axs,['synthetic_rho0.9_seed0','wdbc'],['(a) C0','(b) WDBC']):
            for m in methods:
                row=next(r for r in rows if r['problem']==problem and r['test']==test and r['method']==m)
                run=load(f'{problem}_{test}_{m}_r{row["median_run"]}');hs=run['history']
                ax.plot([r['time'] if axis=='time' else r['k'] for r in hs],[max(r['gap'],1e-16) for r in hs],
                    color=COLORS[m],ls=STYLES[m],lw=1.5,label=LABEL[m])
            ax.set_yscale('log');ax.set_xscale('symlog',linthresh=.001 if axis=='time' else 1.)
            ax.set_xlim(left=0)
            ax.axhline(1e-13 if test=='fixed' else 1e-6,color='.5',lw=.7,ls=':')
            ax.set_xlabel('Time (s)' if axis=='time' else 'Iteration');ax.set_title(title);ax.grid(alpha=.2)
        axs[0].set_ylabel('Smoothed primal-dual gap' if test=='fixed' else 'Original primal-dual gap')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.15,1,1));fig.savefig(EXP/(name+'.pdf'),bbox_inches='tight');fig.savefig(EXP/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    curves('fixed','iteration','fixed_iterations');curves('fixed','time','fixed_time');curves('original','time','original_time')
    fig,axs=plt.subplots(1,2,figsize=(6.8,3.35),sharey=True);profile=[]
    for ax,test,methods,title in zip(axs,['fixed','original'],[FIXED,ORIGINAL],['(a) Fixed smoothing','(b) Original accuracy']):
        problems=sorted({r['problem'] for r in rows if r['test']==test});ratios={m:[] for m in methods}
        for problem in problems:
            times={m:next(r for r in rows if r['test']==test and r['method']==m and r['problem']==problem) for m in methods}
            good={m:r['median'] for m,r in times.items() if all(q=='converged' for q in r['statuses'])};best=min(good.values())
            for m in methods:ratios[m].append(times[m]['median']/best if m in good else float('inf'))
        xmax=max(v for vs in ratios.values() for v in vs if np.isfinite(v))*1.05
        for m in methods:
            xs=np.unique([1.,*sorted(v for v in ratios[m] if np.isfinite(v)),xmax]);ys=[sum(v<=q for v in ratios[m])/len(problems) for q in xs]
            ax.step(xs,ys,where='post',label=LABEL[m],color=COLORS[m],ls=STYLES[m],lw=1.5)
            profile.append({'test':test,'method':m,'ratios':ratios[m],'wins':sum(v<=1+1e-12 for v in ratios[m])})
        ax.set_xscale('log');ax.set_xlim(1,xmax);ax.set_ylim(0,1.03);ax.set_title(title);ax.set_xlabel('Time / best time');ax.grid(alpha=.2)
    axs[0].set_ylabel('Fraction of seven instances')
    for ax in axs:ax.legend(fontsize=10.5,loc='lower right',framealpha=.9)
    fig.tight_layout();fig.savefig(EXP/'profiles.pdf',bbox_inches='tight');fig.savefig(EXP/'profiles.png',dpi=180,bbox_inches='tight');plt.close(fig);save('profiles',profile)

if __name__=='__main__':
    if '--plots-only' in sys.argv:plots()
    else:suite()
