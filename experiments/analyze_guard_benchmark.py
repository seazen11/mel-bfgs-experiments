"""Independent original dual-gap audit, raw summaries, and unsmoothed curves."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit,xlogy
from scale_benchmark import load

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data-dir',type=Path,required=True);a=ap.parse_args();out=a.output
    rows=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text());audit=[];agg=[]
    for f,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()==h
    for cid in dict.fromkeys(r['case']['id'] for r in rows):
        p=load(a.data_dir/(cid+'.npz'));rr=[r for r in rows if r['case']['id']==cid]
        assert hashlib.sha256((a.data_dir/(cid+'.npz')).read_bytes()).hexdigest()==protocol['data_sha256'][cid]
        for r in rr:
            assert 'x' in r,('Incomplete worker record',r['file'])
            x=np.array(r['x']);ax=p.A@x;q=expit(-p.y*ax);v=p.A.T@(-p.y*q)/p.N;b=(-v).reshape(-1,p.group)
            u=(b*np.minimum(1,p.lam/np.maximum(np.linalg.norm(b,axis=1),1e-300))[:,None]).ravel()
            gap=float(np.logaddexp(0,-p.y*ax).mean()+p.tau/2*(x@x)+p.lam*np.linalg.norm(x.reshape(-1,p.group),axis=1).sum()+(xlogy(q,q)+xlogy(1-q,1-q)).mean()+np.linalg.norm(v+u)**2/(2*p.tau))
            assert abs(gap-r['gap'])<1e-10,r['file']
            assert (gap<=r['tolerance'])==(r['status']=='converged'),r['file']
            assert r['max_linear_residual']<=1e-7 and r['max_directional_ratio']<=.5 and r['dense_fallbacks']==0
            margins=[t['threshold']-t['accepted_value'] for t in r.get('guard_log',[])];assert all(m>=0 for m in margins)
            audit.append(dict(file=r['file'],gap=gap,status=r['status'],logged_guard_checks=len(margins),minimum_logged_margin=min(margins,default=None)))
        for method in dict.fromkeys(r['method'] for r in rr):
            rs=[r for r in rr if r['method']==method];med=lambda k:float(np.median([r[k] for r in rs]));counts={k:float(np.median([r['counts'].get(k,0) for r in rs])) for k in set().union(*(r['counts'] for r in rs))}
            agg.append(dict(case=cid,method=method,runs=len(rs),success=sum(r['status']=='converged' for r in rs),seconds=med('seconds'),min_seconds=min(r['seconds'] for r in rs),max_seconds=max(r['seconds'] for r in rs),peak_mb=med('peak_process_bytes')/2**20,outer=med('outer_iterations'),counts=counts,failures=[r['status'] for r in rs if r['status']!='converged']))
        del p
    for name,data in [('independent_audit',audit),('aggregate',agg)]: (out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
    lines=['# Controlled follow-up measurements','','Median [minimum, maximum] from three runs in one batch. Failed-run time is consumed work, not time to solution. Peak memory is whole-process working set. Logs are preserved.','','|Case|Method|Success|Seconds [range]|MiB|SSN|Loss calls|Fallbacks|','|---|---|---:|---|---:|---:|---:|---:|']
    for r in agg:
        c=r['counts'];lines.append(f"|{r['case']}|{r['method']}|{r['success']}/{r['runs']}|{r['seconds']:.3f} [{r['min_seconds']:.3f}, {r['max_seconds']:.3f}]|{r['peak_mb']:.1f}|{c.get('ssn_steps',0):g}|{c.get('loss_values',0):g}|{c.get('reference_fallbacks',0)+c.get('forward_backward_fallbacks',0):g}|")
    lines+=['','All original gaps were recalculated with a separate explicit dual formula. Logged threshold margins were checked arithmetically; intermediate vectors were not retained, so this is not an independent trajectory or interval-arithmetic proof. Complete operation counts are in aggregate.json and the terminal JSON records.','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'P-CONT':('#888888','--'),'P-CONT-C':('#56B4E9','-.'),'P-CONT-D':('#D55E00','-'),'FBE-LBFGS':('#0072B2','--'),'PQN':('#009E73',':'),'R-FISTA':('#CC79A7',':'),'PN-MF':('#222222','-.')}
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42});fig,axs=plt.subplots(2,2,figsize=(7,6))
    for ax,cid in zip(axs.flat,dict.fromkeys(r['case']['id'] for r in rows)):
        for method in dict.fromkeys(r['method'] for r in rows):
            rs=[r for r in rows if r['case']['id']==cid and r['method']==method];r=sorted(rs,key=lambda r:r['seconds'])[len(rs)//2];color,ls=styles[method]
            ts=[h['time'] for h in r['history']]+[r['seconds']];gs=[max(h['gap'],1e-16) for h in r['history']]+[max(r['gap'],1e-16)]
            ax.plot(ts,gs,color=color,ls=ls,label=method)
            if r['status']!='converged':ax.plot(ts[-1],gs[-1],'x',color=color)
        ax.set(title=cid,yscale='log',xlabel='Total solve time (s)',ylabel='Original primal-dual gap');ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0);ax.grid(alpha=.2)
    h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3);fig.tight_layout(rect=(0,.08,1,1));fig.savefig(out/'comparison.pdf');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    print(f"PASS {len(audit)} independent gaps; {sum(r['status']=='converged' for r in rows)} successful; {sum(r['logged_guard_checks'] for r in audit)} logged margins")
if __name__=='__main__':main()
