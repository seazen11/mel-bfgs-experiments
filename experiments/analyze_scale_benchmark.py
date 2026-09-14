"""Independently verify terminal original gaps and summarize the scale campaign."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from scale_benchmark import load,prepare

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('runs/scale_review'));ap.add_argument('--ablation',type=Path);a=ap.parse_args();out=a.output
    rows=json.loads((out/'summary.json').read_text());audit=[];aggregates=[]
    # Process one case at a time; do not retain all large matrices in memory.
    for cid in dict.fromkeys(r['case']['id'] for r in rows):
        rr=[r for r in rows if r['case']['id']==cid];path=out/'data'/f'{cid}.npz'
        if not path.exists():path.parent.mkdir(exist_ok=True);prepare(rr[0]['case'],path)
        p=load(path)
        for r in rr:
            if 'x' not in r:continue
            x=np.asarray(r['x']);gap=p.gap(x)
            assert abs(gap-r['gap'])<1e-10*max(1.,abs(gap)),r['file']
            if r['status']=='converged':assert gap<=r['tolerance'],r['file']
            assert r['max_linear_residual']<=1e-7 and r['max_directional_ratio']<=.5
            assert r['dense_fallbacks']==0
            audit.append(dict(file=r['file'],gap=gap,status=r['status']))
        for method in dict.fromkeys(r['method'] for r in rr):
            rs=[r for r in rr if r['method']==method];valid=[r for r in rs if 'seconds' in r]
            def median(key):return float(np.median([r[key] for r in valid])) if valid else None
            counts={k:float(np.median([r['counts'].get(k,0) for r in valid])) for k in set().union(*(r.get('counts',{}) for r in valid))}
            aggregates.append(dict(case=cid,method=method,runs=len(rs),success=sum(r['status']=='converged' for r in rs),seconds=median('seconds'),min_seconds=min((r['seconds'] for r in valid),default=None),max_seconds=max((r['seconds'] for r in valid),default=None),peak_mb=median('peak_process_bytes')/2**20 if valid else None,outer=median('outer_iterations'),final_alpha=median('final_alpha'),counts=counts,failures=[r['status'] for r in rs if r['status']!='converged']))
        del p
    (out/'independent_audit.json').write_text(json.dumps(audit,indent=2)+'\n',newline='\n')
    (out/'aggregate.json').write_text(json.dumps(aggregates,indent=2)+'\n',newline='\n')
    def fmt(x):return 'NA' if x is None else f'{x:.4g}'
    lines=['# 扩大规模正式结果','','每设置三次测量。失败时间表示已消耗预算，不能解释为完成求解所需时间。峰值为整个独立进程的驻留内存，包括数据和预热。','','| 设置 | 方法 | 达标/次数 | 中位秒 | 最小—最大 | 峰值 MiB | 外层循环 | SSN / 近端内层 | 最终 α |','|---|---|---:|---:|---|---:|---:|---:|---:|']
    for r in aggregates:
        c=r['counts'];lines.append(f"|{r['case']}|{r['method']}|{r['success']}/{r['runs']}|{fmt(r['seconds'])}|{fmt(r['min_seconds'])}—{fmt(r['max_seconds'])}|{fmt(r['peak_mb'])}|{fmt(r['outer'])}|{c.get('ssn_steps',0):g}/{c.get('inner_proximal_steps',0):g}|{fmt(r['final_alpha']) if r['method'] not in ['PQN','R-FISTA','PN-MF'] else 'NA'}|")
    lines+=['','## 操作计数（三次中位数）','','| 设置/方法 | A / AT | 梯度 | Hessian 向量积 | 原证书 / 投影残差证书 | 谱检查 | 小系统 | 内/外回溯 | 参考步 / 回退 | 失败试探 | 阶段切换 / 重置 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in aggregates:
        c=r['counts'];v=lambda k:f'{c.get(k,0):g}'
        lines.append(f"|{r['case']}/{r['method']}|{v('A_products')}/{v('AT_products')}|{v('gradients')}|{v('hessian_vector_products')}|{v('certificates')}/{v('projected_residual_certificates')}|{v('metric_checks')}|{v('reduced_solves')}|{v('inner_backtracks')}/{v('outer_backtracks')}|{v('reference_gradient_steps')}/{v('reference_fallbacks')}|{v('trial_failures')}|{v('stage_changes')}/{v('stage_resets')}|")
    lines+=['','## 未达标记录','']+[f"- {r['file']}: {r['status']}; gap={r.get('gap','unavailable')}" for r in rows if r['status']!='converged']
    (out/'SCALE_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'CONT':'#0072B2','CONT-P':'#56B4E9','P-CONT':'#D55E00','PQN':'#009E73','R-FISTA':'#CC79A7','PN-MF':'#333333','MEL':'#777777'}
    styles={'CONT':'--','CONT-P':'-.','P-CONT':'-','PQN':(0,(4,1,1,1)),'R-FISTA':':','PN-MF':(0,(5,2)),'MEL':(0,(1,1))}
    methods=list(colors)[:-1];plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
    for filename,ids in [('large_curves',['s16000','samples2000']),('recovery_curves',['tight4000','group8000'])]:
        fig,axs=plt.subplots(1,2,figsize=(7,3.5))
        for ax,cid in zip(axs,ids):
            for method in methods:
                rs=[r for r in rows if r['case']['id']==cid and r['method']==method and 'seconds' in r]
                if not rs:continue
                r=sorted(rs,key=lambda t:t['seconds'])[len(rs)//2]
                ts=[h['time'] for h in r['history']]+[r['seconds']];gs=[max(h['gap'],1e-16) for h in r['history']]+[max(r['gap'],1e-16)]
                ax.plot(ts,gs,color=colors[method],ls=styles[method],label=method)
                if r['status']!='converged':ax.plot(ts[-1],gs[-1],'x',color=colors[method])
            ax.set(yscale='log',xlabel='Total solve time (s)',title=cid);ax.set_xscale('symlog',linthresh=.01);ax.set_xlim(left=0);ax.grid(alpha=.2)
        axs[0].set_ylabel('Original primal-dual gap');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=11);fig.tight_layout(rect=(0,.17,1,1));fig.savefig(out/(filename+'.pdf'));fig.savefig(out/(filename+'.png'),dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7,3.5))
    for method in methods:
        rs=[next(r for r in aggregates if r['case']==cid and r['method']==method) for cid in ['s4000','s8000','s16000']]
        axs[0].plot([4000,8000,16000],[r['seconds'] if r['success']==3 else np.nan for r in rs],marker='o',ls=styles[method],color=colors[method],label=method)
        axs[1].plot([4000,8000,16000],[r['peak_mb'] for r in rs],marker='o',ls=styles[method],color=colors[method],label=method)
    for ax in axs:ax.set_xlabel('Dimension n');ax.set_xscale('log');ax.set_xticks([4000,8000,16000],['4000','8000','16000']);ax.xaxis.set_minor_locator(plt.NullLocator());ax.grid(alpha=.2)
    axs[0].set_ylabel('Median solution time (s)');axs[0].set_yscale('log');axs[1].set_ylabel('Peak process memory (MiB)')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=11);fig.tight_layout(rect=(0,.17,1,1));fig.savefig(out/'large_scaling.pdf');fig.savefig(out/'large_scaling.png',dpi=170);plt.close(fig)
    if a.ablation:
        ab=json.loads((a.ablation/'summary.json').read_text());ab_audit=[];ab_agg=[]
        for cid in dict.fromkeys(r['case']['id'] for r in ab):
            p=load(out/'data'/f'{cid}.npz');rs=[r for r in ab if r['case']['id']==cid]
            for r in rs:
                if 'x' not in r:continue
                gap=p.gap(np.array(r['x']));assert abs(gap-r['gap'])<1e-10
                if r['status']=='converged':assert gap<=r['tolerance']
                ab_audit.append(dict(file=r['file'],gap=gap,status=r['status']))
            valid=[r for r in rs if 'seconds' in r]
            ab_agg.append(dict(case=cid,success=sum(r['status']=='converged' for r in rs),runs=len(rs),seconds=float(np.median([r['seconds'] for r in valid])),min_seconds=min(r['seconds'] for r in valid),max_seconds=max(r['seconds'] for r in valid),median_ssn=float(np.median([r['counts'].get('ssn_steps',0) for r in valid])),median_stages=float(np.median([r['counts'].get('stage_changes',0) for r in valid]))))
            del p
        for name,data in [('independent_audit',ab_audit),('aggregate',ab_agg)]:
            (a.ablation/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',newline='\n')
        text=['# 参考保障消融','','P-CONT-NG 为主实验之后的独立测量批次。每设置三次测量。未达标时间不是求解时间。','','| 设置 | 达标 | 中位秒 | 最小—最大 | 中位 SSN | 阶段切换 |','|---|---:|---:|---|---:|---:|']
        for r in ab_agg:text.append(f"|{r['case']}|{r['success']}/{r['runs']}|{r['seconds']:.4g}|{r['min_seconds']:.4g}—{r['max_seconds']:.4g}|{r['median_ssn']:g}|{r['median_stages']:g}|")
        text+=['','## 未达标记录','']+[f"- {r['file']}: {r['status']}" for r in ab if r['status']!='converged']
        (a.ablation/'ABLATION_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8',newline='\n')
        print('PASS:',len(ab_audit),'independent ablation terminal checks.')
    print('PASS:',len(audit),'independent terminal checks;',sum(r['status']=='converged' for r in rows),'successes;',len(rows),'measured records; three figure groups.')
if __name__=='__main__':main()
