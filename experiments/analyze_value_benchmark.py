"""Verify real terminal states and generate review tables, without changing raw runs."""
import os,sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from value_benchmark import Problem
import numpy as np
import argparse
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'runs/value_review');args=ap.parse_args();OUT=args.output.resolve()
rows=json.loads((OUT/'summary.json').read_text())
problems={};checked=[]
for r in rows:
    cid=r['case']['id']
    if cid not in problems:
        conf=dict(r['case']);conf.pop('id');conf.pop('tol',None);problems[cid]=Problem(**conf)
    if 'x' not in r:continue
    p=problems[cid];g=p.gap(np.array(r['x']))
    assert abs(g-r['gap'])<1e-11,(r['file'],g,r['gap'])
    if r['status']=='converged':assert g<=r['tolerance'],r['file']
    assert r['max_directional_ratio']<=.5 and r['max_linear_residual']<=1e-7
    checked.append({'file':r['file'],'independent_gap':g,'status':r['status']})
(OUT/'independent_audit.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
print('Independently checked',len(checked),'terminal states')
agg=json.loads((OUT/'aggregate.json').read_text())
lines=['# 原问题精度统一比较：正式运行记录','',
       '每设置三次正式测量。数据为独立 worker 的实际结果；失败时间是已花费时间，不是 time-to-solution。峰值为整个 worker 的操作系统 peak resident working set，包含依赖、数据准备与预热，不能解释为纯算法工作空间。','',
       '| 实例 | 方法 | 成功 | 中位秒 | 最小—最大秒 | 峰值 MiB | 外层/算法更新 | 内层 SSN |','|---|---|---:|---:|---|---:|---:|---:|']
for r in agg:
    def fmt(v):return 'NA' if v is None else f'{v:.4g}'
    lines.append(f"| {r['case']} | {r['method']} | {r['success']}/{r['runs']} | {fmt(r['median_seconds'])} | {fmt(r['min_seconds'])}—{fmt(r['max_seconds'])} | {fmt(r['median_peak_mb'])} | {fmt(r['median_outer'])} | {fmt(r['median_ssn'])} |")
lines+=['','## 失败清单','']+[f"- {r['file']}: {r['status']}; final gap={r.get('gap','unavailable')}" for r in rows if r['status']!='converged']
lines+=['','## 全部关键运算（按设置给出三次中位数）','','| 实例/方法 | A/AT 乘 | 梯度 | Hessian | 证书 | 谱检查 | Metric 乘 | Reduced/Dense solve | 内层回溯 | 外层回溯 | PG safeguard | 重置/跳过 | 稠密回退 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for a in agg:
    rs=[r for r in rows if r['case']['id']==a['case'] and r['label']==a['method'] and 'counts' in r]
    def med(key):return f'{np.median([r["counts"].get(key,0) for r in rs]):g}' if rs else 'NA'
    lines.append(f"| {a['case']}/{a['method']} | {med('A_products')}/{med('AT_products')} | {med('gradients')} | {med('dense_hessians')} | {med('certificates')} | {med('metric_checks')} | {med('metric_products')} | {med('reduced_solves')}/{med('dense_solves')} | {med('inner_backtracks')} | {med('outer_backtracks')} | {med('proximal_safeguards')} | {med('metric_resets')}/{med('curvature_skips')} | 0 |")
(OUT/'EXPERIMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors={'MEL':'#0072B2','CONT':'#D55E00','PQN':'#009E73','R-FISTA':'#CC79A7','PN':'#777777','MEL-PN':'#000000'}
plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
for filename,ids in [('value_curves',['C0','n400']),('structure_curves',['ill','group400'])]:
    fig,axs=plt.subplots(1,2,figsize=(7,3.4))
    for ax,cid in zip(axs,ids):
        for method,color in colors.items():
            rr=[r for r in rows if r['case']['id']==cid and r['label']==method and 'seconds' in r]
            if not rr:continue
            r=sorted(rr,key=lambda z:z['seconds'])[len(rr)//2]
            times=[v['time'] for v in r['history']]+[r['seconds']]
            gaps=[max(v['gap'],1e-16) for v in r['history']]+[max(r['gap'],1e-16)]
            ax.plot(times,gaps,label=method,color=color,ls={'MEL':'-','CONT':'--','PQN':'-.','R-FISTA':':','PN':(0,(5,1)),'MEL-PN':(0,(1,1))}[method])
            if r['status']!='converged':ax.plot(r['seconds'],max(r['gap'],1e-16),'x',color=color)
        ax.set(yscale='log',xlabel='Total solve time (s)',title=cid);ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(left=0);ax.grid(alpha=.2);ax.axhline(1e-6,color='.5',ls=':',lw=.7)
    axs[0].set_ylabel('Original primal-dual gap');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,fontsize=11)
    fig.tight_layout(rect=(0,.16,1,1));fig.savefig(OUT/(filename+'.pdf'));fig.savefig(OUT/(filename+'.png'),dpi=170);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(7,3.2))
for method in ['MEL','CONT','PQN','R-FISTA']:
    rs=[next(r for r in agg if r['case']==cid and r['method']==method) for cid in ['C0','n400','n1200','n2400']]
    ns=[80,400,1200,2400]
    axs[0].plot(ns,[r['median_seconds'] if r['success']==r['runs'] else np.nan for r in rs],'-o',label=method,color=colors[method])
    axs[1].plot(ns,[r['median_peak_mb'] for r in rs],'-o',label=method,color=colors[method])
for ax in axs:ax.set_xlabel('Dimension n');ax.set_xscale('log');ax.grid(alpha=.2)
axs[0].set_ylabel('Median solve time (s)');axs[0].set_yscale('log');axs[1].set_ylabel('Peak process memory (MiB)')
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=4,fontsize=11);fig.tight_layout(rect=(0,.12,1,1))
fig.savefig(OUT/'scaling.pdf');fig.savefig(OUT/'scaling.png',dpi=170);plt.close(fig)
print('Generated three figure groups and complete operation tables')
