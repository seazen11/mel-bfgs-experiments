from run_experiments import *

def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))

def audit():
    violations=[]; ratios=[]; linear=[]; n=0
    for path in list(OUT.glob('*_fixed_*.json'))+list(OUT.glob('*_original_*.json')):
        run=json.loads(path.read_text());
        for row in run['history']:
            if 'eta' not in row:continue
            n+=1; ratio=row['relative_model_residual']/row['eta']; ratios.append(ratio)
            if ratio>1.001:violations.append([path.name,row['k'],ratio])
            linear.extend([h.get('linear_residual',0) for h in row['inner']])
    ans={'outer_steps':n,'max_relative_tolerance_ratio':max(ratios),'max_linear_relative_residual':max(linear),'violations':violations}
    dump('audit',ans); print('AUDIT',ans,flush=True)
    return ans

def repeated():
    ps=datasets(); rows=[]
    for p in ps:
        print('Repeated timings',p.name,flush=True)
        for test,methods in [('fixed',['lbfgs','newton','direct_lbfgs']),('original',['lbfgs','direct_lbfgs','fista'])]:
            for method in methods:
                values=[]; steps=[]; statuses=[]
                for rep in range(4):
                    alpha=.01 if test=='fixed' else 1e-6/(p.n*p.lam*p.lam)
                    tol=1e-13 if test=='fixed' else 1e-6
                    if method in ['lbfgs','newton']:r=nested(p,alpha,tol=tol,variant=method,original=test=='original')
                    else:r=baseline(p,alpha,tol=tol,method=method,original=test=='original')
                    if rep:
                        values.append(r['history'][-1]['time']); steps.append(len(r['history'])-1); statuses.append(r['status'])
                rows.append({'problem':p.name,'test':test,'method':method,'times':values,'median':np.median(values),
                             'min':min(values),'max':max(values),'iterations':steps,'statuses':statuses})
        dump('repeated_timings',rows)
    # Memory sensitivity and an SSN versus gradient-inner comparison on identical models.
    p=ps[3]; ms=[]
    for mem in [3,5,10,20]:
        r=nested(p,.01,tol=1e-13,memory=mem); dump('memory_'+str(mem),r)
        ms.append({'memory':mem,'status':r['status'],'iterations':len(r['history'])-1,'seconds':r['history'][-1]['time']})
    dump('memory_summary',ms)
    p=ps[0]; dump('inner_ssn_ablation',nested(p,.1,tol=1e-6))

def report():
    summary=read('summary'); compact=read('compact'); bias=read('bias'); checks=read('checks'); aud=audit(); timing=read('repeated_timings')
    manifest=read('manifest'); lines=[]
    lines += ['# 嵌套 MEL-BFGS：本地实验报告','',
        '本报告记录实际运行结果。预期与观察分开陈述。此前其他算法生成的 GISETTE、ARCENE 等结果未参与本次比较。',
        '', '## 1. 目标函数与数据', '',
        r'采用 $F(x)=N^{-1}\sum_{i=1}^N\log(1+\exp(-y_i a_i^Tx))+\frac{\tau}{2}\|x\|^2+\lambda\|x\|_1$，其中 $\tau=0.02$，$\lambda=0.1\|A^Ty/(2N)\|_\infty$。岭项归入 $f$。', '',
        r'因此 $f$ 是强凸、二阶连续可微函数，$L_f=\|A\|_2^2/(4N)+\tau$。正则项的 Moreau 包络为逐坐标 Huber 函数，梯度为 $\operatorname{clip}(x/\alpha,-\lambda,\lambda)$，广义 Hessian 为对角矩阵。这一结构满足内层强半光滑性，并使基矩阵求解成本为线性量级。$L_h=\lambda\sqrt n$。', '',
        '- 合成数据：600 个样本、80 个变量；相关系数为 0 和 0.9；各使用随机种子 0、1、2，共 6 个问题。真实系数前 10 项非零，标签按逻辑模型抽样。',
        '- 真实数据：[UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)，569 个样本、30 个变量，已下载到 `data/wdbc.data`。标签 M/B 映射为 +1/−1。',
        '- 每列以全部样本的均值、标准差进行标准化，无截距。这里测量优化性能，不报告分类泛化能力，因此不设置训练／测试划分。',
        f'- WDBC 文件 SHA-256：`{manifest["wdbc_sha256"]}`。',
        '', '该问题适合检验论文的可分近端结构，但不能覆盖一般分析算子或昂贵近端映射。真实数据规模较小；大维度实验仅测试线性代数成本，不代表端到端大规模数据性能。',
        '', '## 2. 实现与精度标准', '',
        r'外层使用 $B_k$ 近似 $\nabla^2f$，不是近似整个平滑目标的 Hessian。L-BFGS 保留 10 对曲率信息，以紧凑 Hessian 形式构造；谱界设为 $m_B=\tau/10$、$M_B=10L_f$，越界时重置。子问题采用 SSN、Woodbury 求解和 Armijo 回溯。外层、内层 Armijo 参数均为 $10^{-4}$，回缩比例为 $1/2$。', '',
        r'内层停止条件为 $\|R\|\le\eta_k\|s\|$，其中 $\eta_k=\min\{m_B/4,0.1\|\nabla\Phi_\alpha(x_k)\|\}$。内层最多 2000 步，外层最多 600 步；直接平滑 L-BFGS 和 FISTA 最多 15000 步。', '',
        '对照包括：直接对平滑目标运行 SciPy L-BFGS-B（无边界约束、记忆 10）、嵌套完整 BFGS、嵌套精确 Hessian、原问题上的 FISTA，以及几何延拓。FISTA 使用原正则项的精确软阈值近端，不受平滑偏差影响。', '',
        r'固定平滑实验取 $\alpha=10^{-2}$，平滑原始—对偶间隙不超过 $10^{-13}$。同原问题精度实验统一要求原问题原始—对偶间隙不超过 $10^{-6}$，取最终 $\alpha=10^{-6}/(n\lambda^2)$。延拓从 0.1 起按 0.1 倍缩小到同一最终参数，每阶段控制平滑间隙不超过 $\alpha n\lambda^2/4$。', '',
        r'设 $p_i=\operatorname{sigmoid}(-y_i a_i^Tx)$，$v=A^T(-y\odot p)/N$，$u=\operatorname{clip}(-v/(1+\tau\alpha),-\lambda,\lambda)$。使用的对偶下界为', '',
        r'$$D_\alpha=-\frac1N\sum_i[p_i\log p_i+(1-p_i)\log(1-p_i)]-\frac{\|v+u\|^2}{2\tau}-\frac{\alpha}{2}\|u\|^2.$$', '',
        r'该式来自 logistic 损失、岭项和 $h_\alpha$ 的 Fenchel 共轭；$u$ 在盒约束内，因此对任意当前点均可行。$\Phi_\alpha(x)-D_\alpha$ 是平滑误差上界；令 $\alpha=0$ 即得到原问题的误差上界。原问题与平滑问题使用各自的对偶下界，不互相替代。', '',
        '计算采用双精度和单 BLAS 线程。接近机器精度时 Armijo 检查允许约 $4\times10^{-16}$ 的舍入裕量，内层残差存在 $2\times10^{-13}$ 的绝对停止下限。以下审计检查该下限是否在主实验中破坏相对精度。这里的“证书”指双精度数值误差上界估计，不是区间算术证明。', '',
        f'- 梯度方向差分误差：{checks["gradient_error"]:.3e}；紧凑／稠密方向相对差：{checks["woodbury_error"]:.3e}。',
        f'- 审计 {aud["outer_steps"]} 个已返回外层方向，相对残差／容许残差的最大比值为 {aud["max_relative_tolerance_ratio"]:.3g}；违规方向数为 {len(aud["violations"])}。',
        f'- 主实验 SSN 线性系统最大相对残差为 {aud["max_linear_relative_residual"]:.3e}。',
        '', '## 3. 固定平滑问题：收敛与速率', '',
        '预期：低秩结构应减少 SSN 线性代数成本；精确 Hessian 的方向误差应趋于零；固定记忆 L-BFGS 不自动满足 Dennis–Moré 条件。', '',
        '以下时间为预热一次后独立重复三次的中位数，单位秒。计时包含迭代中的证书计算、谱界检查、内层迭代和线搜索，不含数据预处理、参考解求解和结果写盘。Python 嵌套实现与 SciPy 编译实现的开销不同，故不作实现无关的复杂度结论。', '',
        '| 问题 | 嵌套 L-BFGS（步／秒） | 精确 Hessian（步／秒） | 直接平滑 L-BFGS（步／秒） |','|---|---:|---:|---:|']
    names=[p['name'] for p in manifest['problems']]
    for name in names:
        vals=[]
        for method in ['lbfgs','newton','direct_lbfgs']:
            r=next(r for r in timing if r['problem']==name and r['method']==method and r['test']=='fixed')
            vals.append(f'{r["iterations"][0]} / {r["median"]:.4f}')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '所有固定平滑对照均达到了目标精度。嵌套 L-BFGS 相比直接平滑 L-BFGS 使用更少的外层步，在本次测量中更快。由于变量数只有 30 或 80，精确 Hessian 的成本较低，不能据此宣称拟牛顿优于 Newton。', '',
        r'局部速率诊断采用独立参考解，先用直接平滑 L-BFGS 求解，再用精确 Hessian 修正。参考解的梯度给出距离上界 $\|\nabla\Phi\|/\tau$。只讨论明显高于该误差和浮点误差的迭代区间。', '',
        '| 问题 | 参考解距离上界 | L-BFGS 最后三个方向误差 | Newton 最后三个方向误差 |','|---|---:|---|---|']
    for name in names:
        vals=[]
        for meth in ['lbfgs','newton']:
            hs=read(name+'_fixed_'+meth)['history']; vals.append(', '.join(f'{h["dm"]:.2e}' for h in hs if 'dm'in h)[-120:])
        # Keep exactly the last three diagnostics, not a misleading fitted order.
        vals=[', '.join(f'{h["dm"]:.2e}' for h in read(name+'_fixed_'+meth)['history'] if 'dm'in h) for meth in ['lbfgs','newton']]
        vals=[', '.join(v.split(', ')[-3:]) for v in vals]
        lines.append(f'| {name} | {read(name+"_reference")["distance_certificate"]:.2e} | {vals[0]} | {vals[1]} |')
    lines += ['',r'方向误差为 $\|(B_k-\nabla^2f(x_\alpha^*))s_k\|/\|s_k\|$。固定记忆方法的有限曲线不能证明该量渐近趋于零，也不能证明外层超线性定理的假设成立。精确 Hessian 对照提供满足方向一致性的可解释验证。', '',
        '冻结子问题的 SSN 结果如下。正则项为分段二次包络，活跃结构稳定后可有限步到达数值解；此时不能对末尾零误差强行拟合二次收敛阶。', '',
        '| 问题 | SSN 步数 | 最后三个残差 |','|---|---:|---|']
    for name in names:
        hs=read(name+'_inner')['history']; lines.append(f'| {name} | {len(hs)-1} | '+', '.join(f'{r["residual"]:.2e}' for r in hs[-3:])+' |')
    rate_rows=[]
    lines += ['', '进一步使用真实迭代误差比，而不是目标间隙比，检查局部速率。表中只保留相邻误差均大于参考解距离上界 20 倍的比值，列出最后三个可分辨值。', '',
        '| 问题 | L-BFGS 相邻误差比 | Newton 相邻误差比 |','|---|---|---|']
    for name in names:
        vals=[]; floor=20*read(name+'_reference')['distance_certificate']
        for meth in ['lbfgs','newton']:
            hs=read(name+'_fixed_'+meth)['history']; ratios=[]
            for old,new in zip(hs,hs[1:]):
                if min(old['reference_error'],new['reference_error'])>floor:
                    ratios.append(new['reference_error']/old['reference_error'])
            rate_rows.append({'problem':name,'method':meth,'resolvable_ratios':ratios,'reference_floor':floor})
            vals.append(', '.join(f'{q:.3e}' for q in ratios[-3:]))
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    dump('local_rates',rate_rows)
    lines += ['', 'Newton 对照的误差比下降与局部超线性结论相符；有限数据仍不构成渐近数学证明。L-BFGS 的比值需结合方向误差判断，不能由较少迭代次数代替速率条件。', '',
        '图中的原始—对偶间隙上界可能不单调，因为对偶点也随迭代改变；SSN 残差也可能在活跃结构变化时上升。论文的 Armijo 单调性针对目标或模型值，不要求这些诊断量每一步都下降。', '',
        '![收敛诊断](results/convergence.png)', '', '## 4. 原问题相同精度比较', '',
        '预期：减小平滑参数会减小偏差，也会增加局部刚性。延拓可能改善初始化，但阶段开销可能抵消收益。所有方法均按原问题间隙停止。', '',
        '| 问题 | 嵌套 L-BFGS 秒 | 直接平滑 L-BFGS 秒 | FISTA 秒 | 延拓秒（单次） |','|---|---:|---:|---:|---:|']
    for name in names:
        vals=[]
        for method in ['lbfgs','direct_lbfgs','fista']:
            r=next(r for r in timing if r['problem']==name and r['method']==method and r['test']=='original'); vals.append(f'{r["median"]:.4f}')
        rr=next(r for r in summary if r['problem']==name and r['method']=='continuation'); vals.append(f'{rr["seconds"]:.4f}')
        lines.append('| '+name+' | '+' | '.join(vals)+' |')
    lines += ['', '本次所有主实验均达到原问题 $10^{-6}$ 的间隙要求。延拓数据只有单次运行，不能据此作稳定的时间优势判断。FISTA 使用简单精确近端，在这些小规模可分问题上具有很强的竞争力。', '',
        '初始实现将 SSN 上限设为 100 步，直接小参数求解曾触发内层上限；检查发现此时线性系统残差已接近机器精度，主要困难是回溯频繁。统一增加到 2000 步后主实验完成。该观察说明小平滑参数的全局阶段可能较慢，局部超线性结论不能保证从任意初始点快速进入局部区域。', '',
        '## 5. 低秩求解成本与平滑误差', '',
        '线性代数测试采用同一个正定 BFGS 矩阵和对角 Moreau 曲率，比对 Woodbury 求解与构造稠密矩阵后 Cholesky 求解。每配置预热一次、重复五次，报告中位数。矩阵和曲率对的公共构造不计时；稠密矩阵组装包含在稠密求解时间中。存储仅统计度量表示，不是程序峰值内存。', '',
        '| 维度 | 记忆 | 紧凑毫秒 | 稠密毫秒 | 比值 | 方向相对差 |','|---:|---:|---:|---:|---:|---:|']
    for r in compact:lines.append(f'| {r["n"]} | {r["memory"]} | {r["compact_seconds"]*1000:.3f} | {r["dense_seconds"]*1000:.3f} | {r["dense_seconds"]/r["compact_seconds"]:.1f} | {r["relative_error"]:.2e} |')
    lines += ['', '该结果支持“基矩阵可廉价求解时，低秩结构降低单次线性系统成本”。它不证明任意近端算子都具有这一优势，也不直接给出端到端加速比。', '',
        r'平滑偏差另用有闭式解的 $F(x)=\sum_i(a_ix_i^2/2-c_ix_i)+0.2\|x\|_1$ 验证，维度为 100，$a_i$ 在 $[0.3,2]$ 上等距取值。原解为 $x_i^*=\operatorname{sign}(c_i)(|c_i|-0.2)_+/a_i$。若 $|c_i|\le0.2(1+\alpha a_i)$，平滑解为 $c_i/(a_i+1/\alpha)$；否则等于原解。闭式解排除了优化误差的干扰。', '',
        '| alpha | 实际原目标偏差 | 理论上界 | 解距离 |','|---:|---:|---:|---:|']
    for r in bias:lines.append(f'| {r["alpha"]:.0e} | {r["original_gap"]:.3e} | {r["gap_bound"]:.3e} | {r["distance"]:.3e} |')
    lines += ['', '实际偏差均在理论上界内，并随 alpha 减小而下降。该特殊对角问题的距离可能比一般上界更快下降，不能用其经验阶替换论文的一般距离界。', '', '![偏差与成本](results/bias_and_cost.png)', '',
        '## 6. 消融、结论与下一步', '',
        '| 记忆数 | 外层步数 | 秒（单次） | 状态 |','|---:|---:|---:|---|']
    for r in read('memory_summary'):lines.append(f'| {r["memory"]} | {r["iterations"]} | {r["seconds"]:.4f} | {r["status"]} |')
    for meth in ['inner_ssn_ablation','inner_gradient_ablation']:
        r=read(meth); lines+=['',f'- {meth}：{r["status"]}；'+(f'外层 {len(r["history"])-1} 步，耗时 {r["history"][-1]["time"]:.4f} 秒。' if 'history'in r else r.get('reason',''))]
    lines += ['', '内层梯度消融使用同一外层框架及相同的 2000 步内层上限。若返回 inner_failed，表示在这一预算内未满足残差要求，不能把它记作已收敛的速度比较。记忆规模的单次时间只作敏感性观察。', '',
        '结果支持低秩线性求解的计算价值、SSN 对所选子问题的快速局部求解，以及平滑误差控制。结果不足以证明固定记忆 L-BFGS 普遍具有外层超线性收敛，也不足以证明该方法普遍优于 FISTA。数学结论由附录证明建立，数值实验只验证选定实例的行为。', '',
        '后续建议：增加高维稀疏真实数据；加入具有块结构的 group-lasso；单独测量非对角近端导数的基求解成本；开展更多随机实例、较大维度端到端测试以及不同最终原问题精度的比较。新问题应先核验近端计算精度与假设，再作速度比较。', '',
        '## 7. 复现', '',
        '在项目根目录运行 `experiments/run.ps1`。运行环境、依赖版本和数据校验值见 `results/manifest.json`。所有主实验的逐步残差、步长、迭代点与证书保存在 `results/*_fixed_*.json`、`results/*_original_*.json`。重复计时原始值见 `results/repeated_timings.json`；源代码为 `run_experiments.py` 和 `analyze_results.py`。', '',
        '本报告的时间是本机测量值，系统负载和实现会影响结果。实验未使用统计显著性检验，不将单次或三次计时差异扩大为一般性能结论。']
    (ROOT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump('source_hashes',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')})

if __name__=='__main__':
    if '--report-only' not in sys.argv:repeated()
    report()
