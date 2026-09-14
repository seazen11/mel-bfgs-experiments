# 文献核验与对照（2026-09-14）

## 已核验的来源

1. Stephen Becker, Jalal Fadili, Peter Ochs (2019), *On Quasi-Newton Forward–Backward Splitting: Proximal Calculus and Convergence*, SIAM Journal on Optimization 29(4), 2445–2481. [作者原文](https://www.mop.uni-saarland.de/pub/BFO19/QuasiNewtonProxCalc_arXiv.pdf)，[出版 DOI](https://doi.org/10.1137/18M1167152)。核查了 Section 3 的 proximal calculus、Section 3.2.2 的 SSN、Sections 4–5 的特定更新保证及 Section 6 的实验。
2. Yongcun Song, Zimeng Wang, Xiaoming Yuan, Hangrui Yue (2026), *A Single-Loop Stochastic Proximal Quasi-Newton Method for Large-Scale Nonsmooth Convex Optimization*, JMLR 27(103), 1–43. [期刊页面](https://jmlr.org/papers/v27/25-0632.html)，[全文](https://jmlr.org/papers/volume27/25-0632/25-0632.pdf)。重点核查 Section 5 的直接子问题、对偶构造、紧凑求解及 Table 1 成本；Section 6 的数据规模。
3. Jason D. Lee, Yuekai Sun, Michael A. Saunders (2014), *Proximal Newton-Type Methods for Minimizing Composite Functions*, SIAM Journal on Optimization 24(3), 1420–1443. [作者全文](https://stanford.edu/group/SOL/multiscale/papers/14siopt-proxNewton.pdf)，[出版 DOI](https://doi.org/10.1137/130921428)。是原始复合模型、inexactness 和局部 Newton 型分析的直接先行工作。
4. Katya Scheinberg, Xiaocheng Tang (2016), *Practical Inexact Proximal Quasi-Newton Method with Global Complexity Analysis*, Mathematical Programming 160, 495–529. [作者发表列表](https://coral.ise.lehigh.edu/katyas/publications/)，[作者预印本](https://arxiv.org/abs/1311.6547)。不把迭代复杂度简单当作包含任意内层求解的总耗时。
5. Shida Wang, Jalal Fadili, Peter Ochs (2025), *Quasi-Newton Methods for Monotone Inclusions: Efficient Resolvent Calculus and Primal-Dual Algorithms*, SIAM Journal on Imaging Sciences 18(1), 308–344. [官方原文页面](https://epubs.siam.org/doi/abs/10.1137/24M1646662)。较一般的 resolvent calculus 与 primal–dual 方法是与非可分结构相关的后续研究。
6. Shida Wang, Jalal Fadili, Peter Ochs (2024), *Global non-asymptotic super-linear convergence rates of regularized proximal quasi-Newton methods on non-smooth composite problems*. [arXiv:2410.11676v2](https://arxiv.org/abs/2410.11676v2)。这里按已核实的预印本身份引用，不虚构正式出版信息；它使用不同的正则化 SR1 算法，其速率不能移植到本文的固定内存 BFGS。

## 对照表

成本中 n 为变量维数，r 为低秩列数，Tprox 为基础近端计算成本。O(n) 的固定内存说法隐藏了内存、内层迭代和回溯依赖。

| 工作 | 问题类别 | 平滑机制 | 曲率 | 子问题求解 | 单步成本的边界 | 收敛保证 | 实验范围 |
|---|---|---|---|---|---|---|---|
| Lee–Sun–Saunders | 凸 smooth + prox-friendly nonsmooth | 保留原始 h | 精确或近似 Hessian | 受控不精确复合模型 | 依赖 Hessian 与内层算法，不能统一化为 O(n) | 全局与条件性局部 Newton 型结果 | 统计学习等复合问题 |
| Scheinberg–Tang | 复合凸优化，包含稀疏问题 | 原始 h | L-BFGS 等 | 近似子问题，可含坐标下降 | 内层精度有实际作用 | 带明确 inexactness 的全局分析 | 稀疏优化；非本文平滑算法 |
| Becker–Fadili–Ochs | 凸复合问题及可处理约束 | 变度量 prox 的等价计算 | 简单矩阵 ± 低秩；特定 SR1/BFGS 更新 | 低维根方程，可用 SSN | 依赖 Tprox、r、Jacobian 及求根次数 | 特定算法全局保证；根求解局部结果 | LASSO、约束及块正则例子 |
| Song 等 | 有限和非光滑凸问题 | 等式分裂下的对偶平滑表示，不改变原始模型 | 随机 L-BFGS，variance reduction | 对偶 SSN + 辅助矩阵 | Table 1 区分 O(ιmn) 乘法和 O(ιm²n) 加法；还需采样和更新 | 随机外层线性结果，条件与本稿不同 | Logistic；真实稀疏数据，合成维数到 10⁶ |
| Wang–Fadili–Ochs 2025 | 单调包含及 saddle point | resolvent 重写 | 低秩准牛顿 | 低秩 resolvent / PDHG | 依赖基础 resolvent，不承诺一般非可分项 O(n) | 收敛；强单调情形速率 | 图像处理 |
| Wang–Fadili–Ochs 2024 | 非光滑复合问题 | cubic/gradient regularization | SR1 | 正则化近端子问题 | 与正则化子问题成本相关 | 特定算法的非渐近超线性结果 | 机器学习示例；并非固定内存 MEL |
| 本文修订 | 确定性凸复合；实验为 ridge-logistic + l1/非重叠组范数 | 原始 h 被 hα 替代，有偏差 | f 的紧凑 BFGS，显式谱检查 | 光滑模型 SSN；可消元为另一度量 prox | O(nr²+r³) 加基础求解、梯度、谱检查、回溯；失败显式记录 | 固定 α 全局；内层局部；外层方向条件下局部；阶段误差界 | 历史七例 + 新维数/参数/组结构实验；不声称覆盖百万维稀疏情形 |

## 等价性与贡献判断

令 v=x−B⁻¹g。联合问题为 h(z)+½(w−v)ᵀB(w−v)+‖w−z‖²/(2α)。
固定 z 消去 w 得到 M=(B⁻¹+αI)⁻¹，z=prox_h^M(v)，再恢复 w=(I+αB)⁻¹(z+αBv)。
因此子问题属于已有变度量近端计算范围。外层仍有区别：中心不是 x−M⁻¹g，更新点也不是 z。
Song 等对偶中的参数用于等式约束下的二次项转移；不能因为也出现 Moreau identity 就认定与本文同一平滑算法。

目前最有依据的主线是**平滑策略的结构解释、可核查误差控制及计算适用边界**。
谱映射与点后验证书是可证明性质，但推导直接，不足以自动构成强理论创新。
紧凑 BFGS、Woodbury、SSN 局部速率和条件性外层拟牛顿速率均已有基础。
新增证据若不能显示端到端价值，应明确定位为分析与计算评估，而非新的通用高效求解器。

## 随机对照的边界与可执行后续方案

本轮没有把自行实现的确定性 PQN 标注为 Song 等算法，也没有报告其未运行的随机结果。
后续先获取作者实现及许可证，固定相同训练目标和正则化尺度，用独立调参实例选择步长/批量。
以至少五个采样种子运行，计入全梯度刷新、Hessian-vector products、子问题和证书计算；同时报告总样本梯度数及墙钟时间。
最终成功必须经全数据原问题 primal–dual gap 验证。若证书只按 epoch 计算，各算法统一检查时点，计入检查费用并说明可能的停止延迟。
不拿随机的单次幸运轨迹与确定性的三次中位时间比较。
