# 理论—实现审查记录

结论以数学命题的假设为边界。“证明核查通过”不表示浮点代码满足无限迭代的渐近假设，也不表示结果具有新颖性。

| 结论/对象 | 必要假设 | 当前实现情况 | 核验方式与结论 |
|---|---|---|---|
| Moreau 梯度、近端导数 | proper closed convex h；准确 prox | l1 和非重叠组范数有闭式 prox | 公式核对；组导数有限差分检查通过。一般 h 不能自动获得可计算导数 |
| 原 Lemma 1 紧凑求解 | A≻0、V≻0、可逆小核 | 符号不定小核，解方程不显式求逆 | determinant lemma + Woodbury 证明通过；不是新定理 |
| 子问题正则性 | B 的正上下界；hα 凸且梯度 Lipschitz | 谱界用 QR 和小矩阵特征值检查 | 强凸性、唯一解、Lq 上界均核查通过 |
| Theorem 1 内层全局 | 上述正则性 + 方向下降 + 步长 | 旧版阈值 1e-7 本身不足以保证 mB/Lq 条件；历史 231 条记录实际满足较强条件 | 已逐条比对历史原始日志，无违反。修订为范数残差与 eᵀd≤νdᵀVd 联合可计算测试；重写下降常数和证明 |
| Theorem 2 内层局部 | semismooth；逆有界；θj→0；强 semismooth + θj=O(‖Rj‖) 给二次 | 浮点上限和有限内层停止不保证渐近条件 | 试探点误差、沿射线积分、单位步接受三段证明核查通过；保留条件性，不能用有限曲线“证明” |
| 外层下降引理 | 真残差 ‖rk‖≤ηk‖sk‖，η̄<mB | 新版无绝对残差成功捷径；太紧时报告失败 | convexity at x+s 给 Δ≤−(mB−η̄)‖s‖²；Armijo 有统一步长下界 |
| Theorem 3 全局 | 紧水平集、谱界、内外误差 | ridge loss 满足强凸与 coercivity；线性检查费用计时 | 梯度趋零、聚点最优、距解集趋零证明完整；无唯一性不宣称全序列点收敛 |
| Corollary 1 速率 | 上述条件；强凸条件另加 | μf=τ 已知 | 1/k objective bound 与 strong-convex linear objective / R-linear iterate 证明通过；固定 α 常数有参数依赖 |
| Theorem 4 外层局部 | 局部误差界、f C²、(Bk−H*)sk=o(‖sk‖)、ηk→0 | 固定记忆、跳过和重置均不能推出方向条件 | 唯一性、试探残差、单位步及误差比证明通过。方向条件仍未解决；只保留条件定理 |
| Uniform Moreau error | h 全局有限且 Lipschitz | l1: Lh²=nλ²；组范数: Lh²=Gλ² | 证明通过。指标函数、一般二次 h 不在范围内；不能将 analysis penalty 的 Lipschitz 性误当 cheap prox |
| Theorem 5 continuation | 各阶段确实达到证书、αj,δj→0 | 新测试按原 gap 最终停止，阶段切换另用平滑 gap | 仅阶段数的误差界；不称总计算复杂度，未证明阶段内总工作上界 |
| 新 elimination lemma | B≻0、α>0、h proper closed convex | 自检比较联合模型残差 | 消元及谱映射证明完整；属于直接线性代数，不包装为首次 proximal calculus |
| 新 pointwise/projected certificate | 点上 h 有限；或 strong-convex f + exact prox | 实验仍使用现有 primal–dual gap | 证明完整：pointwise defect；投影点的 subgradient bound。并未把未实验的自适应策略写为结果 |

## 实现中已确认的边界

1. 旧 `nested` 在任何 variant 下都先分配 n×n 的 `dense`。这会破坏有限内存优势的实际存储解释。旧版本作为历史存档保留，新入口只在明确的 PN/MEL-PN 基线分配稠密矩阵。
2. 正曲率对不是统一谱界。新旧实现都需显式谱检查；计算成本包含在求解器时间内。重建和 QR 的成本为 O(nℓ²+ℓ³)，不能只报两循环 O(nℓ)。
3. 旧版可以因线性残差转稠密求解，日志没有单独回退计数，不能推断历史回退次数为零。新版明确禁止这种回退，失败单列。
4. 新版 SSN 核查实际 norm residual 及 directional residual；局部 θ→0 的无限序列条件仍不由固定浮点门限保证。报告最大实际值不等于证明渐近性质。
5. 旧绝对 floor 虽未使历史 1038 个返回方向违反相对条件，但未来算例没有这种保证。新版低于数值分辨率或内层预算时记录失败。
6. Armijo 舍入裕量只为浮点实现服务。精确算术证明不能自动覆盖任意固定裕量造成的无穷序列；有限实验以最终原始—对偶证书核验。
7. 组范数的分块径向/切向公式已自检，支持非重叠组。重叠组、一般 fused lasso、核范数等仍需单独成本分析，没有在本稿中泛化。
8. 新直接 PQN/PN 是公开、确定性的标准复合模型实现，使用 proximal-residual SSN 和 proximal-gradient safeguard；不是任何作者官方实现的复刻。其近端子问题精度通过可构造 subgradient residual 检查。

## 仍未解决

- 固定内存与重置同时存在时的外层方向一致性：未证明，不作为算法保证。
- 包括每阶段迭代、α 依赖及全部内层成本的 continuation 总复杂度：未建立。
- 浮点/近似 prox 的严格区间证书与有限精度收敛定理：未建立；当前 gap 是双精度计算。
- 非可分正则项的一般 cheap-base 结论：不成立，须按结构讨论。
- 新结果独立达到原创理论论文要求：当前证据不足；需要实质性额外贡献。
