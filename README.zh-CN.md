> **v2.0.0 更新：**已新增统一原问题原始—对偶间隙下的 192 次实验，171 次达标、21 次未达标。全部失败保留。Continuation 在双方均成功的 10 组中有 9 组更快，但未显示相对 R-FISTA 的总体优势；标量度量在一组消融中也更快。下文旧实验作为历史档案保留。
>
> 最新内容：[运行说明](docs/VALUE_BENCHMARKS.md)、[完整结果](docs/review/EXPERIMENT_RESULTS.md)、[理论审查](docs/review/THEORY_AUDIT.md)、[文献对照](docs/review/LITERATURE.md)、[原始记录与图表](paper_results/value_review)。

# MEL-BFGS 可复现实验

[English](README.md) · [正式结果](docs/RESULTS.zh-CN.md) · [数据说明](docs/DATA.md)

本仓库对应论文中的嵌套算法：固定 Moreau-envelope 平滑参数，在外层构造
变量度量子问题，在内层用半光滑牛顿法求解。L-BFGS 紧凑表示用于降低线性求解成本。

## 安装与复现

使用 Python 3.12，创建并激活虚拟环境后运行：

```bash
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py smoke
python reproduce.py figures
python reproduce.py expanded
```

- `verify`：不需要科学计算依赖，核验发布文件校验值、231 条运行记录及残差准则。
- `smoke`：执行数值自检和小规模求解，检查新增的四种算法。
- `figures`：用已发布的真实轨迹重新生成正文四组 PDF/PNG 曲线。
- `expanded`：重新运行正式对照，每个设置预热一次并计时三次。
- `supporting`：重新运行支撑实验；`all` 顺序运行两套实验。

新结果写入 `runs/latest/`，也可用 `--output runs/my-run` 指定独立目录。
正式记录保存在 `paper_results/expanded/`，支撑实验原始记录保存在
`paper_results/supporting.zip`。支撑实验属于较早的独立批次，不应与正式对照混合计时。
WDBC 自动从 UCI 下载并验证校验值，离线放置方式见数据说明。

## 实验结论的范围

固定平滑比较 5 种算法，原问题同精度比较 6 种算法，共涉及 8 种算法。
7 个实例的 77 个设置共计 231 次正式测量，全部满足相应证书要求。
MEL 在本次两类测试中均快于直接 L-BFGS；原问题精度测试中，重启 FISTA 全部最快。
运行时间依赖硬件与环境，不保证在其他机器上得到相同排序。

固定记忆 L-BFGS 本身不能保证外层超线性收敛。该结论仍需论文中的方向一致性、
局部正则性及内层误差条件。曲线用于展示选定实例的行为，不能代替定理证明。

代码和生成结果使用 MIT 许可证，外部数据遵循其原有许可证。
引用格式见英文 README 和 `CITATION.cff`。
