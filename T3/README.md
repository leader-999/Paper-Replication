# Artificial Intelligence, Algorithmic Pricing, and Collusion — 独立复现

对象：Calvano, Calzolari, Denicolò, Pastorello (2020), American Economic Review 110(10), 3267–3297，DOI 10.1257/aer.20190623。

这是依据论文独立编写并实际运行的 Python/Numba CPU 模拟，不使用官方 Fortran/R 复现包。先打开 `results/report.html`：图片、统计表和可下载CSV都已嵌入，断网也可查看。

## 已运行的范围与边界

- 正文11幅图、4张表均有对应的独立模拟或策略分析输出。
- 基准1,000次；折现因子、成本非对称、三企业及多数稳健性配置各1,000次；四企业256次。
- 参数网格为7×7，每点64次。**没有完成原文100×100、每点1,000次的完整网格。**
- 重新配对1,000组，关闭探索并继续学习。
- 正文七类稳健性实验已覆盖；没有全量实现或运行在线附录中长记忆、线性需求、Boltzmann、需求非对称、非对称学习参数等所有扩展。
- 若论文没有唯一指定细节，本项目明确声明独立实现选择，包括小幅降价为一个网格、乐观Q初始化为100、随机进入时停业企业不更新Q、随机需求的静态归一化方式等。请勿将这些扩展的数值视为已经与作者代码核对一致。
- 作者随机种子未知，不要求逐数一致。基准结果及差异由运行产生，没有拟合输出到论文参考值。

## 运行

本次实测 Python 3.13.5，支持Numba的CPU环境。无需GPU或Colab。精确库版本另列于 `requirements-lock.txt`；其他Python版本可使用普通requirements让pip选择兼容版本。

```powershell
python -m pip install -r requirements.txt
python validate.py
python run_experiments.py --profile local --threads 10
python run_rematch.py
python make_report.py
```

程序以本目录为根，不依赖启动时的工作目录；每32次保存检查点，按配置和种子哈希区分，重复执行会续跑。所有用于最终报告的目录由 `results/manifest.json` 唯一指定。其他目录可能来自先行速度校准，不属于最终报告统计。

`--profile baseline` 仅运行基准；`--profile paper` 将学习参数网格扩大为100×100，每点1,000次，并把四企业样本提高为1,000。它仍沿用本项目列出的折现因子和稳健性参数集合，**不表示所有在线附录实验均已实现**。完整网格计算量约为本次网格的3,189倍，需自行安排充足运行时间；不要仅切换参数就声称已完成运行。

## 文件

| 文件 | 用途 |
|---|---|
| `model.py` | Logit需求、连续Nash/联合垄断基准、价格网格、初始化 |
| `learning.py` | 并行Q-learning、全局策略稳定判定、重新配对 |
| `run_experiments.py` | 各实验配置、确定性种子、分批检查点 |
| `analyze.py` | 极限周期、论文口径真Q与Q-loss、偏离路径、折现收益 |
| `run_rematch.py` | 循环错位重配、重新学习 |
| `paper_reference.py` | 人工抄录的论文表1–4参考值，与模拟数据分开 |
| `make_report.py` | CSV、SVG/PNG和自包含HTML报告 |
| `validate.py` | 7项数值与行为核验，包含独立纯Python更新对照 |
| `results/runs/*/spec.json` | 完整参数、种子起点、会话数 |
| `results/runs/*/batch_*.npz` | 每种子的策略、终态、收敛期数、历史；基准另存最终Q |
| `results/runs/*/metrics.npz` | 可重新生成的分析缓存 |
| `results/tables` | 每会话数值、偏离记录、论文参考值、分组汇总 |
| `results/figures` | 11幅对应正文图，可编辑文字SVG和PNG预览 |
| `results/key_results.json` | 核心实际统计 |
| `results/validation.json` | 数值验证状态 |
| `results/browser-qa` | HTML桌面、移动端与离线CSV下载核验 |

## 统计和解释

Δ=(每企业平均利润−连续Nash每企业平均利润)/(连续联合垄断每企业平均利润−连续Nash每企业平均利润)。确定性环境取终态所属周期的逐期平均；先周期内平均，再跨独立训练种子平均。误差线如无其他标注，为均值±1.96×跨种子样本标准差/√N，只反映Monte Carlo误差。

真Q按双方固定极限策略的线性方程计算：当前期允许偏离，之后恢复策略。均衡频率和Q-loss依此与论文比较，不等同于允许在未来所有状态持续更改策略的全局最优控制问题。

随机进入环境没有确定性极限周期：训练后冻结策略，预热10,000期，评估200,000期；ρ=.0001时只有约20次切换，所以仍存在市场结构的有限路径抽样误差。随机需求环境的冻结策略转移确定，利润使用冲击期望值。

表2–3只使用对称固定价格会话，严格降价；等于原价格的格子留空。原文在这些格子填写的0不作为实际偏离。原文图8–9展示的策略实例无法由未知种子精确重建，本项目展示按种子顺序第一个固定价格会话，并对全部基准会话报告吸引域统计。

## 来源

- 论文：https://doi.org/10.1257/aer.20190623
- 官方在线附录：https://www.aeaweb.org/articles/materials/13398
- 官方复现包标识：https://doi.org/10.3886/E119462V1 （本项目未使用其代码或原始结果）

请以HTML覆盖清单为准描述完成程度，不能把这份有限网格的独立复现称为原文及全部在线附录的逐值完整复制。
