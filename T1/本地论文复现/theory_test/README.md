# A Practical End-to-End Inventory Management Model with Deep Learning：本地复现

## 阅读结果

打开 results/report.html。此HTML内嵌全部报告图像、主要数值表、可下载CSV及完整源码，离线可读。

重要：本项目完成公开复现包支持的子集计算，不能宣称原论文所有训练结果均已复现。论文全部17项图表（图2分a/b）已逐项列入覆盖清单：5项公开子集重算、1项结构重绘、2项原文信息整理、9项缺少必要输入。模型训练结果不会用论文原值或模拟结果补齐。

## 重跑

本机已在Anaconda Python环境完成执行。进入本目录运行：

```console
python -m pip install -r requirements.txt
python reproduce.py
```

也可以：

```console
python reproduce.py --data-zip "inputs/replication_package_MS-FTBDA-20-01585.zip" --output "results"
```

运行将解压包内输入，执行三个作者Notebook、核对全部库存轨迹、重算两种成本口径实地实验、验证理论公式，最后更新HTML和数值文件。

输入ZIP已包含在inputs目录；paper.pdf是用户提供的正式发表版本。无需Colab、Google Drive、网络服务或GPU。安装依赖时可能需要网络。结果生成不下载外部数据。

每次作者Notebook执行记录在results/author_runs的UTC时间戳目录；results/recomputed是最近一次成功运行的快照。重复运行会更新报告及派生结果，不改写输入ZIP。

## 代码

- reproduce.py：本地总入口、两种实地成本口径、公式(11)回归、图表、HTML与完整性记录。
- author_evaluation.py：前期已验证的作者Notebook运行器、离线库存轨迹审计与统计对照。
- theory_checks.py：独立枚举、动态规划与真实销量窗口数值检查；没有模型训练。
- paper_reference.py：人工核对的论文印刷参照值与图表覆盖清单，明确不是运行结果。
- requirements.txt：本次验证环境的精确依赖版本。

## 已有实际计算结果

离线总成本（3000行，2607组SKU；作者代码口径）：

| 方法 | 平均总成本 |
|---|---:|
| OPT（作者字段） | 2572.96 |
| E2E_RNN | 3036.76 |
| LightGBM | 3424.87 |
| BM1 | 3842.31 |
| BM2 | 3837.83 |
| Normal | 3961.69 |
| Gamma | 4099.95 |

实地样本四组各500条，候选对照55333条。按原代码h=1,b=9，实验后处理组/对照组平均总成本分别为884.09 / 1985.148。这是公开子集描述性均值，不是新的因果实验结论。

论文实地成本写为h=12,b=88，但原代码用h=1,b=9，两种口径都已重算并分别标记。公式(11)带截距普通OLS、HC3稳健回归和原作者无截距回归分别保存。重置索引不认证个体关系，修正版DiD不提供显著性。

理论检查：600个引理整数枚举案例、516个独立DP案例、300个真实销量窗口折点检查，共1416项通过。受控小案例只用于实现检查，不作为原论文实证复现或数学证明。

## 缺失的原实验

- Figure 2(a)：原图的SKU对应、日期轴、10/50/90%分位数预测不在包内。
- Tables B.2–B.6、Figures B.1–B.3：完整训练集、132维特征构造、训练实现、多种超参数与随机种子模型/预测/权重不在包内。
- Table 1与B.1只是品类/超参数的原文配置整理，不计为重新实验。
- Table 4缺少可认证跨期个体ID与匹配映射，原代码p值仅作代码复算，不认证其推断有效性。

对于每一项缺口，results/evidence/coverage.csv与HTML提供材料清单。论文原表数字在paper_reference_values.json和REFERENCE_ONLY.csv中，绝不与重算结果混淆。

## 结果目录

- results/report.html：主报告。
- results/recomputed/：作者原代码数值、日志、轨迹示例、输入审计、匹配诊断。
- results/table2_recomputed.csv、table3_equation11_recomputed.csv、table4_recomputed.csv：补充重算。
- results/theory_check_summary.csv、theory_numerical_checks.csv：独立数值检验。
- results/figures/：Python生成的SVG与PNG。
- results/evidence/：覆盖清单、图表契约和原文参照值。
- results/manifest.json：输入及源码SHA256、版本、运行耗时。
- results/validation.json、results/browser_qa.json：数值和离线HTML验证。

附录B的固定预测重计价图只是额外分析，不是原文重新训练的b/h敏感性实验；报告中已明确标记。
