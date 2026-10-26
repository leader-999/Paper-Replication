"""Manually transcribed published values. These are references, NOT reproduced results.

Source: user-supplied published PDF, Management Science 69(2), pp.759-773.
Figures and tables were visually checked against rendered pages.
"""
METRICS = ['holding_cost','stockout_cost','total_cost','turnover','stockout_ratio']
FIG2 = {
 'OPT':[3022.60,1925.70,1096.91], 'E2E_RNN':[3766.52,2689.02,1077.50],
 'Bm1':[4207.09,2502.55,1704.54], 'Bm2':[4157.03,2254.99,1902.04],
 'Normal':[4576.05,3369.21,1206.84], 'Gamma':[4476.17,2821.87,1654.30],
 'gbm':[4017.82,2109.93,1907.89],
}
TABLE2 = {
 'treatment':[598.20,496.34,1094.54,18.59,.17],
 'control':[809.54,1027.93,1837.47,20.39,.26],
 'reported_change_percent':[-26.1,-51.7,-40.4,-8.8,-34.6],
 'reported_p':['<0.001','<0.001','<0.001','0.0102','<0.001'],
}
TABLE3 = {'coefficient':[-211.33,-531.53,-742.86,-1.16,-.09],
 'reported_p':['<0.001','<0.001','<0.001','0.007','<0.001']}
TABLE4 = {
 'treatment_pre':[688.13,880.50,1568.63,16.41,.24],
 'treatment_post':[598.20,496.34,1094.54,18.59,.17],
 'control_pre':[786.68,887.64,1674.32,16.40,.22],
 'control_post':[809.54,1027.93,1837.47,20.39,.26],
 'reported_did':[-112.79,-524.45,-637.24,-1.81,-.11],
 'reported_p':['<0.001','<0.001','<0.001','0.041','0.044'],
}
CATEGORIES = {
 'Tea set':['Tea set combination','Tea cup','Tea kettle','Tea tray','Tea can','Tea bowl',
             'Tea accessories','Tea-ware decoration','Coffee set','Tea travel set'],
 'Pastry essentials & seasoning':['Baking supplies','Flour','Mixed grains','Rice','Oil',
                                  'Seasonings','Dry foods','Convenience foods'],
}
HYPERPARAMETERS = [
 ['Demand module hidden state',50,'30, 40, 50, 60'],
 ['VLT MLP layer sizes','50, 20','100/20; 50/20; 30/20'],
 ['Integration hidden layers','100, 100','100; 100/100; 100/100/100; 100/100/100/100'],
 ['Learning rate',.001,'.0001, .001, .01'],['Learning rate decay',.0001,'1e-5, 1e-4, 1e-3'],
 ['Momentum',.85,'.8, .85, .9'],['Batch size',64,'64, 128, 256'],
 ['Initialization','Gaussian mean=0, SD=.01','Gaussian; Uniform'],
 ['Activation','ReLU','ReLU; tanh'],['Dropout',.2,'.1, .2, .3, .4'],
]
# Costs: [total, holding, stockout]. All are printed reference values only.
APPENDIX = {
 'Table B.2':{'page':771,'factor':'hidden_layers','rows':[
  ['OPT',3022.60,1925.70,1096.91],['1',4857.20,3919.17,938.03],
  ['2',3766.52,2689.02,1077.50],['3',3822.69,2453.32,1369.38],['4',3904.74,2231.01,1673.73]]},
 'Table B.3':{'page':771,'factor':'hidden_state','rows':[
  ['OPT',3022.60,1925.70,1096.91],['30',3846.28,2886.45,959.83],
  ['40',3783.32,2710.71,1072.61],['50',3766.52,2689.02,1077.50],['60',3860.99,2185.18,1675.81]]},
 'Table B.4':{'page':771,'factor':'learning_rate','rows':[
  ['OPT',3022.60,1925.70,1096.91],['0.0001',3744.28,2424.71,1319.58],
  ['0.001',3766.52,2689.02,1077.50],['0.01',4231.01,2350.98,1880.03]]},
 'Table B.5':{'page':771,'factor':'training_fraction',
  'columns':['percent','training_SKUs','OPT_cost','E2E_cost','training_seconds'],
  'rows':[[20,3893,3022.60,5608.01,281.61],[40,7786,3022.60,4765.52,624.64],
          [60,11680,3022.60,4226.84,1022.57],[80,15573,3022.60,3783.54,1459.42],
          [100,19466,3022.60,3766.52,1830.36]]},
 'Table B.6':{'page':773,'factor':'initialization_seed','rows':[
  ['OPT',3022.60,1925.70,1096.91],['network1',3766.52,2689.02,1077.50],
  ['network2',3800.54,2766.67,1033.87]]},
}

# One status per numbered visual item; Figure 2 is split into its two result panels.
COVERAGE = [
 ['Figure 1',764,'schematic','结构重绘','网络模块和连接可按图重画；不是训练所得模型。','隐藏维数部分未披露；训练输入缺失'],
 ['Figure 2(a)',766,'missing','无法数值复现','包内只有单条预测序列，没有10%、50%、90%分位数预测和原图三个SKU对应关系。','原图SKU、时间戳、需求历史与多分位数预测'],
 ['Figure 2(b)',766,'subset','公开子集重算','执行作者库存评估，含7种方法；样本不同，不能逐格追平论文。','完整24,333-SKU实验测试数据'],
 ['Figure 3',767,'subset','公开子集重算','重算倾向得分及作者协变量近邻匹配，提供平衡性诊断。','完整6,097条处理组及原匹配索引'],
 ['Table 1',767,'reference','原文信息整理','列出18个品类；公开样本没有品类列，不能验证分组覆盖。','品类与SKU-DC映射'],
 ['Table 2',768,'subset','公开子集重算','四组各500条数据，输出原代码及论文成本口径的描述统计和t检验。','完整实验数据与统一观察窗口'],
 ['Table 3',768,'subset','公开子集重算','保留作者无截距OLS，另按公式(11)加入截距，并提供HC3对照。','完整实验数据'],
 ['Table 4',769,'subset','点估计重算；推断受限','提供四组均值DiD；原代码p值单列，不能认证为有效面板推断。','真实个体对应、时间结构与匹配映射'],
 ['Table B.1',771,'reference','原文配置整理','导出完整默认超参数和搜索范围，未声称已重新训练。','完整特征定义、训练集及训练流程'],
 ['Table B.2',771,'missing','无法重新训练复现','每种网络层数的模型和测试输出均未提供。','训练集、源码、不同深度模型'],
 ['Table B.3',771,'missing','无法重新训练复现','不同隐藏状态大小的训练输出缺失。','训练集、源码、不同宽度模型'],
 ['Table B.4',771,'missing','无法重新训练复现','不同学习率的训练输出缺失。','训练集、优化器细节与各模型'],
 ['Table B.5',771,'missing','无法重新训练复现','训练数据分片和计时条件缺失。','完整训练数据、划分、原硬件计时条件'],
 ['Figure B.1',772,'missing','无法重新训练复现','对应Table B.5，不能用已知论文数值冒充运行结果。','不同训练规模的真实运行记录'],
 ['Figure B.2',772,'missing','无法按原实验复现','各b/h下需要重新生成标签并训练；现有固定预测只能做额外重计价。','不同成本比下标签、模型和输出'],
 ['Figure B.3',772,'missing','无法复现权重图','没有两个训练网络的VLT_layer1权重。','两个随机种子的模型检查点'],
 ['Table B.6',773,'missing','无法重新训练复现','只提供单个E2E模型的预测，没有第二种初始化模型。','训练集及两个种子的模型与测试输出'],
]
