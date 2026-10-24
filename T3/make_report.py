"""Compute source tables and export an entirely offline, standalone HTML report."""
from pathlib import Path
import json,base64,html,platform,sys,hashlib
import numpy as np,pandas as pd,numba
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from model import Config,make_environment
from analyze import metrics,cycle,response,deviation_tables,cycle_metrics
from run_experiments import ROOT,load
import paper_reference as ref

OUT=ROOT/'results';FIG=OUT/'figures';TABLE=OUT/'tables'
FIG.mkdir(exist_ok=True);TABLE.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.labelcolor':'#263644','axes.edgecolor':'#81929c','xtick.color':'#526471','ytick.color':'#526471','legend.frameon':False})
BLUE='#356f97';ORANGE='#c6763c';TEAL='#458b85'
downloads={};figures={};captions={}
def export_table(df,name,index=True):
    text=df.to_csv(index=index,float_format='%.9g',lineterminator='\n');(TABLE/(name+'.csv')).write_bytes(b'\xef\xbb\xbf'+text.encode('utf-8'));downloads[name+'.csv']=text
    return df.to_html(border=0,classes='data',float_format=lambda v:f'{v:.4f}',na_rep='—',index=index)
def save(fig,name,caption):
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    figures[name]=base64.b64encode((FIG/(name+'.svg')).read_bytes()).decode();captions[name]=caption
def figure(name):return f'<figure><img alt="{html.escape(captions[name])}" src="data:image/svg+xml;base64,{figures[name]}"><figcaption>{html.escape(captions[name])}</figcaption></figure>'
def band(ax,t,values,label,color):
    center=np.nanmean(values,axis=0);se=np.nanstd(values,axis=0,ddof=1)/np.sqrt(values.shape[0]);ax.plot(t,center,label=label,color=color);ax.fill_between(t,center-1.96*se,center+1.96*se,color=color,alpha=.15)
def tab1(data,met):
    fixed=met['length']==1;sym=met['symmetric'];eq=met['equilibrium'];masks=[fixed&sym,fixed&~sym,fixed,met['length']==2,met['length']>=3,np.ones(len(eq),bool),eq]
    rows=[]
    for mask in masks:
        if not mask.any():rows.append([0]+[np.nan]*7);continue
        rows.append([mask.mean(),met['gain'][mask].mean(),met['gain'][mask].std(ddof=1),eq[mask].mean(),met['on_loss'][mask].mean(),met['on_loss'][mask].std(ddof=1),met['off_loss'][mask].mean(),met['off_loss'][mask].std(ddof=1)])
    return pd.DataFrame(np.array(rows).T,index=ref.TABLE1.index,columns=ref.TABLE1.columns)

def main():
    numba.set_num_threads(6);manifest=json.loads((OUT/'manifest.json').read_text());runs={};summary=[]
    for item in manifest:
        name=item['name'];folder=ROOT/item['path'];spec,data=load(folder);cfg=Config(**spec['config']);env=make_environment(cfg)
        cache=folder/'metrics.npz'
        if cache.exists():met=dict(np.load(cache))
        else:
            met=metrics(env,data,name=='baseline' or name.startswith('grid_'));np.savez_compressed(cache,**met)
        valid=data['converged'];g=met['gain'][valid];se=g.std(ddof=1)/np.sqrt(len(g)) if len(g)>1 else np.nan
        summary.append(dict(experiment=name,sessions=len(g),requested=len(valid),convergence=valid.mean(),gain=g.mean() if len(g) else np.nan,sd=g.std(ddof=1) if len(g)>1 else np.nan,mcse=se,ci_low=g.mean()-1.96*se if len(g) else np.nan,ci_high=g.mean()+1.96*se if len(g) else np.nan,mean_periods=data['steps'].mean(),max_periods=data['steps'].max(),alpha=cfg.alpha,beta=cfg.beta))
        runs[name]=(env,data,met)
        # Session-level source data contains no published reference numbers.
        df=pd.DataFrame({'seed':data['seeds'],'converged':valid,'periods':data['steps'],'cycle_length':met['length'],'gain':met['gain']})
        for i in range(cfg.n):df[f'profit_{i+1}']=met['profits'][:,i]
        if 'equilibrium' in met:
            for key in ['equilibrium','on_loss','off_loss']:df[key]=met[key]
        export_table(df,name+'_sessions',False)
        print('Analyzed',name,flush=True)
    totals=pd.DataFrame(summary);summary_html=export_table(totals,'experiment_summary',False)
    env,data,met=runs['baseline'];cfg=env['config'];basegain=met['gain'];n=len(basegain)
    own1=tab1(data,met);table1=export_table(own1,'table1_reproduced');paper1=export_table(ref.TABLE1,'table1_published');diff1=export_table(own1-ref.TABLE1,'table1_difference')
    gridrows=[]
    for name,(e,d,r) in runs.items():
        if name.startswith('grid_'):
            irf=np.array([response(e,p,s,periods=2) for p,s in zip(d['policies'],d['states'])]);ratio=irf[:,2,1]/irf[:,0,1]
            gridrows.append(dict(alpha=e['config']['alpha'],beta=e['config']['beta'],gain=r['gain'].mean(),equilibrium=r['equilibrium'].mean(),response_ratio=ratio.mean(),sessions=len(r['gain'])))
    grid=pd.DataFrame(gridrows);export_table(grid,'grid_sources',False)
    for name,col,title in [('figure01','gain','Mean profit gain Δ'),('figure02','equilibrium','On-path equilibrium frequency'),('figure07','response_ratio','Nondeviator price at t=2 / pre-price')]:
        matrix=grid.pivot(index='alpha',columns='beta',values=col);fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
        mesh=ax.pcolormesh(matrix.columns.values*1e5,matrix.index.values,matrix.values,shading='nearest',cmap='viridis');fig.colorbar(mesh,ax=ax,label=title);ax.set(xlabel='Exploration decay β × 10⁵',ylabel='Learning rate α',title=title)
        save(fig,name,f'{name}: 7×7 个独立训练参数点，每点64次；未插值成原文100×100网格。颜色表示{title}。')
    deltarows=[]
    for name,(e,d,r) in runs.items():
        if name=='baseline' or name.startswith('delta_'):
            g=r['gain'][d['converged']];deltarows.append([e['config']['delta'],g.mean(),g.std(ddof=1)/np.sqrt(len(g)),len(g)])
    delta=pd.DataFrame(deltarows,columns=['delta','gain','mcse','sessions']).sort_values('delta');export_table(delta,'figure03_sources',False)
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');ax.errorbar(delta.delta,delta.gain,yerr=1.96*delta.mcse,color=BLUE,marker='o',capsize=3);ax.set(xlabel='Discount factor δ',ylabel='Profit gain Δ',title='Patience and learned cooperation');ax.axhline(0,color='#aaa',lw=.7);save(fig,'figure03','图3：每个折现因子1,000次训练；误差线为跨独立种子的近似95% Monte Carlo置信区间。')
    irf=np.array([response(env,p,s) for p,s in zip(data['policies'],data['states'])]);small=np.array([response(env,p,s,mode='small',periods=15) for p,s in zip(data['policies'],data['states'])])
    for name,arr in [('figure04',irf),('figure06',small)]:
        t=np.arange(arr.shape[1]);fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');band(ax,t,arr[:,:,0],'Forced deviator',ORANGE);band(ax,t,arr[:,:,1],'Other agent',BLUE);ax.axhline(env['pn'][0],ls='--',color='#9aa6ad',label='Nash');ax.axhline(env['pm'][0],ls=':',color='#6d7b85',label='Monopoly');ax.set(xlabel='Periods after intervention (t=1)',ylabel='Price');ax.legend(ncol=2,fontsize=9)
        save(fig,name,('图4：强制企业1在t=1选择对手冲击前价格的静态最优反应。' if name=='figure04' else '图6：小幅偏离明确定义为冲击前价格下移一个网格；正文未给出唯一幅度，属于独立重构。')+'1,000次会话，周期会话先对各相位平均；阴影为均值的95% Monte Carlo区间。')
        export_table(pd.DataFrame({'t':t,'deviator_mean':arr[:,:,0].mean(axis=0),'other_mean':arr[:,:,1].mean(axis=0),'deviator_mcse':arr[:,:,0].std(axis=0,ddof=1)/np.sqrt(n),'other_mcse':arr[:,:,1].std(axis=0,ddof=1)/np.sqrt(n)}),name+'_sources',False)
    fixed=met['length']==1;changes=irf[fixed,:11]-irf[fixed,0,None,:];t=np.arange(11);fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained');dist=[]
    for i,ax in enumerate(axes):
        vals=changes[:,:,i];q=np.quantile(vals,[0,.25,.75,1],axis=0);ax.vlines(t,q[0],q[3],color='#aab4ba',lw=1,linestyle='--');ax.fill_between(t,q[1],q[2],step='mid',alpha=.3,color=BLUE);ax.plot(t,vals.mean(axis=0),color='#253c4b');ax.axhline(0,color='#999',lw=.5);ax.set(xlabel='Time',ylabel='Price minus pre-price',title=['Forced deviator','Other agent'][i])
        for k in range(11):dist.append(dict(agent=i+1,t=k,mean=vals[:,k].mean(),minimum=q[0,k],q25=q[1,k],q75=q[2,k],maximum=q[3,k]))
    save(fig,'figure05',f'图5：仅固定价格的{fixed.sum()}次会话；线为均值，色带为25–75分位，虚线为最小—最大范围。');export_table(pd.DataFrame(dist),'figure05_sources',False)
    deviations=pd.DataFrame(deviation_tables(env,data,met));export_table(deviations,'deviations_raw',False);table23={}
    for label,col in [('2A','nondev_change'),('2B','dev_change'),('3A','discount_gain'),('3B','unprofitable')]:
        matrix=deviations.pivot_table(index='pre_index',columns='dev_index',values=col,aggfunc='mean').reindex(columns=np.arange(0,14,2))
        counts=deviations[['session','pre_index']].drop_duplicates().groupby('pre_index').size()
        matrix.insert(0,'Frequency',counts.reindex(matrix.index)/counts.sum())
        matrix.index=[round(env['grids'][0,int(i)],4) for i in matrix.index];matrix.columns=['Frequency']+[round(env['grids'][0,i],4) for i in np.arange(0,14,2)]
        table23[label]=export_table(matrix,'table'+label+'_reproduced')
        export_table(getattr(ref,'TABLE'+label),'table'+label+'_published')
    # Graph examples: first fixed-price session in seed order, no screening for shape.
    rep=int(np.flatnonzero(fixed)[0]);p=data['policies'][rep];S=p.shape[1];m=15;transition=p[0].astype(int)*m+p[1];G=nx.DiGraph();G.add_nodes_from(range(S));G.add_edges_from(enumerate(transition.tolist()));central=nx.betweenness_centrality(G)
    attractor=set(cycle(p,data['states'][rep],m,1).tolist());nodegain=(env['rewards'][0].mean(axis=1)-env['normalizer_n'])/(env['normalizer_m']-env['normalizer_n']);sizes=np.array([10+700*central[x] for x in range(S)])
    pos=nx.spring_layout(G,seed=20261021,iterations=250);fig,ax=plt.subplots(figsize=(7.2,5.8),layout='constrained');nx.draw_networkx_edges(G,pos,ax=ax,arrows=True,arrowsize=7,width=.4,alpha=.45,edge_color='#78909c');nx.draw_networkx_nodes(G,pos,nodelist=list(range(S)),ax=ax,node_size=sizes,node_color=nodegain,cmap='Blues',edgecolors='white',linewidths=.2);nx.draw_networkx_nodes(G,pos,nodelist=list(attractor),node_shape='s',node_color=ORANGE,node_size=90,ax=ax);ax.set_axis_off();ax.set_title(f'First fixed-price session: seed {int(data["seeds"][rep])}');save(fig,'figure08',f'图8：按种子排序第一个固定价格会话（编号{rep}），225状态全部显示；方块为终态，圆大小表示介数中心性，颜色深浅表示该状态价格的利润增益。布局独立生成。')
    gridp=env['grids'][0];x,y=np.meshgrid(gridp,gridp,indexing='ij');xx=x.ravel();yy=y.ravel();nxp=gridp[p[0]];nyp=gridp[p[1]];fig,ax=plt.subplots(figsize=(6.5,5.5),layout='constrained');ax.quiver(xx,yy,nxp-xx,nyp-yy,color='#78909c',angles='xy',scale_units='xy',scale=1,width=.0025,alpha=.65);ax.scatter(xx,yy,c=nodegain,cmap='Blues',s=12+sizes/3,edgecolors='white',linewidths=.3)
    for s in attractor:ax.scatter(xx[s],yy[s],marker='s',s=80,color=ORANGE,zorder=5)
    ax.set(xlabel='Previous price, firm 1',ylabel='Previous price, firm 2',title='State-to-state price transitions');save(fig,'figure09','图9：与图8完全相同的会话及策略；箭头为前期价格到本期价格的确定性映射，展示225个状态。')
    graphsource=pd.DataFrame({'state':np.arange(S),'next_state':transition,'price1':xx,'price2':yy,'next_price1':nxp,'next_price2':nyp,'betweenness':[central[s] for s in range(S)],'gain':nodegain});export_table(graphsource,'figures08_09_sources',False)
    basins=[]
    for r,pol in enumerate(data['policies']):
        target=set(cycle(pol,data['states'][r],m,1).tolist());tr=pol[0].astype(int)*m+pol[1];reached=[];distance=[]
        for initial in range(S):
            seen=set();s=initial;k=0
            while s not in target and s not in seen:seen.add(s);s=int(tr[s]);k+=1
            reached.append(s in target)
            if s in target:distance.append(k)
        basins.append(dict(session=r,all_reach=all(reached),non_reaching=S-sum(reached),mean_distance=np.mean(distance),max_distance=max(distance)))
    basin=pd.DataFrame(basins);export_table(basin,'basin_statistics',False)
    time=np.arange(data['histories'].shape[1])*1000;history=(data['histories']-env['normalizer_n'])/(env['normalizer_m']-env['normalizer_n']);eps=np.exp(-cfg['beta']*time);below=np.flatnonzero(env['grids'][0]<=env['pn'][0])[-1];pay=env['rewards'][0].mean(axis=1).reshape(m,m);benchmark=[]
    for ep in eps:
        prob=np.full(m,ep/m);prob[below]+=1-ep;benchmark.append((prob@pay@prob-env['normalizer_n'])/(env['normalizer_m']-env['normalizer_n']))
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');band(ax,time,history,'Q-learning',BLUE);ax.plot(time,benchmark,ls='--',color=ORANGE,label='Nash + random exploration');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.set(xlabel='Training periods',ylabel='Profit gain Δ');ax.legend();save(fig,'figure10','图10：1,000会话，每1,000期记录最近100期均值；阴影为跨种子的95% Monte Carlo区间。收敛早于1,500,000期的会话，余下历史用冻结策略补齐。虚线为解析的Nash向下取整加随机探索基准。');export_table(pd.DataFrame({'period':time,'gain':history.mean(axis=0),'mcse':history.std(axis=0,ddof=1)/np.sqrt(n),'exploration_benchmark':benchmark}),'figure10_sources',False)
    rem=dict(np.load(OUT/'rematch.npz'));rh=(rem['histories']-env['normalizer_n'])/(env['normalizer_m']-env['normalizer_n']);t=np.arange(rh.shape[1])*100;fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');band(ax,t,rh,'Rematched agents',BLUE);ax.axhline(basegain.mean(),color='#97a4ad',ls='--',label='Original limit gain');ax.set(xlabel='Periods after rematching',ylabel='Profit gain Δ');ax.legend();save(fig,'figure11','图11：1,000组循环错位重新配对，关闭随机探索、继续Q更新；每100期记录最近100期均值。初始状态随机。阴影为95% Monte Carlo区间。');export_table(pd.DataFrame({'period':t,'gain':rh.mean(axis=0),'mcse':rh.std(axis=0,ddof=1)/np.sqrt(n)}),'figure11_sources',False)
    costrows=[]
    for c in [1,.875,.75,.625,.5,.25]:
        ee,dd,rr=runs['baseline' if c==1 else f'cost_{c}'];pn=ee['profit_n'];ratio=(rr['profits'][:,0]/pn[0])/(rr['profits'][:,1]/pn[1]);costrows.append([c,ee['nash_share'][1]/ee['nash_share'].sum(),rr['gain'].mean(),ratio.mean()])
    cost=pd.DataFrame(costrows,columns=ref.TABLE4.columns);table4=export_table(cost,'table4_reproduced',False);paper4=export_table(ref.TABLE4,'table4_published',False);costdiff=cost.copy();costdiff.iloc[:,1:]=cost.iloc[:,1:]-ref.TABLE4.iloc[:,1:];diff4=export_table(costdiff,'table4_difference',False)
    robust=totals[~totals.experiment.str.startswith(('grid_','delta_'))].copy()
    reference={'baseline':'0.849','players_3':'≈0.64','players_4':'≈0.56','shock_0.15':'≈0.80','shock_0.25':'≈0.70','mu_0':'>0.77','init_nash':'>0.70','init_grim':'>0.70','init_optimistic':'接近1','lower_cost':'>0.60','actions_100':'≈0.70'}
    robust['published_reference']=robust.experiment.map(reference).fillna('正文未报单一精确数值')
    robust.loc[robust.experiment.str.startswith('entry_'),'published_reference']='<0.60（依赖实现口径）'
    robust.loc[robust.experiment=='entry_3_0.001','published_reference']='0.583（官方附录A9）'
    robust.loc[robust.experiment=='entry_3_0.0001','published_reference']='0.566（官方附录A9）'
    robust_html=export_table(robust[['experiment','sessions','gain','ci_low','ci_high','published_reference','convergence','mean_periods']],'robustness_results',False)
    coverage=[]
    for i in range(1,12):coverage.append(dict(item=f'Figure {i}',status='实际模拟/分析已生成',scope='7×7×64，较原文缩减' if i in [1,2,7] else '独立模拟；见该图方法与局限'))
    for i in range(1,5):coverage.append(dict(item=f'Table {i}',status='实际模拟/分析已生成',scope='1,000次基准/成本实验，含原文参考值'))
    coverage.extend([dict(item='V.A–G robustness',status='正文七类稳健性实验已运行',scope='通常1,000次；四企业256次；随机进入和初始化存在重构选择'),dict(item='Online Appendix additional experiments',status='未全量复现',scope='长记忆、线性需求、Boltzmann、需求非对称、非对称学习参数等不在所附正文结果集内'),dict(item='Original 100×100×1,000 grid',status='未运行完整规模',scope='代码提供 --profile paper；本次7×7×64')]);coverage_html=export_table(pd.DataFrame(coverage),'coverage',False)
    session_count=int(totals.requested.sum())+len(rem['states']);period_count=int(sum(int(d['steps'].sum()) for e,d,r in runs.values()))+int(rem['steps'].sum());mean=float(basegain.mean());se=float(basegain.std(ddof=1)/np.sqrt(n));eq=float(met['equilibrium'].mean())
    results=dict(baseline_gain=mean,baseline_mcse=se,baseline_eq=eq,fixed_price_frequency=float(fixed.mean()),on_loss=float(met['on_loss'].mean()),off_loss=float(met['off_loss'].mean()),sessions=session_count,training_periods=period_count,experiments=len(runs)+1,all_states_reach_frequency=float(basin.all_reach.mean()),fewer_than_three_exceptions=float((basin.non_reaching<3).mean()),deviation_unprofitable=float(deviations.unprofitable.mean()),graph_session=rep,converged=int(totals.sessions.sum()),requested=int(totals.requested.sum()),rematch_converged=int(rem['converged'].sum()),grid_gain_equilibrium_correlation=float(grid[['gain','equilibrium']].corr().iloc[0,1]),grid_gain_punishment_correlation=float(grid.gain.corr(1-grid.response_ratio)))
    (OUT/'key_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    intro=f'''<header><div class="eyebrow">INDEPENDENT CPU REPLICATION · AER 2020</div><h1>人工智能、算法定价与合谋</h1><p class="subtitle">Calvano, Calzolari, Denicolò & Pastorello<br>独立实现 · 实际运行结果 · 正文逐图逐表对照</p></header>
    <aside><strong>复现范围：</strong>本报告生成正文全部11幅图、4张表对应的独立模拟结果，并运行正文七类稳健性实验。<strong>这不等于原文所有结果已逐值、全规模复现。</strong>原文100×100网格每点1,000次，本次为7×7网格每点64次；基准与大多数单因素实验各1,000次，四企业256次。在线附录的全部扩展未运行。没有使用作者代码或将论文数值冒充本次输出。</aside>
    <div class="cards"><div><b>{mean:.4f}</b><span>基准平均利润增益 Δ<br>论文：0.849</span></div><div><b>{eq:.1%}</b><span>路径上均衡频率<br>论文：50.5%</span></div><div><b>{session_count:,}</b><span>实际独立训练/重配会话<br>{period_count:,} 期Q更新</span></div></div>
    <p>基准Δ的近似95% Monte Carlo区间为 [{mean-1.96*se:.4f}, {mean+1.96*se:.4f}]；固定价格比例 {fixed.mean():.1%}，论文64.3%。路径上Q-loss为 {met['on_loss'].mean():.4f}（论文0.002），所有状态Q-loss为 {met['off_loss'].mean():.4f}（论文0.018）。利润增益高度接近原文，均衡频率和个别稳健性结果应按实际差异解释。</p>'''
    methods=f'''<section id="methods"><h2>模型、训练与可复核规则</h2><p>需求 qᵢ = exp((aᵢ−pᵢ)/μ) / [exp(a₀/μ)+Σⱼ exp((aⱼ−pⱼ)/μ)]；利润 πᵢ=(pᵢ−cᵢ)qᵢ。Q←(1−α)Q+α[πᵢ+δ max Q(s′,·)]；独立ε-greedy探索 εₜ=exp(−βt)。基准 n=2、c=1、a=2、a₀=0、μ=.25、δ=.95、α=.15、β=4×10⁻⁶、15价格、一期记忆。</p><p>连续Nash价格 {env['pn'][0]:.9f}，联合垄断价格 {env['pm'][0]:.9f}；15点网格 [{env['grids'][0,0]:.9f}, {env['grids'][0,-1]:.9f}]。Δ=(平均企业利润−πᴺ)/(πᴹ−πᴺ)。Q初始化为对手均匀定价时的期望利润/(1−δ)。并列最优取最低动作，初始状态随机，随机数生成器xorshift64。各实验种子及参数保存在spec.json和CSV中。</p><p>收敛判定：所有状态的贪婪策略连续100,000期不变，最大1,000,000,000期；本次训练收敛 {int(totals.sessions.sum()):,}/{int(totals.requested.sum()):,}。极限结果追踪终态所属的确定性周期，先在周期内平均，再在种子间平均。Q-loss通过固定双方极限策略的线性方程求真Q：只在当前期偏离，以后恢复策略；它不同于允许未来反复偏离的全局最优控制。图2的“均衡”沿用论文这一比较口径。</p><p>Monte Carlo区间：独立会话均值±1.96×跨种子样本标准差/√N，仅描述模拟抽样误差。未使用参数拟合来强制靠近论文结果；缺少作者随机种子意味着不应期待逐数相等。</p></section>'''
    mismatch='<aside><strong>尚未对齐的结果：</strong>随机进入/退出实验的利润增益高于论文所述低于0.60的水平，三企业配置也高于官方附录A9的0.583/0.566。因此该扩展只完成独立实现与运行，尚未复现成功；当前状态、停业时Q更新和利润归一化口径需要进一步与作者实现核对。报告保留原始输出，没有调整数值来掩盖差距。</aside>'
    sections=[intro,mismatch,methods,'<section id="coverage"><h2>结果覆盖清单</h2>'+coverage_html+'</section>','<section id="baseline"><h2>表1 · 基准结果与论文逐项对照</h2><h3>本次模拟</h3>'+table1+'<details><summary>查看论文参考值与差值（本次−原文）</summary>'+paper1+diff1+'</details></section>']
    for i in [1,2,3,4,5,6,7,8,9]:sections.append(f'<section id="fig{i}"><h2>图{i}</h2>'+figure(f'figure{i:02d}')+'</section>')
    sections.append(f'<section><h2>网格相关性与图结构的总体检查</h2><p>本次49点网格的利润增益与路径上均衡频率Pearson相关系数为 {results["grid_gain_equilibrium_correlation"]:.3f}（原文0.12）；利润增益与惩罚幅度的相关系数为 {results["grid_gain_punishment_correlation"]:.3f}（原文0.762）。网格分辨率和每点样本数不同，因此相关系数不应逐值相等。</p><p>从全部225个状态出发均进入终态所属吸引子的会话占 {basin.all_reach.mean():.1%}（原文92%）；例外状态少于3个的会话占 {(basin.non_reaching<3).mean():.1%}（原文98%）。展示会话为按种子顺序第一个固定价格会话，并非作者图8的原会话。每个会话的可达性与路径长度均提供CSV。</p></section>')
    for tab,labels in [(2,['2A','2B']),(3,['3A','3B'])]:
        content=f'<section id="table{tab}"><h2>表{tab} · '+('偏离后的价格变化' if tab==2 else '偏离的折现收益与不盈利频率')+'</h2><p>仅使用对称固定价格会话；按冲击前价格分组，列为每隔一个网格的偏离价格。仅计算严格降价，等于冲击前价格的单元格留空。表中0.01表示1%。每条偏离原始记录、分组均值和论文抄录值可下载；原文的零值占位不参与差异分析。</p>'
        for label in labels:content+=f'<h3>Panel {label[-1]}</h3>'+table23[label]+f'<details><summary>论文 Panel {label[-1]} 参考值</summary>'+getattr(ref,'TABLE'+label).to_html(border=0,na_rep='—',classes='data',float_format=lambda v:f'{v:.2f}')+'</details>'
        sections.append(content+'</section>')
    sections.append('<section id="table4"><h2>表4 · 成本非对称</h2>'+table4+'<p>企业2的Nash份额分母仅含在位企业；相对收益比先在会话内计算再跨会话取平均。</p><details><summary>论文参考值与差值</summary>'+paper4+diff4+'</details></section>')
    sections.append('<section id="robust"><h2>正文V.A–G · 稳健性实际结果</h2>'+robust_html+'<p>不要将每个方向性结论都自动判为成功。若某项明显偏离论文，应结合以下实现选择和抽样规模进一步核对；本表保留全部实际输出。</p><ul><li>随机需求：对每个冲击分别求连续静态基准后平均，作为归一化参照；当前冲击在定价时不可见。</li><li>随机进入：状态记录前期不活跃标记；本期活动状态在选择前按ρ切换，进入者停业时不更新Q。初始化将停业视为对手动作之一。归一化为两种市场结构静态总利润各占50%。冻结策略用200,000期随机评估、10,000期预热；这些为明确的重构选择，尚未与作者实现核对。</li><li>乐观初始化：全部Q=100且ε=0；正文只称“很大的常数”，未给唯一数值。Nash及grim初始化为本代码中明确定义的策略价值。</li><li>小幅偏离采用下移一个价格网格；μ=0按同质商品、最低价格获胜和并列均分处理；下界扩展采用c−0.01。</li><li>ξ扩展使用0、0.5、1.0三个检验值；这些覆盖该实验类型，不表示正文或附录所有参数点均已运行。</li></ul></section>')
    for i in [10,11]:sections.append(f'<section id="fig{i}"><h2>图{i}</h2>'+figure(f'figure{i:02d}')+'</section>')
    sections.append('<section id="reproduce"><h2>运行代码与数据</h2><pre>python -m pip install -r requirements.txt\npython validate.py\npython run_experiments.py --profile local --threads 10\npython run_rematch.py\npython make_report.py\n\n# 原文规模的参数网格及每点1,000次：计算量大，未在本次运行\npython run_experiments.py --profile paper --threads 10</pre><p>按32次会话分批保存并自动续跑。local为本报告的实际配置；paper将网格扩为100×100×1,000并将四企业提高为1,000次。paper仍不代表在线附录全部实验已经实现；折现因子和扩展检验值按本项目列出的离散集合运行。</p><details><summary>查看全部实验统计</summary>'+summary_html+'</details><details><summary>查看独立实现源代码</summary>'+''.join('<h3>'+f.name+'</h3><pre>'+html.escape(f.read_text(encoding='utf-8'))+'</pre>' for f in ROOT.glob('*.py') if f.name!='make_report.py')+'</details><h3>原始数值下载（离线可用）</h3><div class="downloads">'+''.join(f'<button data-download="{html.escape(name)}">{html.escape(name)}</button>' for name in downloads)+'</div></section>')
    sections.append(f'<footer>运行环境：Python {platform.python_version()} · NumPy {np.__version__} · Numba {numba.__version__} · CPU。报告根据本地结果自动生成，全部图片和CSV嵌入本文件。<br>原文：<a href="https://doi.org/10.1257/aer.20190623">AER 110(10):3267–3297</a>；<a href="https://www.aeaweb.org/articles/materials/13398">官方在线附录</a>；作者复现包标识 <a href="https://doi.org/10.3886/E119462V1">10.3886/E119462V1</a>（本次未使用）。</footer>')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f6;color:#243744;font:16px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1150px;margin:auto;padding:45px 35px}header{padding:25px 0 35px;border-bottom:2px solid #29495e}h1{font-size:42px;line-height:1.3;margin:14px 0;color:#183b51}h2{font-size:25px;color:#214c66;margin-top:0}h3{font-size:18px}.eyebrow{color:#5c7b8d;letter-spacing:2px;font-size:12px}.subtitle{color:#607783}section{background:white;border:1px solid #e0e6e9;border-radius:9px;padding:28px;margin:26px 0;overflow:hidden}aside{padding:20px 24px;background:#fff5df;border-left:4px solid #b78937;margin:25px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.cards>div{background:#e5eef2;padding:24px;border-radius:7px}.cards b{display:block;font-size:34px;color:#244f6c}.cards span{font-size:14px;color:#5c717d}figure{margin:0}figure img{display:block;max-width:100%;height:auto;margin:auto}figcaption{font-size:14px;color:#657986;padding-top:15px}.tablewrap{overflow:auto}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;max-width:100%;margin:15px 0}th,td{padding:8px 12px;white-space:nowrap;border-bottom:1px solid #e5eaee;text-align:right}th{background:#f0f4f6}tbody tr:hover{background:#f7fafb}summary{cursor:pointer;color:#2c6485;font-weight:600;padding:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:18px;border-radius:5px;font-size:13px;line-height:1.55}button{border:1px solid #b9cbd5;background:#f7fbfd;border-radius:5px;padding:7px 10px;cursor:pointer;color:#315a72;font-size:12px}.downloads{display:flex;gap:8px;flex-wrap:wrap}nav{display:flex;gap:18px;flex-wrap:wrap;padding:18px 0}a{color:#326c91}footer{font-size:13px;color:#647b89;padding:30px 0}@media(max-width:650px){main{padding:20px 12px}h1{font-size:30px}section{padding:18px 12px}.cards{grid-template-columns:1fr}.cards b{font-size:28px}body{font-size:15px}}'''
    script='const files='+json.dumps(downloads,ensure_ascii=False).replace('</','<\\/')+';document.querySelectorAll("[data-download]").forEach(b=>b.onclick=()=>{const n=b.dataset.download,a=document.createElement("a");a.href=URL.createObjectURL(new Blob(["\\ufeff",files[n]],{type:"text/csv;charset=utf-8"}));a.download=n;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});'
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>算法定价与合谋 · 独立复现报告</title><style>'+css+'</style><body><main><nav><a href="#methods">方法</a><a href="#coverage">覆盖清单</a><a href="#baseline">表1</a><a href="#table2">表2–3</a><a href="#table4">表4</a><a href="#robust">稳健性</a><a href="#reproduce">代码与数据</a></nav>'+''.join(sections)+'</main><script>'+script+'</script></body></html>'
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    (OUT/'figure_contract.json').write_text(json.dumps(dict(backend='Python/matplotlib',archetype='quantitative grid',claim='Assess independent reproduction of supracompetitive profits, punitive responses, robustness and time scale without substituting published values.',evidence='Figures 1–11, Tables 1–4 and raw seed-level data',exports=['editable SVG','PNG preview','CSV','self-contained HTML'],statistics='Independent seed as replication unit; mean and 95% Monte Carlo CI unless caption states quantiles/range',risks=['reduced parameter grid','unmatched RNG','declared reconstruction choices','appendix incomplete']),indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':main()
