"""Run all computations supported by the supplied package; build a standalone HTML report.

Usage: python reproduce.py [--data-zip inputs/replication_package_MS-FTBDA-20-01585.zip]
Published reference values and missing experiments are never represented as fresh results.
"""
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED
from importlib.metadata import version
from io import BytesIO
import argparse
import base64
import hashlib
import html
import json
import shutil
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import MaxNLocator
import scipy.stats as stats
import statsmodels.api as sm
import author_evaluation as author
import paper_reference as paper
from theory_checks import run_checks

BASE = Path(__file__).resolve().parent
COLORS = ['#9BA9B3','#257F82','#728BB4','#9B87A8','#B69B74','#B8A982','#73958A']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'legend.frameon':False,'axes.titleweight':'bold','figure.facecolor':'white'})

def write_json(value,path):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(frame,path,index=False):
    frame.to_csv(path,index=index,encoding='utf-8-sig')

def unpack_input(archive,target):
    target.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with ZipFile(archive) as z:
        for info in z.infolist():
            parts=PurePosixPath(info.filename).parts
            if info.is_dir() or not parts or parts[0]!='replication_package_MS-FTBDA-20-01585':
                continue
            if '__MACOSX' in parts or '.idea' in parts or parts[-1]=='.DS_Store': continue
            if Path(parts[-1]).suffix not in ['.csv','.ipynb','.docx']: continue
            destination=target.joinpath(*parts[1:]).resolve()
            if not destination.is_relative_to(target.resolve()): raise ValueError('Unsafe archive path')
            destination.parent.mkdir(parents=True,exist_ok=True)
            data=z.read(info)
            destination.write_bytes(data)
            manifest.append({'archive_member':info.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    required=['df_test.csv',*author.FILES.values(),'control_demand_vlt.csv']
    for name in required:
        if not (target/name).is_file(): raise FileNotFoundError(name)
    return manifest

def read_readme(root):
    with ZipFile(root/'README.docx') as doc:
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        xml=ET.fromstring(doc.read('word/document.xml'))
        return '\n'.join(''.join(t.text or '' for t in p.findall('.//w:t',ns)) for p in xml.findall('.//w:p',ns))

def extended_field(root,output):
    # Same observed trajectories, two explicit cost schedules. No decisions are retrained.
    frames={name:pd.read_csv(root/filename) for name,filename in author.FILES.items()}
    rowsets=[];tests=[];regressions=[];did=[]
    for label,h,b in [('author_h1_b9',1,9),('paper_h12_b88',12,88)]:
        data={}
        for group,frame in frames.items():
            m=author.field_metrics(frame).copy()
            m['holding_cost']*=h
            m['stockout_cost']*=b/9
            m['total_cost']=m.holding_cost+m.stockout_cost
            data[group]=m
            for metric in paper.METRICS:
                rowsets.append({'schedule':label,'h':h,'b':b,'group':group,'metric':metric,
                                'n':len(m),'mean':m[metric].mean()})
        for metric in paper.METRICS:
            t=data['treatment_post'][metric]
            c=data['control_post'][metric]
            test=stats.ttest_ind(t,c,equal_var=True)
            tests.append({'schedule':label,'metric':metric,'treatment_mean':t.mean(),'control_mean':c.mean(),
                'difference':t.mean()-c.mean(),'percent_change':100*(t.mean()-c.mean())/c.mean(),
                'pooled_t':test.statistic,'pooled_p':test.pvalue})
            post=pd.concat([data['treatment_post'].assign(e2e=1),data['control_post'].assign(e2e=0)],ignore_index=True)
            x=sm.add_constant(post[['e2e','ave_demand','vlt_num']],has_constant='add')
            for covariance in ['nonrobust','HC3']:
                fit=sm.OLS(post[metric],x).fit(cov_type=covariance)
                ci=fit.conf_int().loc['e2e']
                regressions.append({'schedule':label,'metric':metric,'intercept':True,'covariance':covariance,
                    'coefficient':fit.params['e2e'],'p':fit.pvalues['e2e'],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1]})
            a=data['treatment_pre'][metric].mean();aa=t.mean()
            c0=data['control_pre'][metric].mean();cc=c.mean()
            did.append({'schedule':label,'metric':metric,'treatment_pre':a,'treatment_post':aa,
                'treatment_change':aa-a,'control_pre':c0,'control_post':cc,'control_change':cc-c0,
                'DiD':(aa-a)-(cc-c0),'audited_p':'not identified from supplied identity mapping'})
    for name,rows in [('field_schedules_means',rowsets),('table2_recomputed',tests),
                       ('table3_equation11_recomputed',regressions),('table4_recomputed',did)]:
        write_csv(pd.DataFrame(rows),output/(name+'.csv'))
    # Compare observed subset statistics with published values, without an equality claim.
    comp=[]
    for i,metric in enumerate(paper.METRICS):
        row=next(r for r in tests if r['schedule']=='author_h1_b9' and r['metric']==metric)
        comp.append({'metric':metric,'paper_treatment':paper.TABLE2['treatment'][i],
            'subset_author_treatment':row['treatment_mean'],'paper_control':paper.TABLE2['control'][i],
            'subset_author_control':row['control_mean'],'paper_change_percent':paper.TABLE2['reported_change_percent'][i],
            'subset_change_percent':row['percent_change'],'paper_p':paper.TABLE2['reported_p'][i],
            'subset_p':row['pooled_p']})
    write_csv(pd.DataFrame(comp),output/'table2_published_vs_subset.csv')
    return pd.DataFrame(tests),pd.DataFrame(regressions),pd.DataFrame(did)

def fixed_policy_repricing(recomputed,output):
    original=pd.read_csv(recomputed/'section4_original_costs.csv',index_col=0)
    rows=[]
    for ratio in [1,3,5,7,9,12,15,18,23]:
        for method in ['E2E_RNN','Normal','Gamma']:
            holding=float(original.loc['Holding cost',method])
            shortage=float(original.loc['Stockout cost',method])/9*ratio
            rows.append({'b_over_h':ratio,'method':method,'holding_cost':holding,'stockout_cost':shortage,
                         'total_cost':holding+shortage,'policy':'frozen predictions and frozen trajectories at b/h=9'})
    frame=pd.DataFrame(rows)
    write_csv(frame,output/'supplement_frozen_policy_repricing.csv')
    return frame

def save_figure(fig,name,figdir):
    fig.savefig(figdir/(name+'.svg'),bbox_inches='tight')
    fig.savefig(figdir/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

def build_figures(recomputed,output,tests,did,repricing):
    figdir=output/'figures';figdir.mkdir(exist_ok=True)
    original=pd.read_csv(recomputed/'section4_original_costs.csv',index_col=0)
    # Quantitative grid: the released subset favours E2E among deployable methods.
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    x=np.arange(7)
    ax.bar(x,original.loc['Holding cost'],color=COLORS,width=.65)
    ax.bar(x,original.loc['Stockout cost'],bottom=original.loc['Holding cost'],
           color=COLORS,alpha=.48,hatch='//',width=.65)
    for i,total in enumerate(original.loc['Total cost']): ax.text(i,total+45,f'{total:,.0f}',ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['OPT*','E2E-RNN','BM1','BM2','Normal','Gamma','E2E-GBM'],
        ylabel='Mean evaluated inventory cost per SKU group',title='Figure 2(b): recomputed on the disclosed subset',ylim=(0,4700))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#728BB4',label='Holding'),Patch(facecolor='#B9C8D9',hatch='//',label='Stockout')],loc='upper left')
    fig.text(.5,-.025,'n = 2,607 SKU groups / 3,000 rows; author evaluation convention; *author OPT field',ha='center',fontsize=9)
    save_figure(fig,'fig2b_subset_costs',figdir)
    # Two distinct cost schedules shown in a quantitative grid; no cross-scale cost mixing.
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,metric,title in zip(axes,['total_cost','turnover','stockout_ratio'],['Total inventory cost','Inventory / demand','Stockout fraction']):
        r=tests[(tests.schedule=='author_h1_b9')&(tests.metric==metric)].iloc[0]
        ax.bar(['Treatment','Control'],[r.treatment_mean,r.control_mean],color=['#257F82','#A8B2BD'],width=.55)
        ax.set(title=title,ylabel='Mean, disclosed post-period sample')
        ax.yaxis.set_major_locator(MaxNLocator(5))
        for k,val in enumerate([r.treatment_mean,r.control_mean]): ax.text(k,val,f'{val:,.3g}',ha='center',va='bottom')
        ax.margins(y=.15)
    fig.suptitle('Table 2: disclosed sample, h=1 and b=9; n=500 per group',fontsize=12)
    save_figure(fig,'table2_subset_comparison',figdir)
    fig,ax=plt.subplots(figsize=(7,3.7),layout='constrained')
    r=did[(did.schedule=='author_h1_b9')&(did.metric=='total_cost')].iloc[0]
    ax.plot(['Pre','Post'],[r.treatment_pre,r.treatment_post],'o-',color='#257F82',label='Treatment')
    ax.plot(['Pre','Post'],[r.control_pre,r.control_post],'s--',color='#6F7F91',label='Control')
    ax.set(ylabel='Mean inventory cost',title=f'Descriptive DiD on disclosed sample: {r.DiD:,.2f}')
    ax.legend();fig.text(.5,-.03,'No audited causal p-value: entity linkage and observation windows are not verified.',ha='center',fontsize=9)
    save_figure(fig,'table4_descriptive_did',figdir)
    fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
    for m,c in zip(['E2E_RNN','Normal','Gamma'],['#257F82','#728BB4','#B69B74']):
        f=repricing[repricing.method==m]
        ax.plot(f.b_over_h,f.total_cost,'o-',label=m,color=c)
    ax.set(xlabel='b/h',ylabel='Repriced cost, original frozen trajectories',title='Supplement only: fixed-policy repricing')
    ax.legend();fig.text(.5,-.04,'Not Figure B.2: no relabeling, retraining, or policy optimization at the new ratios.',ha='center',fontsize=9)
    save_figure(fig,'supplement_frozen_repricing',figdir)
    # Schematic-led figure: structural reconstruction, not a trained-network result.
    fig,ax=plt.subplots(figsize=(12,5.1),layout='constrained');ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    nodes={
        'df':(.085,.83,'Demand\nfeatures','#E1ECF1'), 'basic':(.085,.53,'Basic\nfeatures','#E1ECF1'),
        'vlt':(.085,.22,'VLT\nfeatures','#E1ECF1'), 'r':(.49,.08,'Review\nperiod','#E1ECF1'),
        's':(.82,.08,'Initial\nstock','#E1ECF1'), 'rnn':(.28,.80,'DF module\nMQRNN','#CBE6E2'),
        'v1':(.26,.27,'VLT layer 1\n50 units','#CBE6E2'), 'v2':(.43,.27,'VLT layer 2\n20 units','#CBE6E2'),
        'l3':(.56,.60,'Layer 3\n100 units','#CBE6E2'),'l4':(.74,.60,'Layer 4\n100 units','#CBE6E2'),
        'out1':(.92,.60,'Out 1\nOrder','#F0DFC9'), 'out2':(.52,.85,'Out 2\nDemand','#F0DFC9'),
        'out3':(.64,.27,'Out 3\nLead time','#F0DFC9')}
    nodes['r']=(.55,.08,'Review\nperiod','#E1ECF1')
    edges=[('df','rnn'),('basic','rnn'),('basic','v1'),('vlt','v1'),('v1','v2'),('rnn','l3'),
           ('rnn','out2'),('v2','l3'),('v2','out3'),('r','l3'),('s','l4'),('l3','l4'),('l4','out1')]
    for a,b in edges:
        x1,y1,*_=nodes[a];x2,y2,*_=nodes[b]
        dx,dy=x2-x1,y2-y1
        edge=min(.071/max(abs(dx),1e-9),.065/max(abs(dy),1e-9))
        start=(x1+edge*dx,y1+edge*dy)
        end=(x2-edge*dx,y2-edge*dy)
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
            linewidth=1.1,color='#526574',shrinkA=2,shrinkB=2))
    for key,(x,y,label,color) in nodes.items():
        ax.add_patch(FancyBboxPatch((x-.061,y-.055),.122,.11,boxstyle='round,pad=.01',facecolor=color,edgecolor='#607481',linewidth=.8))
        ax.text(x,y,label,ha='center',va='center',fontsize=9)
    ax.set_title('Figure 1: structural reconstruction from the paper',pad=16)
    fig.text(.5,-.02,'Shared basic inputs and late inventory/review inputs follow the published diagram. No learned weights are available.',ha='center',fontsize=9)
    save_figure(fig,'figure1_architecture',figdir)
    for name in ['matching_scores','matching_balance']:
        shutil.copy2(recomputed/(name+'.png'),figdir/(name+'.png'))

def html_table(frame):
    def fmt(x):
        if isinstance(x,(float,np.floating)):
            if not np.isfinite(x):return '—'
            if 0<abs(x)<.0001:return f'{x:.3e}'
            return f'{x:,.4f}'.rstrip('0').rstrip('.')
        return str(x)
    return '<div class="table-wrap">'+frame.to_html(index=False,escape=True,border=0,formatters={c:fmt for c in frame.columns})+'</div>'

def inline_image(path,caption):
    encoded=base64.b64encode(path.read_bytes()).decode('ascii')
    mime='image/svg+xml' if path.suffix=='.svg' else 'image/png'
    return f'<figure><img src="data:{mime};base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'

def csv_download(path,label=None):
    payload=base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<a class="download" download="{html.escape(path.name)}" href="data:text/csv;base64,{payload}">↓ {html.escape(label or path.name)}</a>'

def build_html(root,recomputed,output,theory,tests,regression,did,manifest,elapsed):
    read=lambda name:pd.read_csv(recomputed/name)
    costs=pd.read_csv(recomputed/'section4_original_costs.csv',index_col=0)
    figdir=output/'figures'
    evidence=output/'evidence';evidence.mkdir(exist_ok=True)
    reference={'provenance':'manually transcribed from published PDF; NOT rerun results',
        'Figure2b':paper.FIG2,'Table2':paper.TABLE2,'Table3':paper.TABLE3,'Table4':paper.TABLE4,
        'Table1':paper.CATEGORIES,'TableB1':paper.HYPERPARAMETERS,'Appendix':paper.APPENDIX}
    write_json(reference,evidence/'paper_reference_values.json')
    coverage=pd.DataFrame(paper.COVERAGE,columns=['item','journal_page','class','status','result','missing_input'])
    write_csv(coverage,evidence/'coverage.csv')
    reference_offline=pd.DataFrame.from_dict(paper.FIG2,orient='index',columns=['paper_total','paper_holding','paper_stockout'])
    comparison=costs.T.join(reference_offline).reset_index(names='method')
    comparison['subset_vs_paper_total_difference']=comparison['Total cost']-comparison.paper_total
    write_csv(comparison,output/'figure2b_published_vs_subset.csv')
    chart_contracts=[
        {'figure':'figure1_architecture','claim':'Published module connections can be reconstructed; weights cannot.',
         'archetype':'schematic-led composite','data':'Figure 1, journal p.764','statistic':'none'},
        {'figure':'fig2b_subset_costs','claim':'E2E has lowest cost among deployable policies in the disclosed author evaluation.',
         'archetype':'quantitative grid','data':'recomputed/section4_original_costs.csv',
         'n':'3000 rows aggregated to 2607 SKU groups','statistic':'arithmetic mean; no uncertainty intervals'},
        {'figure':'table2_subset_comparison','claim':'The disclosed post-period sample has lower treatment means.',
         'archetype':'quantitative grid','data':'table2_recomputed.csv','n':'500 per group','statistic':'means; no causal claim'},
        {'figure':'table4_descriptive_did','claim':'Four group means yield a negative descriptive inventory-cost DiD.',
         'archetype':'quantitative grid','data':'table4_recomputed.csv','statistic':'difference of group means; audited p unavailable'},
        {'figure':'supplement_frozen_repricing','claim':'Fixed trajectory costs depend mechanically on cost weights.',
         'archetype':'quantitative grid','data':'supplement_frozen_policy_repricing.csv','statistic':'deterministic repricing, no retraining'},
        {'figure':'matching_scores','claim':'Both matching implementations improve the displayed score overlap.',
         'archetype':'quantitative grid','data':'recomputed/matching_pairs.csv',
         'n':'500 treated, 55333 candidate controls; matching with replacement',
         'statistic':'density histogram; no causal identification claim'},
        {'figure':'matching_balance','claim':'Released covariates have smaller absolute standardized differences after matching.',
         'archetype':'quantitative grid','data':'recomputed/matching_balance.csv',
         'statistic':'fixed pre-match pooled SD; repeated controls counted as matched rows'},
    ]
    write_json(chart_contracts,evidence/'figure_contracts.json')
    coverage_rows=''
    for _,r in coverage.iterrows():
        coverage_rows+=f'<tr data-class="{r["class"]}"><td><strong>{html.escape(r["item"])}</strong><br><small>p.{r.journal_page}</small></td><td><span class="badge {r["class"]}">{html.escape(r.status)}</span></td><td>{html.escape(r.result)}</td><td>{html.escape(r.missing_input)}</td></tr>'
    appendix=''
    for name,entry in paper.APPENDIX.items():
        columns=entry.get('columns',['variant','paper_total','paper_holding','paper_stockout'])
        reference_frame=pd.DataFrame(entry['rows'],columns=columns)
        write_csv(reference_frame,evidence/(name.replace(' ','_')+'_REFERENCE_ONLY.csv'))
        appendix+=f'<details><summary>{html.escape(name)} · 原文参照值，未重新训练</summary><p class="warning">以下数值来自论文p.{entry["page"]}，不是本次运行结果。重跑所需材料见覆盖清单。</p>{html_table(reference_frame)}</details>'
    table2_paper=pd.DataFrame({'metric':paper.METRICS,**paper.TABLE2})
    table3_paper=pd.DataFrame({'metric':paper.METRICS,**paper.TABLE3})
    table4_paper=pd.DataFrame({'metric':paper.METRICS,**paper.TABLE4})
    cfg=pd.DataFrame(paper.HYPERPARAMETERS,columns=['parameter','paper_default','paper_search_range'])
    write_csv(cfg,evidence/'Table_B1_REFERENCE_ONLY.csv')
    c1=read('inventory_implementation_audit.csv')
    inventory_example=read('example_sku_inventory_trace.csv')
    example=inventory_example[(inventory_example.method=='E2E_RNN') & inventory_example.counted_in_author_cost.astype(bool)].head(12)
    means=read('field_means_author_convention.csv').rename(columns={'Unnamed: 0':'metric'})
    eq11=regression[(regression.schedule=='author_h1_b9')&(regression.covariance=='nonrobust')]
    field12=tests[tests.schedule=='paper_h12_b88']
    subsettests=tests[tests.schedule=='author_h1_b9']
    true_did=did[did.schedule=='author_h1_b9']
    integrity=pd.DataFrame([
        ['输入数据','原作者ZIP逐文件SHA256，原包未改写。'],
        ['模型输出','直接使用包内E2E_RNN_pred / gbm_pred等字段；没有伪造训练或权重。'],
        ['实现差异','旧pandas接口等价替换，原始匹配画图单元跳过后重画；统计计算保留。'],
        ['Normal公式','只在独立对照中改为论文公式(7)及90%分位数，主结果仍是原代码。'],
        ['实验成本','实地实验分别报告代码h=1,b=9和论文h=12,b=88；不同口径不可混用。'],
        ['DiD推断','公开索引不足以认证个体关联；原p值仅作代码复算，不认证修正版显著性。'],
        ['训练敏感性','缺失训练数据和多模型输出，未用论文印刷值填充运行结果。'],
        ['额外分析','固定策略重计价、理论数值检验均单列，不能替代附录的训练实验。'],
    ],columns=['检查项','处理方式'])
    source_blocks=''
    for name in ['reproduce.py','author_evaluation.py','theory_checks.py','paper_reference.py']:
        content=(BASE/name).read_text(encoding='utf-8')
        source_blocks+=f'<details><summary>{name} · 查看完整源码</summary><pre><code>{html.escape(content)}</code></pre></details>'
    archive_entries=[]
    for folder in [output,recomputed,evidence]:
        for file in sorted(folder.glob('*.csv')):
            archive_entries.append(csv_download(file))
    categories=pd.DataFrame([(parent,child) for parent,children in paper.CATEGORIES.items() for child in children],columns=['second_level','third_level'])
    gap=100*(costs.loc['Total cost','E2E_RNN']/costs.loc['Total cost','Bm2']-1)
    report=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T1 本地论文复现结果与全项覆盖报告</title><style>
:root{{--ink:#172f42;--muted:#637584;--teal:#257f82;--line:#dce4e8;--bg:#f3f6f7;--warn:#875d20}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.75 "Microsoft YaHei","PingFang SC",Arial,sans-serif}}a{{color:var(--teal)}}aside{{position:fixed;inset:0 auto 0 0;width:225px;background:#142f42;color:white;padding:34px 24px;overflow:auto}}aside b{{display:block;font-size:21px;line-height:1.4}}aside small{{color:#a9c2cc}}nav{{margin-top:28px}}nav a{{display:block;color:#d8e6eb;text-decoration:none;padding:9px 0;border-bottom:1px solid #345060}}main{{margin-left:225px;max-width:1450px;padding:42px 4.5vw 80px}}header{{margin-bottom:26px}}h1{{font-size:32px;line-height:1.4;margin:12px 0}}h2{{font-size:23px;border-bottom:2px solid var(--teal);padding-bottom:10px;margin-top:0}}h3{{font-size:18px}}p{{margin:12px 0}}.eyebrow{{color:var(--teal);letter-spacing:.1em;font-size:12px;font-weight:bold}}.sub{{color:var(--muted)}}section{{background:white;border:1px solid var(--line);border-radius:12px;padding:26px 30px;margin:24px 0;scroll-margin-top:20px}}.notice,.warning{{background:#fff5e5;border-left:4px solid #c69b55;padding:14px 18px;color:#694e28}}.notice strong{{color:#694e28}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px}}.card strong{{display:block;font-size:30px;color:var(--teal)}}.card span{{color:var(--muted);font-size:13px}}.table-wrap{{overflow:auto;margin:16px 0;max-height:650px;border:1px solid var(--line);border-radius:7px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}th{{background:#eaf1f3;position:sticky;top:0;white-space:nowrap}}tbody tr:nth-child(even){{background:#f7f9fa}}figure{{margin:22px 0;padding:16px;border:1px solid var(--line);border-radius:9px;background:#fff}}figure img{{width:100%;height:auto;display:block}}figcaption{{font-size:13px;color:var(--muted);padding-top:12px}}.badge{{display:inline-block;border-radius:5px;padding:3px 8px;font-size:12px;white-space:nowrap}}.subset,.schematic{{background:#e1f0ec;color:#216863}}.missing{{background:#fff0db;color:#805716}}.reference{{background:#e9edf4;color:#566781}}details{{border:1px solid var(--line);border-radius:7px;margin:12px 0;padding:12px 16px}}summary{{cursor:pointer;font-weight:bold}}pre{{overflow:auto;max-height:550px;background:#122b3c;color:#dceaf0;padding:18px;font:12px/1.65 Consolas,monospace;border-radius:6px;white-space:pre}}code{{font-family:Consolas,monospace}}.toolbar{{display:flex;gap:10px;flex-wrap:wrap;margin:12px 0}}button,input{{padding:9px 12px;border:1px solid #bdcdd5;border-radius:6px;background:white;font:inherit}}button{{cursor:pointer;color:var(--teal)}}.download{{display:inline-block;margin:5px 8px 5px 0;padding:5px 9px;border:1px solid var(--line);border-radius:5px;font-size:12px;text-decoration:none}}footer{{color:var(--muted);font-size:12px}}.formula{{padding:16px;background:#eef5f5;border-radius:8px;font-family:Consolas,"Microsoft YaHei",monospace}}@media(max-width:900px){{aside{{position:static;width:auto;padding:20px}}nav{{display:flex;gap:16px;flex-wrap:wrap;margin-top:12px}}nav a{{border:0}}main{{margin:0;padding:22px}}h1{{font-size:25px}}.cards{{grid-template-columns:1fr}}section{{padding:20px}}}}@media print{{aside,.toolbar{{display:none}}main{{margin:0;max-width:none;padding:0}}body{{background:white}}section{{break-inside:auto;border:0}}.table-wrap{{max-height:none;overflow:visible}}details[open] pre{{max-height:none}}figure{{break-inside:avoid}}}}
</style></head><body><aside><b>T1 · 本地复现</b><small>Management Science · 2023<br>计算结果 / 原文参照 / 数据缺口</small><nav>
<a href="#summary">执行摘要</a><a href="#coverage">全部图表覆盖</a><a href="#offline">离线库存评估</a><a href="#field">实地实验统计</a><a href="#theory">理论与结构</a><a href="#appendix">附录逐项核查</a><a href="#audit">实现与数据审计</a><a href="#repro">代码与下载</a></nav></aside><main>
<header><div class="eyebrow">LOCAL REPLICATION · AUDITABLE RESULTS</div><h1>A Practical End-to-End Inventory<br>Management Model with Deep Learning</h1><p class="sub">本地复现结果与全项覆盖报告 · 基于用户提供的作者公开复现包 · 2026-10-20</p>
<div class="notice"><strong>完整性结论：</strong>已重算公开子集支持的主要离线评估、实地实验统计和匹配，完成理论数值检验。<strong>不能声称论文所有训练实验已复现：</strong>完整训练集、训练源码、不同超参数及随机种子权重均未包含在输入包中。下文覆盖论文全部17个图表项目（图2分左右两项），逐项说明完成度。</div></header>
<div class="cards"><div class="card"><strong>3,000 / 2,607</strong><span>离线记录数 / 实际不同SKU组数</span></div><div class="card"><strong>{costs.loc['Total cost','E2E_RNN']:,.2f}</strong><span>公开子集E2E-RNN平均总成本，原代码口径</span></div><div class="card"><strong>{int(theory.cases.sum()):,}</strong><span>闭式解、独立DP及真实销量窗口数值检验，全部通过</span></div></div>
<section id="summary"><h2>01 / 本次实际运行了什么</h2><p>执行作者的三个Notebook计算单元，并修复旧版pandas接口；独立跟踪器逐一核对了 <strong>2,607 × 7 = 18,249</strong> 组库存轨迹。E2E-RNN在公开子集的6种可部署策略中成本最低，比BM2低 <strong>{abs(gap):.2f}%</strong>；OPT仅沿用作者字段名称，不认证为其他库存规则下的最优下界。</p>
<p>主结果来自公开数据，<strong>没有新训练E2E网络</strong>。补充计算包含论文实地成本权重、公式(11)回归、Normal基准公式对照、固定策略重计价以及附录A数值验证。数据不同意味着论文原表与子集结果不能逐格相等。</p>
<p>论文第5节写 <strong>h=12,b=88</strong>；包内Notebook实际用 <strong>h=1,b=9</strong>。本报告同时给出两种口径，避免将成本单位混用。四组各500条样本，而论文处理和对照组各6,097个SKU-DC组合；候选对照文件有55,333条特征。</p></section>
<section id="coverage"><h2>02 / 全部图表的复现状态</h2><p>绿色表示结构重绘或公开子集重算；蓝色是原文配置整理；黄色表示缺失必要输入。原文参照值均明确标记，不能算作实验复现。</p>
<div class="toolbar"><input id="coverage-search" placeholder="搜索图表或缺失材料" aria-label="搜索覆盖清单"><button data-filter="all">全部</button><button data-filter="subset">子集重算</button><button data-filter="missing">缺失输入</button></div><div class="table-wrap"><table id="coverage-table"><thead><tr><th>原文项目</th><th>状态</th><th>实际处理</th><th>仍需材料</th></tr></thead><tbody>{coverage_rows}</tbody></table></div></section>
<section id="offline"><h2>03 / Figure 2(b)：离线补货评估</h2>{inline_image(figdir/'fig2b_subset_costs.svg','公开子集实际重算：按SKU分组后的平均成本，持有与缺货成本分开显示；无误差线。')}
<h3>运行结果与论文原值并列</h3><p>左侧运行结果与右侧论文参照来自不同样本，差值只描述数值差异，不是“复现精度”。计算流程仍保留作者代码的取整、库存转移和计费点选择。</p>{html_table(comparison)}
<h3>Normal公式的单项对照</h3><p>作者代码使用Z=1.64及不同的方差组合；对照版使用 Φ⁻¹(0.9) 和公式(7)：(R+μ<sub>L</sub>)σ²<sub>D</sub> + μ²<sub>D</sub>σ²<sub>L</sub>。只改变预测公式，其他评估规则不变，不能当成完全修正后的库存模型。</p>{html_table(read('section4_normal_formula_sensitivity.csv').rename(columns={'Unnamed: 0':'metric'}))}
<h3>Figure 2(a)：为什么没有重画预测置信带</h3><div class="warning">df_test只包含sf_rnn单条预测序列，没有原图的10%、50%、90%分位数输出、原图SKU对应和完整日期轴。不能从单条序列倒推出三个分位数；本报告没有生成虚假的预测带。</div></section>
<section id="field"><h2>04 / Tables 2–4：实地实验公开子集</h2><p>保持作者对库存序列的计费口径：成本累加前T−1点，缺货率除以T，负周转指标截断为0。观察窗口不同、两个实验后对照样本只有一个库存点等问题在审计部分列出。</p>
<h3>Table 2：包内成本口径 h=1,b=9</h3>{html_table(subsettests)}{inline_image(figdir/'table2_subset_comparison.svg','公开实验后样本均值，处理组与对照组各500条；是描述性组间比较，不是新增随机实验。')}
<h3>Table 2：按论文文本成本 h=12,b=88 重新计价</h3><p>使用相同已观测库存曲线，只改变成本权重，不重新优化订货决策；周转和缺货率不受成本权重影响。</p>{html_table(field12)}
<details><summary>Table 2 · 论文原值，仅作参照（p.768）</summary>{html_table(table2_paper)}</details>
<h3>Table 3：原始无截距回归与公式(11)回归</h3><p>原Notebook未加入常数项。下表先保留原结果，再报告带截距的普通OLS；稳健HC3对照另存CSV。公开子集的周转指标对是否加入截距非常敏感，不能选择性展示显著结果。</p>{html_table(read('field_regression_author.csv'))}{html_table(eq11)}
<details><summary>Table 3 · 两种成本权重下的带截距回归与HC3完整对照（实际重算）</summary>{html_table(regression)}</details>
<details><summary>Table 3 · 论文原系数，仅作参照</summary>{html_table(table3_paper)}</details>
<h3>Table 4：描述性DiD</h3><div class="formula">DiD = (处理组后 − 处理组前) − (对照组后 − 对照组前)</div>{html_table(true_did)}{inline_image(figdir/'table4_descriptive_did.svg','按统一非负周转口径计算的四组均值差值。未提供经过认证的DiD p值。')}
<p>原代码DiD p值保存在field_did.csv，列名含unverified_pairing。四组都使用0…499索引，不足以证明跨期同一个体关联；随意按行相减产生的检验不能直接认证。修正版本只报告点估计。</p>
<details><summary>Table 4 · 两种成本权重下的完整描述性DiD（实际重算）</summary>{html_table(did)}</details>
<details><summary>Table 4 · 论文原值，仅作参照</summary>{html_table(table4_paper)}</details>
<h3>Figure 3：匹配与平衡性</h3>{inline_image(figdir/'matching_scores.png','同一个逻辑回归倾向得分模型：匹配前、作者协变量近邻匹配后，以及倾向得分近邻匹配后。')}{inline_image(figdir/'matching_balance.png','协变量绝对标准化均值差，固定使用匹配前合并标准差；0.1虚线为描述性诊断参考。')}{html_table(read('matching_summary.csv'))}
<p>作者代码实际在平均需求和提前期的原始协变量上近邻匹配，而非在倾向得分上近邻匹配。两种方式均有放回，500个匹配行实际复用部分候选对照。匹配特征的处理前属性和候选对照到订单结果的映射均未认证，不能把新匹配对象强行连接到现有对照订单。</p></section>
<section id="theory"><h2>05 / 模型结构、引理A.1与定理1</h2>{inline_image(figdir/'figure1_architecture.svg','按Figure 1重绘结构，采用Table B.1披露的层大小；仅表示结构，不表示已获得训练权重。')}
<div class="formula">a* = max(累计需求 d[vₘ,s*] − 到货前净库存 I[vₘ], 0)<br>s* = vₘ + floor[b(vₘ₊₁ − vₘ)/(h+b)]</div>
<p>引理检验分别用独立整数枚举和真实销量窗口的全部折点比较；定理检验用独立的整数动态规划，与逐到货区间闭式决策比较。小规模受控案例仅用于验证程序和公式，<strong>不是替代京东实证数据，也不是数学证明</strong>。</p>{html_table(theory)}
<p>备注1关于可计算标签的含义可由闭式算法体现，但本次不声称复现原论文的训练速度或泛化最优性。代码使用正h、正b、非负需求、给定到货顺序与无容量/固定订货费的设定。</p>
<details><summary>Table 1 · 原文18个品类（配置整理，非样本统计验证）</summary>{html_table(categories)}</details></section>
<section id="appendix"><h2>06 / 附录B：逐项记录，避免伪复现</h2><h3>Table B.1：原文配置</h3>{html_table(cfg)}
<div class="warning"><strong>以下训练实验没有可执行的完整输入：</strong>Table B.2–B.6、Figure B.1–B.3需要训练集、完整特征、训练流程或多组模型。原始示例只有5行SKU-DC销量、2行库存、1行提前期订单且部分特征为空，无法重建132维训练样本。完整样本未公开不是算力问题，本次没有用测试集拆分、论文原值、随机权重或模拟曲线冒充这些实验。</div>
{appendix}
<h3>Figure B.1 / B.2 / B.3 的具体缺口</h3><p>B.1缺少不同训练规模的训练记录和计时；B.2缺少每个成本比下重新生成标签和训练后的预测；B.3缺少两个随机种子的VLT_layer1权重矩阵。原文Table B.5中的训练耗时只在原值参照表出现，不当成本机实测时间。</p>
<h3>额外可运行分析：固定策略重计价</h3>{inline_image(figdir/'supplement_frozen_repricing.svg','补充分析：冻结b/h=9时的预测和库存轨迹，仅重新加权成本。此图不是原Figure B.2的训练敏感性复现。')}
<p>完整性缺口不能通过画一条相似曲线解决；补齐所需训练数据和检查点后才能执行相应实验。coverage.csv列出了每项所需材料。</p></section>
<section id="audit"><h2>07 / 实现审计和数据证据</h2>{html_table(integrity)}<h3>全部SKU轨迹的实现行为统计</h3>{html_table(c1)}
<p>负补货、到货时移除负库存、跳过重叠窗口等计数来自作者函数行为记录，不代表每项可脱离实际业务语义直接判错。但这些行为不等同于公式(1)的完全积压需求模型；因此不能在未修订整个仿真协议时，将包内OPT当作新模型的严格下界。</p>
<h3>一个多订单SKU的逐期核对示例（截取）</h3>{html_table(example[['SKU','round','day_in_window','computed_replenishment','stock_before','demand','stock_after','holding_cost','stockout_cost']])}
<h3>观察窗口与输入数据</h3>{html_table(read('input_audit.csv'))}<p>complete_tm的日期范围是提供的订单完成时间范围，不直接等于每个库存数组的日期窗口。按T−1重新定义分母时，只有一个库存点的记录没有可评价天数，敏感性表将其记为缺失并报告有效样本数。</p>
<details><summary>README原文数据说明（提取的文本证据）</summary><pre>{html.escape(read_readme(root))}</pre></details></section>
<section id="repro"><h2>08 / 复现代码、运行方式与下载</h2><p>无需Colab。将交付包解压后，在该目录运行：</p><pre>python -m pip install -r requirements.txt\npython reproduce.py</pre>
<p>也可以指定原始ZIP和输出目录：</p><pre>python reproduce.py --data-zip "你的作者复现包.zip" --output "results"</pre>
<p>运行会重新执行作者三个Notebook，验证轨迹与理论案例，生成全部CSV、图像、环境与输入哈希、当前HTML。HTML内图像和表格自包含，断网也能查看；下方CSV可直接从本页面下载。</p><div>{''.join(archive_entries)}</div>
<h3>完整源代码</h3>{source_blocks}
<p>来源：用户提供的2023正式发表版本PDF（p.759–773）和作者复现包；所有原值保存在paper_reference_values.json并标注REFERENCE ONLY。论文DOI：<a href="https://doi.org/10.1287/mnsc.2022.4564">10.1287/mnsc.2022.4564</a>。本次数据重新计算和原文人工转录分别存储。</p></section>
<footer>分析耗时约 {elapsed:.1f} 秒（本机、含验证；不是论文训练时间）。报告无外部脚本依赖。输入哈希、源码哈希与依赖版本见manifest.json；执行日志见recomputed目录。</footer>
</main><script>(()=>{{let mode='all';const box=document.getElementById('coverage-search');const apply=()=>{{const q=box.value.toLowerCase();document.querySelectorAll('#coverage-table tbody tr').forEach(row=>{{row.style.display=((mode==='all'||row.dataset.class===mode)&&row.textContent.toLowerCase().includes(q))?'':'none';}});}};box.addEventListener('input',apply);document.querySelectorAll('[data-filter]').forEach(btn=>btn.addEventListener('click',()=>{{mode=btn.dataset.filter;apply();}}));}})();</script></body></html>'''
    (output/'report.html').write_text(report,encoding='utf-8')
    return coverage

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-zip',type=Path,default=BASE/'inputs/replication_package_MS-FTBDA-20-01585.zip')
    parser.add_argument('--output',type=Path,default=BASE/'results')
    args=parser.parse_args();start=time.perf_counter()
    if not args.data_zip.exists(): raise FileNotFoundError('Provide the original author package via --data-zip')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    print('[1/7] Extracting reviewed input package',flush=True)
    data_root=output/'input_data'
    entries=unpack_input(args.data_zip,data_root)
    print('[2/7] Running the three author evaluation notebooks and CPU audit',flush=True)
    original_output=author.run_all(data_root,output/'author_runs')
    recomputed=output/'recomputed';shutil.copytree(original_output,recomputed,dirs_exist_ok=True)
    print('[3/7] Recomputing field tables using both published and code cost schedules',flush=True)
    tests,regression,did=extended_field(data_root,output)
    print('[4/7] Independent numerical tests for Lemma A.1 and Theorem 1',flush=True)
    theory=run_checks(data_root,output)
    repricing=fixed_policy_repricing(recomputed,output)
    print('[5/7] Drawing figures with traceable source data',flush=True)
    build_figures(recomputed,output,tests,did,repricing)
    print('[6/7] Building standalone HTML, reference tables and coverage ledger',flush=True)
    coverage=build_html(data_root,recomputed,output,theory,tests,regression,did,entries,time.perf_counter()-start)
    metadata={'scope':'disclosed-subset replication; full paper training results unavailable',
        'data_zip_sha256':hashlib.sha256(args.data_zip.read_bytes()).hexdigest(),
        'archive_manifest':entries,'python':sys.version,
        'packages':{name:version(name) for name in ['numpy','pandas','scipy','matplotlib','seaborn','scikit-learn','statsmodels','tqdm']},
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.glob('*.py')},
        'coverage_counts':coverage['class'].value_counts().to_dict(),
        'theory_cases':int(theory.cases.sum()),'theory_failures':int(theory.failed.sum()),
        'runtime_seconds':round(time.perf_counter()-start,3)}
    write_json(metadata,output/'manifest.json')
    print('[7/7] Validating numerical identities and HTML',flush=True)
    text=(output/'report.html').read_text(encoding='utf-8')
    assert len(coverage)==17 and '训练实验没有可执行的完整输入' in text
    assert '<html lang="zh-CN">' in text and 'data:image/svg+xml;base64,' in text
    assert len(tests)==10 and len(regression)==20 and len(did)==10
    assert np.allclose(tests[tests.metric.eq('total_cost')].treatment_mean,
        tests[tests.metric.eq('holding_cost')].treatment_mean.to_numpy()+tests[tests.metric.eq('stockout_cost')].treatment_mean.to_numpy())
    write_json({'status':'passed','html':'self-contained images and embedded CSV downloads',
        'paper_items':17,'all_training_results_reproduced':False,
        'theory_checks_passed':int(theory.passed.sum()),'inventory_traces_validated':2607*7},output/'validation.json')
    print('HTML:',output/'report.html',flush=True)
    print('Published-item coverage:',metadata['coverage_counts'])

if __name__=='__main__':main()
