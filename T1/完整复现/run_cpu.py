"""CPU replication of the disclosed T1 evaluation package; no model training."""
# %% Imports and utilities
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from importlib.metadata import version
from datetime import datetime, timezone
import ast
import hashlib
import io
import json
import os
import platform
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

METHODS = ['OPT', 'E2E_RNN', 'Bm1', 'Bm2', 'Normal', 'Gamma', 'gbm']
METRICS = ['holding_cost', 'stockout_cost', 'total_cost', 'turnover', 'stockout_ratio']
FILES = {
    'treatment_pre': 'order_e2e_pre_500.csv',
    'treatment_post': 'order_e2e_post_500.csv',
    'control_pre': 'order_jd_pre_500.csv',
    'control_post': 'order_jd_post_500.csv',
}

def save_table(frame, path, index=True):
    frame.to_csv(path, index=index, encoding='utf-8-sig')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_nb(root, name):
    path = root / name
    if not path.exists():
        checkpoint = {
            'E2E_Section4.ipynb': 'E2E_Section4-checkpoint.ipynb',
            'Field_experiments_summary.ipynb': 'Field_experiments_summary 2-checkpoint.ipynb',
            'Propensity_score_matching.ipynb': 'Propensity_score_matching 2-checkpoint.ipynb',
        }[name]
        path = root / '.ipynb_checkpoints' / checkpoint
    return path, json.loads(path.read_text(encoding='utf-8'))

@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def replay(root, name, output, callback=None, skip=()):
    """Run author calculations, changing only the obsolete pandas drop API."""
    path, notebook = read_nb(root, name)
    namespace = {'__name__': '__author_notebook__'}
    changes = []
    with working_directory(root), (output / (path.stem + '.log')).open('w', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), warnings.catch_warnings():
            warnings.simplefilter('always')
            for i, cell in enumerate(notebook['cells']):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell.get('source', []))
                if not source.strip():
                    continue
                if i in skip:
                    changes.append({'cell': i, 'change': 'Plot-only cell skipped; plots regenerated separately.'})
                    continue
                modern = source.replace("df_lr.drop('y',1)", "df_lr.drop(columns='y')")
                if modern != source:
                    changes.append({'cell': i, 'change': "drop('y',1) -> drop(columns='y'); same columns"})
                print('AUTHOR CELL', i)
                exec(compile(modern, f'{path.name}:cell-{i}', 'exec'), namespace)
                if callback:
                    callback(i, namespace)
    return namespace, {'source': str(path.name), 'sha256': digest(path), 'changes': changes}

# %% Data checks and original Section 4
def audit_inputs(root, output):
    records = []
    field = {}
    for group, filename in FILES.items():
        frame = pd.read_csv(root / filename)
        field[group] = frame
        dates = pd.to_datetime(frame['complete_tm'], format='%m/%d/%y', errors='raise')
        inv = frame.test_inv.map(ast.literal_eval)
        demand = frame.test_demand.map(ast.literal_eval)
        assert frame.sku_dc_pair_index.is_unique
        assert inv.map(len).gt(0).all()
        assert np.isfinite(frame[['ave_demand', 'ave_inv']].to_numpy()).all()
        records.append({
            'file': filename, 'rows': len(frame), 'missing_cells': int(frame.isna().sum().sum()),
            'distinct_ids': frame.sku_dc_pair_index.nunique(),
            'min_complete_date': dates.min().date().isoformat(),
            'max_complete_date': dates.max().date().isoformat(),
            'min_inventory_points': inv.map(len).min(), 'max_inventory_points': inv.map(len).max(),
            'one_point_inventory_rows': int(inv.map(len).eq(1).sum()),
            'equal_inv_demand_lengths': int((inv.map(len) == demand.map(len)).sum()),
            'zero_average_demand': int(frame.ave_demand.eq(0).sum()),
            'negative_average_inventory': int(frame.ave_inv.lt(0).sum()),
        })
    save_table(pd.DataFrame(records), output / 'input_audit.csv', index=False)
    id_rows = []
    for a, b in [('treatment_pre', 'treatment_post'), ('control_pre', 'control_post'), ('treatment_post', 'control_post')]:
        ids_a, ids_b = set(field[a].sku_dc_pair_index), set(field[b].sku_dc_pair_index)
        id_rows.append({'group_a': a, 'group_b': b, 'shared_numeric_labels': len(ids_a & ids_b),
                        'same_order': field[a].sku_dc_pair_index.equals(field[b].sku_dc_pair_index),
                        'identity_linkage_verified': False})
    save_table(pd.DataFrame(id_rows), output / 'id_audit.csv', index=False)
    return field

def section4(root, output):
    ns, change = replay(root, 'E2E_Section4.ipynb', output)
    costs = ns['df_aggcom'].astype(float)
    save_table(costs, output / 'section4_original_costs.csv')
    raw = pd.read_csv(root / 'df_test.csv')
    assert len(raw) == 3000 and raw.SKU.nunique() == 2607
    assert np.isfinite(costs.to_numpy()).all()
    assert np.allclose(costs.loc['Total cost'], costs.loc['Holding cost'] + costs.loc['Stockout cost'])
    reference = pd.Series({'OPT':2572.96, 'E2E_RNN':3036.76, 'Bm1':3842.31, 'Bm2':3837.83,
                           'Normal':3961.69, 'Gamma':4099.95, 'gbm':3424.87})
    assert np.max(np.abs(costs.loc['Total cost'] - reference)) < .011
    # Correct only the Normal formula to equation (7); retain all other author evaluation rules.
    corrected = raw.copy()
    corrected['demand_hist'] = corrected['demand_hist'].map(lambda x: pd.to_numeric(x.strip('][').split()))
    variance = ((corrected.review_period + corrected.vendor_vlt_mean) * corrected.demand_std ** 2
                + corrected.demand_mean ** 2 * corrected.vendor_vlt_std ** 2)
    corrected['Normal_eq7_pred'] = (corrected.demand_mean * (corrected.review_period + corrected.vendor_vlt_mean)
                                   + stats.norm.ppf(.9) * np.sqrt(variance)).astype(int)
    grouped = corrected.groupby('SKU').agg(lambda x: x.tolist())
    trajectories = grouped.apply(ns['get_agginv'], name='Normal_eq7', axis=1)
    h = trajectories.map(lambda item: sum(max(v, 0) for v in item[1]))
    b = trajectories.map(lambda item: 9 * sum(max(-v, 0) for v in item[1]))
    sensitivity = pd.DataFrame({'Normal_original': costs['Normal'],
        'Normal_eq7_only': [float((h+b).mean()), float(h.mean()), float(b.mean())]}, index=costs.index)
    save_table(sensitivity, output / 'section4_normal_formula_sensitivity.csv')
    return ns, costs, change

# %% Trace the original inventory code without changing its transitions
def trace_inventory(row, method):
    inv1, inv2, trace = [row['initial_stock'][0]], [], []
    rd = len(row['OPT_pred'])
    for r in range(rd):
        days = len(row['demand_hist'][r]) - (1 if r < rd-1 else 0)
        lead = int(round(row['vlt'][r]))
        for t in range(days):
            fallback = False
            if t == 0:
                if r == 0:
                    reference_stock = inv1[0]
                else:
                    try:
                        reference_stock = inv1[-lead-1]
                    except IndexError:
                        reference_stock = inv1[1]
                        fallback = True
                replen = int(round(row[method+'_pred'][r] - reference_stock))
            before = inv1[-1]
            demand = float(row['demand_hist'][r][t])
            after, counted, phase, cancelled_backlog = None, False, 'skipped_overlap', 0.
            if t < lead:
                if r == 0:
                    after = before-demand
                    inv1.append(after)
                    phase = 'before_first_arrival'
            elif t == lead:
                cancelled_backlog = max(-before, 0)
                after = (before if before >= 0 else 0) + replen - demand
                inv1.append(after)
                inv2.append(after)
                counted, phase = True, 'arrival'
            else:
                after = before-demand
                inv1.append(after)
                inv2.append(after)
                counted, phase = True, 'after_arrival'
            trace.append({'round':r, 'day_in_window':t, 'predicted_lead_days':lead,
                'prediction':float(row[method+'_pred'][r]), 'stock_reference':reference_stock,
                'computed_replenishment':replen, 'stock_before':before, 'demand':demand,
                'stock_after':after, 'phase':phase, 'counted_in_author_cost':counted,
                'backlog_removed_on_arrival':cancelled_backlog,
                'holding_cost':max(after,0) if counted else 0,
                'stockout_cost':9*max(-after,0) if counted else 0,
                'fallback_at_round_start':fallback if t == 0 else False})
    return [inv1[1:], inv2], pd.DataFrame(trace)

def inventory_audit(ns, output):
    grouped = ns['df_test']
    # Prefer a multi-order SKU so the trace includes cross-order stock references.
    sku = next((idx for idx, row in grouped.iterrows() if len(row['OPT_pred']) > 1), grouped.index[0])
    rows, examples = [], []
    for method in METHODS:
        negatives = arrivals = cancelled = fallback = skipped = 0
        for sku_id, row in grouped.iterrows():
            result, trace = trace_inventory(row, method)
            expected = ns['get_agginv'](row, method)
            assert np.allclose(result[0], expected[0]) and np.allclose(result[1], expected[1])
            starts = trace.day_in_window.eq(0)
            negatives += int((starts & trace.computed_replenishment.lt(0)).sum())
            arrivals += int(trace.phase.eq('arrival').sum())
            cancelled += int(trace.backlog_removed_on_arrival.gt(0).sum())
            fallback += int(trace.fallback_at_round_start.sum())
            skipped += int(trace.phase.eq('skipped_overlap').sum())
            if sku_id == sku:
                trace.insert(0, 'SKU', sku_id)
                trace.insert(1, 'method', method)
                examples.append(trace)
                assert np.isclose(trace.holding_cost.sum() + trace.stockout_cost.sum(),
                    sum(max(v,0)+9*max(-v,0) for v in result[1]))
        rows.append({'method':method, 'negative_replenishment_rounds':negatives, 'arrival_events':arrivals,
            'arrivals_dropping_negative_stock':cancelled, 'fallback_rounds':fallback,
            'skipped_overlap_points':skipped})
    save_table(pd.DataFrame(rows), output / 'inventory_implementation_audit.csv', index=False)
    save_table(pd.concat(examples, ignore_index=True), output / 'example_sku_inventory_trace.csv', index=False)
    return str(sku)

# %% Field experiment: author replay and explicitly labelled alternatives
def field_metrics(frame, include_last=False, observed_denominator=False):
    result = pd.DataFrame(index=frame.index)
    result['id'] = frame.sku_dc_pair_index
    sequences = frame.test_inv.map(ast.literal_eval).map(lambda x: np.asarray(x,dtype=float))
    evaluated = sequences if include_last else sequences.map(lambda x: x[:-1])
    result['holding_cost'] = evaluated.map(lambda x: np.maximum(x,0).sum())
    result['stockout_cost'] = evaluated.map(lambda x: 9*np.maximum(-x,0).sum())
    result['total_cost'] = result.holding_cost + result.stockout_cost
    result['turnover_raw'] = frame.ave_inv / frame.ave_demand.replace(0,np.nan)
    result['turnover'] = result.turnover_raw.clip(lower=0)
    denominator = evaluated.map(len) if observed_denominator else sequences.map(len)
    result['stockout_ratio'] = evaluated.map(lambda x: (x<0).sum()) / denominator.replace(0,np.nan)
    result['ave_demand'] = frame.ave_demand
    result['vlt_num'] = frame.vlt.map(lambda x: float(str(x).split()[0]))
    assert np.allclose(result.total_cost,result.holding_cost+result.stockout_cost)
    assert result.stockout_ratio.dropna().between(0,1).all()
    return result

def field_analysis(root, output, frames):
    regressions = []
    def capture(i, ns):
        if i in [23,24,25,26,27]:
            fit = ns['lr']
            regressions.append({'metric':METRICS[i-23], 'e2e_coefficient':fit.params['e2e'],
                'p_value':fit.pvalues['e2e'], 'n':int(fit.nobs), 'intercept':False,
                'standard_errors':'author nonrobust OLS'})
    ns, change = replay(root, 'Field_experiments_summary.ipynb', output, capture)
    save_table(pd.DataFrame(regressions), output / 'field_regression_author.csv', index=False)
    measured = {name:field_metrics(frame) for name,frame in frames.items()}
    summary = pd.DataFrame({name:m[METRICS].mean() for name,m in measured.items()})
    save_table(summary, output / 'field_means_author_convention.csv')
    # Verify independent recomputation against arrays retained in the original notebook.
    for j, metric in enumerate(METRICS):
        assert np.allclose(measured['treatment_post'][metric], ns['post_exp_e2e'][j])
        control_metric = 'turnover_raw' if metric == 'turnover' else metric
        assert np.allclose(measured['control_post'][control_metric], ns['post_exp_jd'][j])
        e_pre = ['e2e_holding_cost','e2e_stockout_cost','e2e_total_cost','e2e_turnover1','e2e_stockout_ratio'][j]
        c_pre = ['jd_holding_cost','jd_stockout_cost','jd_total_cost','jd_turnover1','jd_stockout_ratio'][j]
        assert np.allclose(measured['treatment_pre'][metric], ns[e_pre])
        assert np.allclose(measured['control_pre'][metric], ns[c_pre])
    tests, did_rows = [], []
    for j, metric in enumerate(METRICS):
        t = measured['treatment_post'][metric].to_numpy()
        c = measured['control_post'][metric].to_numpy()
        pooled = stats.ttest_ind(t,c,equal_var=True)
        welch = stats.ttest_ind(t,c,equal_var=False)
        tests.append({'metric':metric, 'treatment_mean':t.mean(), 'control_mean':c.mean(),
            'difference':t.mean()-c.mean(), 'change_percent':100*(t.mean()-c.mean())/c.mean(),
            'author_pooled_t':pooled.statistic, 'author_pooled_p':pooled.pvalue,
            'welch_t':welch.statistic, 'welch_p':welch.pvalue})
        author_post = np.array(ns['post_exp_e2e'][j])-np.array(ns['post_exp_jd'][j])
        author_pre = measured['treatment_pre'][metric].to_numpy()-measured['control_pre'][metric].to_numpy()
        original_test = stats.ttest_ind(author_post,author_pre)
        did = (summary.loc[metric,'treatment_post']-summary.loc[metric,'treatment_pre']
               -summary.loc[metric,'control_post']+summary.loc[metric,'control_pre'])
        did_rows.append({'metric':metric, 'author_point_estimate':author_post.mean()-author_pre.mean(),
            'author_ttest_p_unverified_pairing':original_test.pvalue,
            'consistent_clipped_turnover_point_estimate':did,
            'audited_p_value':np.nan,
            'inference_note':'No audited p-value: longitudinal identity and valid sampling dependence unverified.'})
    save_table(pd.DataFrame(tests), output / 'field_post_ttests.csv', index=False)
    save_table(pd.DataFrame(did_rows), output / 'field_did.csv', index=False)
    post = pd.concat([measured['treatment_post'].assign(e2e=1), measured['control_post'].assign(e2e=0)],ignore_index=True)
    X = sm.add_constant(post[['e2e','ave_demand','vlt_num']], has_constant='add')
    revised = []
    for metric in METRICS:
        fit = sm.OLS(post[metric],X).fit(cov_type='HC3')
        lo, hi = fit.conf_int().loc['e2e']
        revised.append({'metric':metric, 'e2e_coefficient':fit.params['e2e'], 'p_value_HC3':fit.pvalues['e2e'],
                        'CI_low':lo,'CI_high':hi,'n':int(fit.nobs),'intercept':True})
    save_table(pd.DataFrame(revised), output / 'field_regression_intercept_HC3.csv', index=False)
    conventions = []
    for group, frame in frames.items():
        variants = [('author_Tminus1_cost_T_denominator',field_metrics(frame)),
                    ('same_cost_evaluated_points_denominator',field_metrics(frame,False,True)),
                    ('all_inventory_points',field_metrics(frame,True,True))]
        for convention, metrics in variants:
            conventions.append({'group':group,'convention':convention,'rows':len(metrics),
                'valid_stockout_ratio_rows':int(metrics.stockout_ratio.notna().sum()),
                **metrics[METRICS].mean().to_dict()})
        save_table(measured[group], output / f'field_rows_{group}.csv', index=False)
    save_table(pd.DataFrame(conventions),output/'field_measurement_sensitivity.csv',index=False)
    return summary, pd.DataFrame(tests), pd.DataFrame(revised), pd.DataFrame(did_rows), change

# %% Matching diagnostics and plots
def matching_analysis(root,output,frames):
    ns, change = replay(root,'Propensity_score_matching.ipynb',output,skip=(12,15))
    treated = frames['treatment_post'][['ave_demand']].copy()
    treated['vlt_num'] = frames['treatment_post'].vlt.map(lambda s: float(s.split()[0]))
    candidates = pd.read_csv(root/'control_demand_vlt.csv')
    covariates = ['ave_demand','vlt_num']
    control = candidates[covariates]
    assert np.isfinite(control.to_numpy()).all()
    prop = ns['propensity']
    p_t = prop.predict_proba(treated)[:,1]
    p_c = prop.predict_proba(control)[:,1]
    idx_original = ns['matched_idx']
    # Same fitted propensity model, change only matching distance to scalar score.
    idx_score = NearestNeighbors(n_neighbors=1).fit(p_c[:,None]).kneighbors(p_t[:,None],return_distance=False).ravel()
    balance, summaries, pairs = [], [], []
    for method, idx in [('before',np.arange(len(control))),('author_covariate_NN',idx_original),('propensity_score_NN',idx_score)]:
        matched = control.iloc[idx]
        for col in covariates:
            scale = np.sqrt((treated[col].var(ddof=1)+control[col].var(ddof=1))/2)
            balance.append({'matching':method,'covariate':col,'treated_mean':treated[col].mean(),
                'control_mean':matched[col].mean(),'standardization':'fixed pooled pre-match SD',
                'SMD':(treated[col].mean()-matched[col].mean())/scale})
        summaries.append({'matching':method,'n_treated':len(treated),'control_rows_used':len(idx),
            'distinct_control_rows':len(np.unique(idx)), 'reuse_count':len(idx)-len(np.unique(idx)),
            'max_score_distance':float(np.max(np.abs(p_t-p_c[idx]))) if method!='before' else np.nan})
        if method!='before':
            for i,j in enumerate(idx):
                pairs.append({'matching':method,'treatment_row':i,'treatment_anonymized_id':frames['treatment_post'].sku_dc_pair_index.iloc[i],
                    'candidate_row':int(j),'candidate_source_index':candidates.iloc[j,0],
                    'treatment_score':p_t[i],'control_score':p_c[j],'score_distance':abs(p_t[i]-p_c[j]),
                    'linked_order_outcomes_available':False})
    balance=pd.DataFrame(balance)
    save_table(balance,output/'matching_balance.csv',index=False)
    save_table(pd.DataFrame(summaries),output/'matching_summary.csv',index=False)
    save_table(pd.DataFrame(pairs),output/'matching_pairs.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.4),layout='constrained')
    score_bins=np.linspace(min(p_c.min(),p_t.min()),max(p_c.max(),p_t.max()),26)
    for ax,(label,idx) in zip(axes,[('Before matching',np.arange(len(control))),('Author: covariate NN',idx_original),('Propensity-score NN',idx_score)]):
        ax.hist(p_c[idx],bins=score_bins,density=True,alpha=.5,label='Control',color='#D98448')
        ax.hist(p_t,bins=score_bins,density=True,histtype='step',linewidth=1.7,label='Treatment',color='#377EA0')
        ax.set(title=label,xlabel='Propensity score',ylabel='Density')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.legend(fontsize=8)
    fig.savefig(output/'matching_scores.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.6),layout='constrained')
    for method,marker in [('before','o'),('author_covariate_NN','s'),('propensity_score_NN','^')]:
        sub=balance[balance.matching.eq(method)]
        ax.plot(sub.SMD.abs(),sub.covariate,marker=marker,label=method)
    ax.axvline(.1,color='grey',linestyle='--',linewidth=1)
    ax.set(xlabel='Absolute standardized mean difference',title='Covariate balance (fixed pre-match scale)')
    ax.legend(fontsize=8)
    fig.savefig(output/'matching_balance.png',dpi=180)
    plt.close(fig)
    return balance,pd.DataFrame(summaries),change

# %% Reports and entry point
def markdown_table(frame):
    out=frame.copy().round(4)
    lines=['| '+ ' | '.join(map(str,out.columns))+' |','| '+' | '.join(['---']*len(out.columns))+' |']
    lines.extend('| '+' | '.join(map(str,row))+' |' for row in out.itertuples(index=False,name=None))
    return '\n'.join(lines)

def run_all(data_dir,output_dir):
    root=Path(data_dir).resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output=Path(output_dir).resolve()/stamp
    output.mkdir(parents=True,exist_ok=False)
    print('[1/5] Checking input data',flush=True)
    field=audit_inputs(root,output)
    print('[2/5] Replaying Section 4 and tracing all SKU/method trajectories',flush=True)
    ns,costs,c1=section4(root,output)
    example=inventory_audit(ns,output)
    print('[3/5] Replaying field statistics and labelled alternatives',flush=True)
    summary,tests,regression,did,c2=field_analysis(root,output,field)
    print('[4/5] Replaying matching and comparing propensity-score matching',flush=True)
    balance,matching,c3=matching_analysis(root,output,field)
    print('[5/5] Saving report and validation record',flush=True)
    metadata={'time_utc':stamp,'python':platform.python_version(),
        'packages':{n:version(n) for n in ['numpy','pandas','scipy','matplotlib','seaborn','scikit-learn','statsmodels']},
        'input_hashes':{p.name:digest(p) for p in root.glob('*.csv')},
        'notebook_replays':[c1,c2,c3], 'example_sku':example,
        'validation':{'section4_reference':'passed, within 0.011 of rounded reference',
            'inventory_trace':'all 2607 SKUs x 7 methods equal author trajectories',
            'field_arrays':'all four group arrays agree with author calculations',
            'cost_identity':'holding + stockout = total', 'data_identity_linkage':'unverified'},
        'scope':'CPU evaluation and analysis of disclosed subset, no neural network training'}
    (output/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    report='''# T1 CPU复现结果

本次完成公开数据的CPU评估、实现审计、实地实验统计和匹配诊断。没有重新训练神经网络。

## 1. 第4节：作者原始评估结果

'''+markdown_table(costs.T.reset_index(names='method'))+'''

输入为3000行、2607个不同SKU标识。表内OPT沿用作者命名，未认证为其他仿真规则下的严格最优下界。

`section4_normal_formula_sensitivity.csv` 只将Normal预测改为论文公式（7）并使用90%分位数；仍保留作者库存转移和截断取整规则，不能称为完整修正版仿真。

## 2. 库存实现审计

对全部2607个SKU、7个方法逐一核对：新增跟踪器输出与作者库存函数一致，逐期正负库存成本可复算。示例详见 `example_sku_inventory_trace.csv`。所有方法的异常计数在 `inventory_implementation_audit.csv`。

代码会在某些情况下算出负补货量，到货时可能移除此前负库存，并跳过部分重叠窗口点；这些是对作者代码行为的记录，并非直接认定每项都是错误。理论模型的积压需求、字段含义与这个测试实现仍须区分。

## 3. 实地实验：四组均值（作者口径）

'''+markdown_table(summary.reset_index(names='metric'))+'''

`field_post_ttests.csv` 给出作者等方差t检验及Welch对照。`field_regression_author.csv` 保留无截距、普通标准误；另表 `field_regression_intercept_HC3.csv` 加入截距并采用HC3稳健标准误，是明确标注的分析变体。两类分析均不是新的随机实验因果认证。

## 4. DiD：描述性点估计

'''+markdown_table(did[['metric','author_point_estimate','consistent_clipped_turnover_point_estimate']])+'''

作者对实验后对照组的周转指标未截断，其他相关数组截断为非负，故两种周转DiD点估计可能不同。原代码按行相减后做t检验，其p值仅为代码复算，见 `field_did.csv` 中带有 unverified_pairing 的列；本次没有为修正版提供DiD显著性结论。

四组文件均使用重置的数值索引。相同编号本身不能证明同一个体前后对应或处理组与对照组匹配。前后观察窗口长度、日期范围见 `input_audit.csv`，也不能据此直接对齐到论文完整实验窗口。严谨面板DiD及聚类标准误需要可验证个体ID和时间结构；当前结果是四组均值的描述性对比。

## 5. 匹配诊断

'''+markdown_table(matching)+'''

作者代码在原始协变量上做一对一、有放回近邻匹配；对照版本使用同一个拟合出的倾向得分做有放回近邻匹配。匹配前后的标准化均值差固定使用匹配前合并标准差，见 `matching_balance.csv`。

候选对照文件没有可验证的订单结果映射，不能把新匹配对象强行连接到给定500条对照订单后重新估计效果。匹配变量来自当前提供的数据，未认证为处理前协变量，因此这些是描述性匹配诊断。

## 6. 口径敏感性与兼容性

作者成本累计前T-1个库存点，缺货率分母为T；`field_measurement_sensitivity.csv` 另给出分母改为实际计费点数、以及计入全部库存点两种敏感性分析。实验后对照组有2条记录仅含1个库存点，原代码将其成本与缺货率算为0；在按T-1计费点计算缺货率的变体中，这两条记录没有可评价点，记为缺失而非0，并报告有效样本数。原始序列边界含义尚不充分明确，不能自动选任一变体为真值。

仅对旧版pandas写法 `drop('y',1)` 作等价兼容替换；原始匹配Notebook的两个画图单元跳过，另生成PNG诊断图。作者统计计算顺序保留，源码哈希、具体修改和环境版本见 `run_metadata.json`；详细标准输出见 `.log` 文件。

## 7. 已完成与后续边界

- 已完成：离线评估、Normal公式单项对照、全部轨迹核对、四组样本统计、原始回归与截距HC3对照、DiD描述性复算、两种匹配方式及平衡性诊断。
- 完整E2E训练、训练标签的端到端重建、缺失身份映射下的严格因果推断不属于本包已完成结果。训练集和模型训练源码仍需补齐。
- Colab只需CPU。上传原始ZIP，运行专用Notebook后下载结果；本地已执行验证，Colab浏览器上传和下载需由用户操作。
'''
    (output/'CPU复现报告.md').write_text(report,encoding='utf-8')
    print('Completed:',output,flush=True)
    print(costs.round(2).to_string())
    return output

# %% CLI only
if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent.parent/'replication_package_MS-FTBDA-20-01585')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    args=parser.parse_args()
    run_all(args.data_dir,args.output_dir)
