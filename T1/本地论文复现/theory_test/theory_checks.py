"""Numerical checks of Appendix A; controlled tests are not empirical results."""
from functools import lru_cache
import json
import numpy as np
import pandas as pd

def interval_cost(demand, stock, order, h, b):
    end = stock + order - np.cumsum(demand)
    return float((h*np.maximum(end,0) + b*np.maximum(-end,0)).sum())

def closed_form(demand, stock, h, b, nonnegative=True):
    cumulative=np.cumsum(demand)
    k=int(b*len(demand)//(h+b))
    order=float(cumulative[k]-stock)
    return max(order,0.) if nonnegative else order

def greedy_solution(demand, cuts, stock, h, b):
    orders=[];cost=0.
    for left,right in zip(cuts[:-1],cuts[1:]):
        segment=demand[left:right]
        order=closed_form(segment,stock,h,b)
        orders.append(order)
        cost+=interval_cost(segment,stock,order,h,b)
        stock=stock+order-int(sum(segment))
    return cost,orders

def exhaustive_dp(demand, cuts, initial, h, b):
    """Enumerate all nonnegative integer actions up to a dominance bound."""
    @lru_cache(None)
    def dp(m, stock):
        if m==len(cuts)-1:
            return 0.
        left,right=cuts[m:m+2]
        segment=demand[left:right]
        # More than all remaining demand incurs avoidable holding cost, h>0.
        upper=max(sum(demand[left:])-stock,0)
        best=float('inf')
        for order in range(upper+1):
            cost=interval_cost(segment,stock,order,h,b)
            cost+=dp(m+1,stock+order-sum(segment))
            best=min(best,cost)
        return best
    return dp(0,initial)

def run_checks(data_dir, output, seed=20261020, n_cases=600):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n_cases):
        n=int(rng.integers(1,8))
        demand=tuple(map(int,rng.integers(0,4,n)))
        stock=int(rng.integers(-3,7))
        h,b=[(1,1),(1,3),(1,9),(2,3),(12,88)][i%5]
        a=closed_form(demand,stock,h,b,False)
        cost=interval_cost(demand,stock,a,h,b)
        best=min(interval_cost(demand,stock,q,h,b) for q in range(-stock-2,sum(demand)-stock+3))
        rows.append({'test':'Lemma A.1 / controlled integer enumeration','case':i,
                     'demand':json.dumps(demand),'initial_stock':stock,'h':h,'b':b,
                     'closed_form_cost':cost,'independent_optimum':best,'pass':abs(cost-best)<1e-8})
        if n>1:
            num_cuts=int(rng.integers(1,min(n,4)))
            interior=sorted(map(int,rng.choice(np.arange(1,n),size=num_cuts,replace=False)))
            cuts=tuple([0]+interior+[n])
            greedy,orders=greedy_solution(demand,cuts,stock,h,b)
            optimum=exhaustive_dp(demand,cuts,stock,h,b)
            rows.append({'test':'Theorem 1 / independent integer DP','case':i,
                'demand':json.dumps(demand),'initial_stock':stock,'h':h,'b':b,
                'arrival_interval_cuts':json.dumps(cuts),'closed_form_orders':json.dumps(orders),
                'closed_form_cost':greedy,'independent_optimum':optimum,'pass':abs(greedy-optimum)<1e-8})
    # Also test Lemma A.1 on actual example daily sales supplied in the archive.
    sales=pd.read_csv(data_dir/'E2E-model-data/Data/rdc_sales_1320.csv')
    for _,row in sales.iterrows():
        seq=row.iloc[1:].to_numpy(dtype=float)
        for start in [0,30,100,200,400]:
            for length in [1,2,5,14]:
                demand=seq[start:start+length]
                if len(demand)!=length: continue
                for h,b in [(1,1),(1,9),(12,88)]:
                    stock=10.
                    order=closed_form(demand,stock,h,b,False)
                    cost=interval_cost(demand,stock,order,h,b)
                    candidates=np.cumsum(demand)-stock
                    optimum=min(interval_cost(demand,stock,q,h,b) for q in candidates)
                    rows.append({'test':'Lemma A.1 / supplied daily sales breakpoints',
                        'case':f'{row.iloc[0]}:{start}:{length}', 'demand':json.dumps(demand.tolist()),
                        'initial_stock':stock,'h':h,'b':b,'closed_form_cost':cost,
                        'independent_optimum':optimum,'pass':abs(cost-optimum)<1e-8})
    results=pd.DataFrame(rows)
    results.to_csv(output/'theory_numerical_checks.csv',index=False,encoding='utf-8-sig')
    summary=results.groupby('test',sort=False).agg(cases=('pass','size'),passed=('pass','sum')).reset_index()
    summary['failed']=summary.cases-summary.passed
    summary.to_csv(output/'theory_check_summary.csv',index=False,encoding='utf-8-sig')
    if not results['pass'].all():
        raise AssertionError('Closed-form/DP disagreement; inspect theory_numerical_checks.csv')
    return summary
