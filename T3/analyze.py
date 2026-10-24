"""Limit cycles, exact Bellman best responses, and frozen-policy deviations."""
import numpy as np
from numba import njit,prange
from model import Config,make_environment

@njit(cache=True)
def cycle(policy,start,card,memory):
    n,S=policy.shape;seen=np.full(S,-1,np.int64);path=np.empty(S+1,np.int64);s=start;t=0;J=card**n;mod=S//J if memory else 1
    while seen[s]<0:
        seen[s]=t;path[t]=s;t+=1;j=0
        for i in range(n):j=j*card+policy[i,s]
        s=(s%mod)*J+j if memory else 0
    return path[seen[s]:t]

@njit(cache=True,parallel=True)
def cycle_metrics(policies,states,rewards,card,memory):
    N,n,S=policies.shape;length=np.zeros(N,np.int64);profits=np.zeros((N,n));sym=np.zeros(N,np.bool_)
    for r in prange(N):
        path=cycle(policies[r],states[r],card,memory);length[r]=len(path)
        for s in path:
            joint=0
            for i in range(n):joint=joint*card+policies[r,i,s]
            for i in range(n):profits[r,i]+=rewards[0,joint,i]/len(path)
        if len(path)==1:
            sym[r]=True
            for i in range(1,n):
                if policies[r,i,path[0]]!=policies[r,0,path[0]]:sym[r]=False
    return length,profits,sym


def metrics(env,data,losses=False):
    cfg=env['config'];length,profits,sym=cycle_metrics(data['policies'],data['states'],env['rewards'].mean(axis=0,keepdims=True),env['card'],cfg['memory'])
    # Entry needs stochastic state transitions, so evaluate its frozen policy by MC.
    if cfg['entry_rho']:
        profits=entry_profit(data['policies'],data['states'],env['rewards'],env['card'],cfg['entry_rho'])
        length[:]=0;sym[:]=False
    gain=(profits.mean(axis=1)-env['normalizer_n'])/(env['normalizer_m']-env['normalizer_n'])
    result=dict(length=length,profits=profits,symmetric=sym,gain=gain)
    if losses:
        on,off,eq=paper_losses(data['policies'],data['states'],env['rewards'].mean(axis=0),cfg['m'],cfg['delta'],cfg['memory'])
        result.update(on_loss=on,off_loss=off,equilibrium=eq)
    return result

def paper_losses(policies,states,reward,m,delta,memory):
    """Paper's linear-system Q: deviate once, then both resume limit strategies.

    This differs from solving a globally optimal, repeatedly deviating MDP policy.
    A3.3-style on-path test compares chosen action with max of this true Q row.
    """
    from scipy.sparse import eye,csr_matrix
    from scipy.sparse.linalg import spsolve
    N,n,S=policies.shape;on=np.zeros(N);off=np.zeros(N);eq=np.ones(N,dtype=bool)
    ss=np.arange(S);J=m**n;mod=S//J if memory else 1
    for r,p in enumerate(policies):
        j=np.zeros(S,dtype=int)
        for k in range(n):j=j*m+p[k]
        nxt=(ss%mod)*J+j if memory else np.zeros(S,dtype=int)
        matrix=eye(S,format='csr')-delta*csr_matrix((np.ones(S),(ss,nxt)),shape=(S,S))
        value=spsolve(matrix,reward[j])
        path=cycle(p,states[r],m,memory)
        for i in range(n):
            actionj=np.zeros((S,m),dtype=int)
            for k in range(n):actionj=actionj*m+(np.arange(m)[None,:] if k==i else p[k,:,None])
            trans=(ss[:,None]%mod)*J+actionj if memory else np.zeros((S,m),dtype=int)
            trueq=reward[actionj,i]+delta*value[trans,i]
            maximum=trueq.max(axis=1);chosen=trueq[ss,p[i]]
            loss=np.maximum(0,maximum-chosen)/np.maximum(abs(maximum),1e-12)
            on[r]+=loss[path].mean()/n;off[r]+=loss.mean()/n
            if np.any(maximum[path]-chosen[path]>1e-7):eq[r]=False
    return on,off,eq

@njit(cache=True,parallel=True)
def entry_profit(policies,states,rewards,card,rho):
    N,n,S=policies.shape;out=np.zeros((N,n))
    for r in prange(N):
        np.random.seed(20261021+r);s=states[r]
        # 200k periods gives ~20 switches even at rho=.0001; MC error retained.
        for t in range(210000):
            active=(s%card)!=card-1
            if np.random.random()<rho:active=not active
            j=0
            for i in range(n):
                a=policies[r,i,s]
                if i==n-1 and not active:a=card-1
                j=j*card+a
            if t>=10000:
                for i in range(n):out[r,i]+=rewards[0,j,i]/200000
            s=j
    return out

def response(env,policy,start,mode='best',periods=30,forced=None):
    """t=0 normal cycle action; t=1 force firm0 while firm1 follows its policy."""
    m=env['config']['m'];grid=env['grids'];reward=env['rewards'].mean(axis=0)
    path=cycle(policy,start,m,env['config']['memory']);trajectories=[]
    for state in path:
        pre=policy[:,state].astype(int);s=int(pre[0]*m+pre[1]);track=np.empty((periods+1,2));track[0]=grid[np.arange(2),pre]
        for t in range(1,periods+1):
            act=policy[:,s].astype(int)
            if t==1:
                if forced is not None:act[0]=forced
                elif mode=='small':act[0]=max(0,int(pre[0])-1)
                else:act[0]=int(np.argmax(reward[np.arange(m)*m+pre[1],0]))
            track[t]=grid[np.arange(2),act];s=int(act[0]*m+act[1])
        trajectories.append(track)
    return np.mean(trajectories,axis=0)

def discounted_value(policy,start,reward,delta,m):
    seen={};seq=[];s=int(start)
    while s not in seen:
        seen[s]=len(seq);j=int(policy[0,s]*m+policy[1,s]);seq.append(float(reward[j,0]));s=j
    k=seen[s];a=np.asarray(seq);prefix=np.dot(delta**np.arange(k),a[:k])
    rest=np.dot(delta**np.arange(len(a)-k),a[k:])/(1-delta**(len(a)-k))
    return prefix+delta**k*rest

def deviation_tables(env,data,met):
    m=env['config']['m'];delta=env['config']['delta'];grid=env['grids'];reward=env['rewards'][0];rows=[]
    for r in np.flatnonzero((met['length']==1)&met['symmetric']&data['converged']):
        p=data['policies'][r];s=int(cycle(p,data['states'][r],m,1)[0]);a=int(p[0,s]);baseline=discounted_value(p,s,reward,delta,m)
        for d in range(a):
            joint=d*m+a;nextactions=p[:,joint]
            value=reward[joint,0]+delta*discounted_value(p,joint,reward,delta,m)
            rows.append(dict(session=int(r),pre_index=a,dev_index=d,pre_price=grid[0,a],dev_price=grid[0,d],nondev_change=grid[1,nextactions[1]]/grid[1,a]-1,dev_change=grid[0,nextactions[0]]/grid[0,d]-1,discount_gain=value/baseline-1,unprofitable=value<=baseline+1e-9))
    return rows
