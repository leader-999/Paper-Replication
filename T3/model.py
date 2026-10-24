"""Independent reconstruction of Calvano et al. (AER 2020), equations (4)-(9)."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import root, brentq

@dataclass
class Config:
    n: int = 2
    m: int = 15
    memory: int = 1
    alpha: float = .15
    beta: float = 4e-6
    delta: float = .95
    mu: float = .25
    xi: float = .1
    costs: tuple = (1.,1.)
    qualities: tuple = (2.,2.)
    a0: float = 0.
    shock: float = 0.
    init: str = 'uniform'
    exploration: bool = True
    lower_at_cost: bool = False
    stable_periods: int = 100_000
    max_steps: int = 1_000_000_000
    entry_rho: float = 0.
    history_end: int = 1_500_000
    history_every: int = 1000
    def mapping(self):return asdict(self)

def demand(prices,qualities,mu,a0=0.):
    prices=np.asarray(prices,dtype=float);qualities=np.asarray(qualities,dtype=float)
    if mu==0:
        u=qualities-prices;best=max(float(u.max()),a0)
        winner=np.isclose(u,best,atol=1e-10,rtol=0)
        count=winner.sum()+int(abs(best-a0)<1e-10)
        return winner.astype(float)/count
    u=(qualities-prices)/mu;outside=a0/mu
    shift=max(float(u.max()),outside)
    numerator=np.exp(u-shift)
    return numerator/(np.exp(outside-shift)+numerator.sum())

def benchmarks(costs,qualities,mu,a0=0.):
    c=np.asarray(costs);a=np.asarray(qualities);n=len(c)
    if mu==0:
        if not np.allclose(c,c[0]) or not np.allclose(a,a[0]):
            raise ValueError('The mu=0 reconstruction supports symmetric firms only.')
        pn=c.copy();pm=np.repeat(a[0]-a0-1e-9,n)
    else:
        def foc(p):
            q=demand(p,a,mu,a0)
            return p-c-mu/(1-q)
        sol=root(foc,c+mu*2,tol=1e-11)
        if not sol.success and np.linalg.norm(foc(sol.x))>1e-8:raise RuntimeError('Nash solver failed')
        pn=sol.x
        def monopoly_foc(markup):
            q=demand(c+markup,a,mu,a0)
            return markup*(1-q.sum())-mu
        markup=brentq(monopoly_foc,mu,20+max(abs(a-c)))
        pm=c+markup
    qn=demand(pn,a,mu,a0);qm=demand(pm,a,mu,a0)
    return pn,pm,(pn-c)*qn,(pm-c)*qm,qn

def make_environment(cfg):
    assert 0<=cfg.delta<1 and cfg.memory>=0
    c=np.asarray(cfg.costs,dtype=float);a=np.asarray(cfg.qualities,dtype=float)
    assert len(c)==cfg.n and len(a)==cfg.n
    pn,pm,profit_n,profit_m,qn=benchmarks(c,a,cfg.mu,cfg.a0)
    lower=pn-cfg.xi*(pm-pn)
    if cfg.lower_at_cost:lower=c-.01
    grids=np.array([np.linspace(lower[i],pm[i]+cfg.xi*(pm[i]-pn[i]),cfg.m) for i in range(cfg.n)])
    card=cfg.m+(1 if cfg.entry_rho else 0)
    joint_count=card**cfg.n
    state_count=joint_count**cfg.memory if cfg.memory else 1
    actions=np.zeros((joint_count,cfg.n),dtype=np.int64)
    for j in range(joint_count):
        x=j
        for i in range(cfg.n-1,-1,-1):actions[j,i]=x%card;x//=card
    shocks=np.array([-cfg.shock,0.,cfg.shock]) if cfg.shock else np.array([0.])
    rewards=np.zeros((len(shocks),joint_count,cfg.n))
    for z,shock in enumerate(shocks):
        for j,choice in enumerate(actions):
            active=choice<cfg.m
            if not active.any():continue
            pp=np.array([grids[i,choice[i]] for i in range(cfg.n) if active[i]])
            q=demand(pp,a[active],cfg.mu,cfg.a0+shock)
            rewards[z,j,active]=(pp-c[active])*q
    average=rewards.mean(axis=0)
    q_initial=np.empty((cfg.n,state_count,cfg.m))
    for i in range(cfg.n):
        for act in range(cfg.m):
            mask=actions[:,i]==act
            for other in range(cfg.n):
                if other!=i and (not cfg.entry_rho or other!=cfg.n-1):mask &= actions[:,other]<cfg.m
            # For entry experiments initialization averages over all valid rival actions;
            # inactive outsiders count as one action. This is a declared reconstruction choice.
            baseline=float(average[mask,i].mean())/(1-cfg.delta)
            q_initial[i,:,act]=baseline
    nash_indices=np.array([np.argmin(abs(grids[i]-pn[i])) for i in range(cfg.n)])
    monopoly_indices=np.array([np.argmin(abs(grids[i]-pm[i])) for i in range(cfg.n)])
    if cfg.init in ['nash','grim']:
        for state in range(state_count):
            state_choices=actions[state%joint_count] if cfg.memory else nash_indices
            coop=np.all(state_choices==monopoly_indices)
            for i in range(cfg.n):
                rival=monopoly_indices.copy() if (cfg.init=='grim' and coop) else nash_indices.copy()
                for act in range(cfg.m):
                    rival[i]=act;code=0
                    for x in rival:code=code*card+int(x)
                    future=nash_indices.copy()
                    if cfg.init=='grim' and coop and act==monopoly_indices[i]:future=monopoly_indices.copy()
                    fcode=0
                    for x in future:fcode=fcode*card+int(x)
                    q_initial[i,state,act]=average[code,i]+cfg.delta*average[fcode,i]/(1-cfg.delta)
    elif cfg.init=='optimistic':q_initial.fill(100.)
    if cfg.entry_rho:
        # Benchmarks average aggregate profits across n and n-1 active firms (50% each).
        _,_,small_n,small_m,_=benchmarks(c[:-1],a[:-1],cfg.mu,cfg.a0)
        benchmark_n=.5*(profit_n.sum()+small_n.sum())/cfg.n
        benchmark_m=.5*(profit_m.sum()+small_m.sum())/cfg.n
    elif cfg.shock:
        # Re-optimize prices for each shock to construct expected static normalizers.
        benchmark_n=np.mean([benchmarks(c,a,cfg.mu,cfg.a0+z)[2].mean() for z in shocks])
        benchmark_m=np.mean([benchmarks(c,a,cfg.mu,cfg.a0+z)[3].mean() for z in shocks])
    else:benchmark_n,benchmark_m=profit_n.mean(),profit_m.mean()
    return dict(config=cfg.mapping(),grids=grids,rewards=rewards,actions=actions,
        q_initial=q_initial,card=card,state_count=state_count,joint_count=joint_count,
        pn=pn,pm=pm,profit_n=profit_n,profit_m=profit_m,nash_share=qn,
        normalizer_n=float(benchmark_n),normalizer_m=float(benchmark_m))
