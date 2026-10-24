"""Run independently reconstructed experiments; cached, deterministic CPU execution."""
import os,json,time,hashlib,argparse
from pathlib import Path
from dataclasses import replace
import numpy as np
import numba
from model import Config,make_environment
from learning import train_batch,rematch_batch
ROOT=Path(__file__).resolve().parent

def run(name,cfg,repetitions,save_q=False,seed=20261021):
    out=ROOT/'results'/'runs';out.mkdir(parents=True,exist_ok=True)
    spec=dict(name=name,config=cfg.mapping(),repetitions=repetitions,seed=seed,save_q=save_q)
    signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:12]
    folder=out/(name+'_'+signature);folder.mkdir(exist_ok=True)
    (folder/'spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    e=make_environment(cfg);start=time.time();done=0
    for first in range(0,repetitions,32):
        path=folder/f'batch_{first:05d}.npz';count=min(32,repetitions-first)
        if not path.exists():
            seeds=np.arange(seed+first,seed+first+count,dtype=np.uint64)
            result=train_batch(e['q_initial'],e['rewards'],e['card'],cfg.m,cfg.memory,cfg.alpha,cfg.beta,cfg.delta,cfg.stable_periods,cfg.max_steps,seeds,save_q,cfg.history_end,cfg.history_every,cfg.exploration,cfg.entry_rho)
            np.savez_compressed(path,policies=result[0],states=result[1],steps=result[2],converged=result[3],histories=result[4],qs=result[5],seeds=seeds)
        done+=count
        print(f'{name}: {done}/{repetitions}; elapsed {time.time()-start:.1f}s',flush=True)
    return folder

def load(folder):
    spec=json.loads((Path(folder)/'spec.json').read_text(encoding='utf-8'))
    chunks=[np.load(p) for p in sorted(Path(folder).glob('batch_*.npz'))]
    data={k:np.concatenate([x[k] for x in chunks]) for k in chunks[0].files if k!='qs'}
    if spec['save_q']:data['qs']=np.concatenate([x['qs'] for x in chunks])
    return spec,data

def suite(profile):
    base=Config();entries=[];seed=20261021
    def add(name,cfg,n,q=False):
        nonlocal seed
        folder=run(name,cfg,n,q,seed);seed+=100000
        entries.append(dict(name=name,path=str(folder.relative_to(ROOT)),config=cfg.mapping(),repetitions=n))
        (ROOT/'results'/'manifest.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    add('baseline',base,1000,True)
    if profile=='baseline':return
    n=1000
    grid=100 if profile=='paper' else 7
    for ai,a in enumerate(np.linspace(.025,.25,grid)):
        for bi,b in enumerate(np.linspace(2e-7,2e-5,grid)):
            add(f'grid_{ai}_{bi}',replace(base,alpha=float(a),beta=float(b),history_end=0,history_every=0),1000 if profile=='paper' else 64)
    for d in [0,.1,.2,.3,.35,.4,.5,.6,.7,.8,.9,.95,.99]:
        if d==.95:continue
        add(f'delta_{d}',replace(base,delta=d,history_end=0,history_every=0),n)
    for c in [.875,.75,.625,.5,.25]:
        add(f'cost_{c}',replace(base,costs=(1.,c),history_end=0,history_every=0),n)
    for players in [3,4]:
        add(f'players_{players}',replace(base,n=players,costs=(1.,)*players,qualities=(2.,)*players,history_end=0,history_every=0),n if profile=='paper' or players==3 else 256)
    for shock in [.15,.25]:add(f'shock_{shock}',replace(base,shock=shock,history_end=0,history_every=0),n)
    for players in [2,3]:
        for rho in [.001,.0001]:add(f'entry_{players}_{rho}',replace(base,n=players,costs=(1.,)*players,qualities=(2.,)*players,entry_rho=rho,history_end=0,history_every=0),n)
    for mu in [0,.05,.1,.2,.5]:add(f'mu_{mu}',replace(base,mu=mu,history_end=0,history_every=0),n)
    for init in ['nash','grim','optimistic']:
        add(f'init_{init}',replace(base,init=init,exploration=(init!='optimistic'),history_end=0,history_every=0),n)
    for xi in [0,.5,1.]:add(f'xi_{xi}',replace(base,xi=xi,history_end=0,history_every=0),n)
    add('lower_cost',replace(base,lower_at_cost=True,history_end=0,history_every=0),n)
    for m in [50,100]:add(f'actions_{m}',replace(base,m=m,history_end=0,history_every=0),n)
    add('memory_zero',replace(base,memory=0,history_end=0,history_every=0),n)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['baseline','local','paper'],default='local');p.add_argument('--threads',type=int,default=10)
    args=p.parse_args();numba.set_num_threads(args.threads);suite(args.profile)
