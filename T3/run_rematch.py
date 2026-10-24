from pathlib import Path
import numpy as np,numba
from run_experiments import ROOT,load
from model import Config,make_environment
from learning import rematch_batch
if __name__=='__main__':
    numba.set_num_threads(6);spec,data=load(next((ROOT/'results/runs').glob('baseline_*')))
    e=make_environment(Config());seed=np.arange(1000,dtype=np.uint64)+991001
    r=rematch_batch(data['qs'],e['rewards'],.15,.95,15,100000,10000000,seed,40000,100)
    np.savez_compressed(ROOT/'results/rematch.npz',policies=r[0],states=r[1],steps=r[2],converged=r[3],histories=r[4],seeds=seed)
    print('Rematching:',len(r[2]),'converged:',r[3].sum(),'mean steps:',r[2].mean(),flush=True)
