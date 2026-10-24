"""Numba CPU kernel. Session RNG: xorshift64, independent deterministic seeds.

Convergence checks changes in the greedy strategy at ANY visited state, not only
repeated realized actions. Only one row can change per update, making this exact.
"""
import numpy as np
from numba import njit,prange

@njit(inline='always')
def draw(r):
    r ^= r << np.uint64(13)
    r ^= r >> np.uint64(7)
    r ^= r << np.uint64(17)
    return r, float(r >> np.uint64(11))*(1./9007199254740992.)

@njit(inline='always')
def argmax_first(row):
    a=0;v=row[0]
    for k in range(1,len(row)):
        if row[k]>v:a=k;v=row[k]
    return a

@njit(cache=True,parallel=True)
def train_batch(q_initial,rewards,card,m,memory,alpha,beta,delta,stable,max_steps,
                seeds,save_q,history_end,history_every,exploration,entry_rho):
    count=len(seeds);n=q_initial.shape[0];S=q_initial.shape[1]
    policies=np.empty((count,n,S),np.int16)
    final_states=np.zeros(count,np.int64);steps=np.zeros(count,np.int64)
    converged=np.zeros(count,np.bool_)
    points=history_end//history_every+1 if history_every>0 else 0
    histories=np.full((count,points),np.nan)
    qs=np.empty((count,n,S,m),np.float64) if save_q else np.empty((1,1,1,1),np.float64)
    joint_count=card**n
    modulus=S//joint_count if memory else 1
    decay=np.exp(-beta)
    for rep in prange(count):
        q=q_initial.copy()
        policy=np.empty((n,S),np.int16)
        for i in range(n):
            for s in range(S):policy[i,s]=argmax_first(q[i,s])
        rng=np.uint64(seeds[rep])+np.uint64(0x9E3779B97F4A7C15)
        if rng==0:rng=np.uint64(1)
        rng,u=draw(rng);state=int(u*S)
        if entry_rho:
            # Choose a valid random lagged profile (only outsider may be absent).
            state=0
            for i in range(n):
                rng,u=draw(rng);x=int(u*m)
                if i==n-1:
                    rng,u=draw(rng)
                    if u<.5:x=m
                state=state*card+x
        unchanged=0;epsilon=1. if exploration else 0.
        recent=np.zeros(100);reward_sum=0.;index=0;recent_count=0
        actions=np.empty(n,np.int64)
        stop=max_steps
        for t in range(max_steps):
            active=True
            if entry_rho:
                active=(state%card)!=m
                rng,u=draw(rng)
                if u<entry_rho:active=not active
            joint=0
            for i in range(n):
                act=int(policy[i,state])
                rng,u=draw(rng)
                if u<epsilon:
                    rng,u=draw(rng);act=int(u*m)
                if i==n-1 and not active:act=m
                actions[i]=act;joint=joint*card+act
            z=0
            if rewards.shape[0]>1:
                rng,u=draw(rng);z=int(u*rewards.shape[0])
            nxt=(state%modulus)*joint_count+joint if memory else 0
            changed=False;reward=0.
            for i in range(n):
                reward+=rewards[z,joint,i]/n
                act=actions[i]
                if act==m:continue
                best=int(policy[i,state]);oldmax=q[i,state,best]
                target=rewards[z,joint,i]+delta*q[i,nxt,policy[i,nxt]]
                new=(1-alpha)*q[i,state,act]+alpha*target
                q[i,state,act]=new
                newbest=best
                if act==best:
                    if new<oldmax:newbest=argmax_first(q[i,state])
                elif new>oldmax or (new==oldmax and act<best):newbest=act
                if newbest!=best:changed=True;policy[i,state]=newbest
            if points>0:
                reward_sum+=reward-recent[index]
                recent[index]=reward;index=(index+1)%100
                recent_count=min(recent_count+1,100)
                if t%history_every==0 and t//history_every<points:
                    histories[rep,t//history_every]=reward_sum/recent_count
            unchanged=0 if changed else unchanged+1
            state=nxt
            if exploration:epsilon*=decay
            if unchanged>=stable:
                converged[rep]=True;stop=t+1;break
        steps[rep]=stop;final_states[rep]=state
        for i in range(n):
            policies[rep,i]=policy[i]
        if save_q:qs[rep]=q
        # Fill remaining history with frozen-policy play after convergence. Explicitly
        # marked in the report; no continued Q-updates are invented after stopping.
        if points>0 and converged[rep] and stop<history_end:
            for t in range(stop,history_end+1):
                active=True
                if entry_rho:
                    active=(state%card)!=m;rng,u=draw(rng)
                    if u<entry_rho:active=not active
                joint=0
                for i in range(n):
                    act=int(policy[i,state])
                    if i==n-1 and not active:act=m
                    joint=joint*card+act
                z=0
                if rewards.shape[0]>1:rng,u=draw(rng);z=int(u*rewards.shape[0])
                reward=0.
                for i in range(n):reward+=rewards[z,joint,i]/n
                reward_sum+=reward-recent[index];recent[index]=reward;index=(index+1)%100
                if t%history_every==0 and t//history_every<points:histories[rep,t//history_every]=reward_sum/100
                state=(state%modulus)*joint_count+joint if memory else 0
    return policies,final_states,steps,converged,histories,qs

@njit(cache=True,parallel=True)
def rematch_batch(qs,rewards,alpha,delta,m,stable,max_steps,seeds,history_end,every):
    N=qs.shape[0];n=2;S=m*m;points=history_end//every+1
    policies=np.empty((N,n,S),np.int16);hist=np.full((N,points),np.nan)
    endstates=np.zeros(N,np.int64);steps=np.zeros(N,np.int64);conv=np.zeros(N,np.bool_)
    for r in prange(N):
        # Cyclic derangement: agent 0 from r, agent 1 from a distinct training session.
        q=np.empty((2,S,m));q[0]=qs[r,0];q[1]=qs[(r+1)%N,1]
        policy=np.empty((2,S),np.int16)
        for i in range(2):
            for s in range(S):policy[i,s]=argmax_first(q[i,s])
        rng=np.uint64(seeds[r])+np.uint64(0x9E3779B97F4A7C15);rng,u=draw(rng);s=int(u*S)
        unchanged=0;rolling=np.zeros(100);ri=0;total=0.
        for t in range(max_steps):
            a=int(policy[0,s]);b=int(policy[1,s]);nxt=a*m+b;changed=False
            reward=(rewards[0,nxt,0]+rewards[0,nxt,1])/2
            for i,act in enumerate((a,b)):
                best=int(policy[i,s]);old=q[i,s,best]
                new=(1-alpha)*q[i,s,act]+alpha*(rewards[0,nxt,i]+delta*q[i,nxt,policy[i,nxt]])
                q[i,s,act]=new;newbest=best
                if act==best:
                    if new<old:newbest=argmax_first(q[i,s])
                elif new>old or (new==old and act<best):newbest=act
                if newbest!=best:changed=True;policy[i,s]=newbest
            total+=reward-rolling[ri];rolling[ri]=reward;ri=(ri+1)%100
            if t%every==0 and t//every<points:hist[r,t//every]=total/min(t+1,100)
            unchanged=0 if changed else unchanged+1;s=nxt
            if unchanged>=stable:conv[r]=True;steps[r]=t+1;break
        if not conv[r]:steps[r]=max_steps
        endstates[r]=s;policies[r]=policy
        if conv[r] and steps[r]<history_end:
            for t in range(steps[r],history_end+1):
                nxt=int(policy[0,s])*m+int(policy[1,s]);reward=rewards[0,nxt].mean()
                total+=reward-rolling[ri];rolling[ri]=reward;ri=(ri+1)%100;s=nxt
                if t%every==0 and t//every<points:hist[r,t//every]=total/100
    return policies,endstates,steps,conv,hist
