"""Hierarchical, machine-specific Bayesian dynamics. NumPy only."""
import csv
import json
from pathlib import Path
import numpy as np

PAYOUTS=(0,1,2,5,20)
TYPES=('steady','balanced','jackpot')

def read_split(path, load_labels=True):
    path=Path(path)
    queries=list(csv.DictReader((path/'queries.csv').open(encoding='utf-8')))
    episodes={}
    for q in queries:
        q['cutoff']=int(q['cutoff'])
        episodes.setdefault(q['episode_id'],{'queries':[],'history':[]})['queries'].append(q)
    for row in csv.DictReader((path/'history.csv').open(encoding='utf-8')):
        episodes[row['episode_id']]['history'].append((row['machine_id'],int(row['tick']),PAYOUTS.index(int(row['payout']))))
    for ep in episodes.values():
        ep['queries'].sort(key=lambda q:q['machine_id'])
        cutoffs={q['cutoff'] for q in ep['queries']}
        if len(cutoffs)!=1:
            raise ValueError('Expected synchronous episode cutoffs')
        cutoff=cutoffs.pop()
        ids={q['machine_id']:i for i,q in enumerate(ep['queries'])}
        data=np.full((cutoff,len(ids)),-1,dtype=np.int16)
        for machine,t,y in ep.pop('history'):
            if not 0<=t<cutoff or machine not in ids or data[t,ids[machine]]!=-1:
                raise ValueError('Duplicate, future, or invalid history')
            data[t,ids[machine]]=y
        ep['data']=data
    labels={}
    if load_labels and (path/'labels.csv').exists():
        labels={r['query_id']:PAYOUTS.index(int(r['outcome'])) for r in csv.DictReader((path/'labels.csv').open())}
    return episodes,labels

def candidates():
    specs=[('constant',0,0)]
    for mode in ('win','large','coarse'):
        specs += [(mode,a,0) for a in range(1,9)]
        specs += [(mode,a,b) for a in range(1,9) for b in range(a+1,9)]
    for offset in range(1,8):
        specs += [('peer',offset,lag) for lag in (1,2,4,8)]
        specs += [('cross',offset,lag) for lag in (1,2)]
    return specs

SPECS=candidates()
C=len(SPECS)

def contexts(data,t):
    m=data.shape[1]
    lag=np.full((9,m),-1,dtype=np.int16)
    for l in range(1,9):
        if t>=l:
            lag[l]=data[t-l]
    result=np.full((m,C),-1,dtype=np.int16)
    result[:,0]=0
    for j,(mode,a,b) in enumerate(SPECS[1:],1):
        if mode in ('peer','cross'):
            other=(np.arange(m)+a)%m
            raw=lag[b,other]
            val=(raw>0).astype(np.int16)
            valid=raw>=0
            if mode=='cross':
                valid &= lag[1]>=0
                val=2*(lag[1]>0)+val
        else:
            raw=lag[a]
            valid=raw>=0
            if mode=='win':
                val=(raw>0).astype(np.int16)
                right=(lag[b]>0).astype(np.int16)
                radix=2
            elif mode=='large':
                val=(raw>=2).astype(np.int16)
                right=(lag[b]>=2).astype(np.int16)
                radix=2
            else:
                val=np.minimum(raw,2)
                right=np.minimum(lag[b],2)
                radix=3
            if b:
                valid &= lag[b]>=0
                val=radix*val+right
        result[:,j]=np.where(valid,val,-1)
    return result

def softmax(logits):
    values=np.exp(logits-logits.max(axis=1,keepdims=True))
    return values/values.sum(axis=1,keepdims=True)

def context_tensor(ep):
    if '_contexts' in ep:
        return ep['_contexts']
    data=ep['data']
    n,m=data.shape
    lag=np.full((9,n+1,m),-1,dtype=np.int16)
    for l in range(1,9):
        if n>=l:
            lag[l,l:]=data[:n+1-l]
    result=np.full((n+1,m,C),-1,dtype=np.int16)
    result[:,:,0]=0
    for j,(mode,a,b) in enumerate(SPECS[1:],1):
        if mode in ('peer','cross'):
            raw=lag[b][:,(np.arange(m)+a)%m]
            valid=raw>=0
            val=(raw>0).astype(np.int16)
            if mode=='cross':
                valid &= lag[1]>=0
                val=2*(lag[1]>0)+val
        else:
            raw=lag[a]
            valid=raw>=0
            if mode=='win':
                val=(raw>0).astype(np.int16); right=(lag[b]>0); radix=2
            elif mode=='large':
                val=(raw>=2).astype(np.int16); right=(lag[b]>=2); radix=2
            else:
                val=np.minimum(raw,2); right=np.minimum(lag[b],2); radix=3
            if b:
                valid &= lag[b]>=0
                val=radix*val+right
        result[:,:,j]=np.where(valid,val,-1)
    ep['_contexts']=result
    return result

def initial_prior():
    # Equal family mass prevents large feature families winning by sheer count.
    names=[s[0] for s in SPECS]
    family={'constant':.20,'win':.30,'large':.15,'coarse':.20,'peer':.10,'cross':.05}
    p=np.array([family[name]/names.count(name) for name in names])
    return np.tile(p,(3,2,1))

def fit_type_prior(episodes,labels):
    counts=np.ones((3,5))*.5
    for ep in episodes.values():
        for i,q in enumerate(ep['queries']):
            typ=TYPES.index(q['machine_type'])
            observed=ep['data'][:,i]
            counts[typ]+=np.bincount(observed[observed>=0],minlength=5)
            if q['query_id'] in labels:
                counts[typ,labels[q['query_id']]]+=1
    return counts/counts.sum(axis=1,keepdims=True)

def replay(ep,type_prior,structure_prior,config,anchors=()):
    """All anchor predictions precede same-tick updates across every machine."""
    data=ep['data']
    all_contexts=context_tensor(ep)
    m=data.shape[1]
    types=np.array([TYPES.index(q['machine_type']) for q in ep['queries']])
    prior=type_prior[types]
    win_counts=np.zeros((m,C,9,2))
    amount_counts=np.zeros((m,C,9,4))
    recent=np.zeros((m,5))
    score=np.zeros((m,2,C))
    ii=np.arange(m)[:,None]
    jj=np.arange(C)[None,:]
    decay=2**(-1/config['half_life'])
    evidence_decay=2**(-1/config['evidence_half_life'])
    alpha=config['strength']
    predictions={}
    anchor_set=set(anchors)|{len(data)}
    for t in range(len(data)+1):
        ctx=all_contexts[t]
        valid=ctx>=0
        safe=np.maximum(ctx,0)
        base=(recent+20*prior)/(recent.sum(axis=1,keepdims=True)+20)
        bpwin=np.stack((base[:,0],1-base[:,0]),axis=1)
        bpamount=base[:,1:]/base[:,1:].sum(axis=1,keepdims=True)
        wc=win_counts[ii,jj,safe]*valid[:,:,None]
        ac=amount_counts[ii,jj,safe]*valid[:,:,None]
        wp=(wc+alpha*bpwin[:,None,:])/(wc.sum(axis=2,keepdims=True)+alpha)
        ap=(ac+alpha*bpamount[:,None,:])/(ac.sum(axis=2,keepdims=True)+alpha)
        weights=softmax((np.log(structure_prior[types])+config['temperature']*score).reshape(m*2,C)).reshape(m,2,C)
        if t in anchor_set:
            win=(weights[:,0,:,None]*wp).sum(axis=1)
            amount=(weights[:,1,:,None]*ap).sum(axis=1)
            predictions[t]=np.column_stack((win[:,0],win[:,1,None]*amount))
        if t==len(data):
            break
        observed=data[t]
        present=observed>=0
        wins=observed>0
        # Log predictive evidence is scored BEFORE updating any same-tick count.
        score*=evidence_decay
        for i in np.flatnonzero(present):
            y=int(wins[i])
            score[i,0]+=np.log(np.maximum(wp[i,:,y],1e-12))-np.log(bpwin[i,y])
            if wins[i]:
                k=observed[i]-1
                score[i,1]+=np.log(np.maximum(ap[i,:,k],1e-12))-np.log(bpamount[i,k])
        win_counts*=decay
        amount_counts*=decay
        recent*=2**(-1/48)
        for i in np.flatnonzero(present):
            js=np.flatnonzero(valid[i])
            win_counts[i,js,ctx[i,js],int(wins[i])]+=1
            if wins[i]:
                amount_counts[i,js,ctx[i,js],observed[i]-1]+=1
            recent[i,observed[i]]+=1
    return predictions,score

def fit_hierarchy(episodes,labels,config):
    type_prior=fit_type_prior(episodes,labels)
    initial=initial_prior()
    totals=np.zeros_like(initial)
    count=np.zeros(3)
    # One empirical-Bayes update, strongly shrunk to the fixed family prior.
    for episode_id,ep in sorted(episodes.items()):
        _,scores=replay(ep,type_prior,initial,config)
        for i,q in enumerate(ep['queries']):
            typ=TYPES.index(q['machine_type'])
            posterior=softmax(np.log(initial[typ])+config['temperature']*scores[i])
            totals[typ]+=posterior
            count[typ]+=1
    structure=(totals+64*initial)/(count[:,None,None]+64)
    return type_prior,structure

def predict(episodes,type_prior,structure_prior,config):
    result={}
    for episode_id,ep in sorted(episodes.items()):
        predictions,_=replay(ep,type_prior,structure_prior,config)
        p=predictions[len(ep['data'])]
        for q,values in zip(ep['queries'],p):
            result[q['query_id']]=tuple(float(x) for x in values)
    return result

def write_csv(path,predictions):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['query_id','p_0','p_1','p_2','p_5','p_20'])
        for key,p in sorted(predictions.items()):
            if len(p)!=5 or not all(np.isfinite(p)) or min(p)<0 or abs(sum(p)-1)>1e-8:
                raise ValueError('Invalid prediction')
            writer.writerow([key]+[format(x,'.17g') for x in p])
