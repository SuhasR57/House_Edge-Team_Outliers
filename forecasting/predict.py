"""Offline deterministic hierarchical inference; no test labels read."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
import numpy as np
from dynamics import read_split,replay,write_csv

def mix_episode(ep,parts):
    first=max(min(p) for p in parts)
    end=len(ep['data']); m=ep['data'].shape[1]
    evidence=np.zeros((m,len(parts)))
    result=None
    for t in range(first,end+1):
        logits=.5*evidence
        weights=np.exp(logits-logits.max(axis=1,keepdims=True))
        weights/=weights.sum(axis=1,keepdims=True)
        stacked=np.stack([p[t] for p in parts],axis=1)
        result=(weights[:,:,None]*stacked).sum(axis=1)
        if t<end:
            evidence*=2**(-1/24)
            for i,y in enumerate(ep['data'][t]):
                if y>=0:
                    evidence[i]+=np.log(np.maximum(stacked[i,:,y],1e-12))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    settings=json.loads((root/'config.json').read_text())
    models=[]
    for i,config in enumerate(settings['components']):
        with np.load(root/f'model_{i}.npz',allow_pickle=False) as saved:
            models.append((saved['type_prior'],saved['structure_prior'],config))
    episodes,_=read_split(Path(args.data)/'test',load_labels=False)
    output={}
    for e,ep in sorted(episodes.items()):
        end=len(ep['data'])
        anchors=range(max(0,end-112),end+1) if len(models)>1 else ()
        parts=[replay(ep,tp,sp,cfg,anchors)[0] for tp,sp,cfg in models]
        values=mix_episode(ep,parts) if len(parts)>1 else parts[0][end]
        for q,p in zip(ep['queries'],values):
            output[q['query_id']]=tuple(float(x) for x in p)
    write_csv(args.out,output)

if __name__=='__main__':
    main()
