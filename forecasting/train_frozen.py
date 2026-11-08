"""Refit frozen settings using only supplied training histories and labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from dynamics import read_split,fit_hierarchy

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--config',default=str(Path(__file__).resolve().parent/'config.json'))
    args=parser.parse_args()
    settings=json.loads(Path(args.config).read_text())
    episodes,labels=read_split(Path(args.data)/'train')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    for i,config in enumerate(settings['components']):
        tp,sp=fit_hierarchy(episodes,labels,config)
        np.savez(out/f'model_{i}.npz',type_prior=tp,structure_prior=sp)
    (out/'config.json').write_text(json.dumps(settings,indent=2))
    print('Saved frozen-settings fit:',out)

if __name__=='__main__':
    main()
