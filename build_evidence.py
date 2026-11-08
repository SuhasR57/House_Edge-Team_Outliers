"""Extract an allowlisted, secret-free evidence packet from local experiment artifacts."""
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

def build(outputs):
    outputs=Path(outputs)
    facts=[]
    sources=[]
    def load(relative):
        path=outputs/relative
        raw=path.read_bytes()
        sources.append({'path':relative,'sha256':hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    def add(key,label,value,source):
        facts.append({'id':key,'label':label,'value':value,'source':source})
    temporal=load('house-edge-temporal/experiments.json')
    hierarchy=load('house-edge-hierarchical-refined/training_report.json')
    verification=load('house-edge-hierarchical-refined/verification.json')
    config=load('house-edge-hierarchical-refined/config.json')
    bootstrap=load('house-edge-hierarchical-refined/uncertainty.json')
    for name,score in [('temporal',temporal['selected_full']),('hierarchical',hierarchy['validation']['full'])]:
        source='house-edge-temporal/experiments.json' if name=='temporal' else 'house-edge-hierarchical-refined/training_report.json'
        add(f'{name}.points',f'{name.title()} full validation points',score['technical_points'],source)
        add(f'{name}.ratio',f'{name.title()} mean normalized error',score['mean_loss_ratio'],source)
        for condition,row in score['by_condition'].items():
            add(f'{name}.{condition}',f'{name.title()} / {condition} normalized error',row['loss_ratio'],source)
    source='house-edge-hierarchical-refined/training_report.json'
    for split in ('old_tuning','old_audit'):
        add(f'hierarchical.{split}.points',f'Hierarchical {split} validation points',hierarchy['validation'][split]['technical_points'],source)
    add('evaluation.caveat','Evaluation limitations',hierarchy['caveat'],source)
    add('evaluation.selection_rule','Model selection procedure',hierarchy['selection_rule'],source)
    add('evaluation.train_audit_brier','Reserved-training observed-outcome Brier',hierarchy['train_audit']['mean_brier'],source)
    add('model.configuration','Frozen hierarchical configuration',config,'house-edge-hierarchical-refined/config.json')
    add('submission.verification','Offline reproduction and file checks',verification,'house-edge-hierarchical-refined/verification.json')
    add('evaluation.bootstrap','Descriptive episode bootstrap',bootstrap,'house-edge-hierarchical-refined/uncertainty.json')
    add('score.definition','House Edge score definition',{'main_max':60,'formula':'60 / (1 + mean normalized condition error)','score_is_accuracy':False},'Organizer 60/40 kit RULES.md')
    add('test.unknown','Hidden-test performance','No hidden-test score is available; validation is not a ranking guarantee.','Project evidence boundary')
    add('sponsor.separation','Gemini role','Gemini reviews evidence and proposes hypotheses. It does not generate, alter, or upload competition forecasts.','Application design')
    return {'version':1,'project':'EdgeShift Lab','created_at':datetime.now(timezone.utc).isoformat(),
            'round':'main','facts':facts,'sources':sources,
            'transmission':'Only this aggregate evidence packet and the review question are sent to Gemini. No raw histories, test forecasts, model checkpoints, credentials, or local absolute paths are included.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--outputs',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    destination=Path(args.out)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(build(args.outputs),indent=2),encoding='utf-8')
    print('Saved aggregate evidence:',destination)
