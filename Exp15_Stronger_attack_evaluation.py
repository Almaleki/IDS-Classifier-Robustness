#!/usr/bin/env python3
"""Supplementary Experiment 1 evaluation ONLY: no training or original-file edits.

Dependencies: pip install torch pandas numpy scikit-learn matplotlib joblib
PyCharm: open this file in your existing project and click Run.
No command-line parameters or new interpreter are required.
RUN_DIR below is taken from your successful five-seed training log.
The dataset path is read from that run's saved preprocessing metadata.
Requires the original datasets/ and balanced/ output folders.

Default: FULL original held-out test set, all five training seeds, all five epsilons.
For a separately labelled exploratory subset, add --max-samples 6000.
CPU evaluation may be lengthy. Completed model/epsilon files are checkpointed.

PGD: original PGD-10 endpoint, then two step sizes (epsilon/5, epsilon/10),
100 steps, clean-start + five random-start trajectories per step size. Reports
best-so-far at 10/50/100 steps and cumulative restart counts 0/1/5. Convergence
CSV includes every iteration's current and best loss and cumulative accuracy.
Independent attack: CUSTOM score-based random-coordinate search, 500 proposals,
starting independently from the clean example. It uses logits, not gradients.
This is NOT AutoAttack, Square Attack, or a certified worst-case evaluation.
All perturbations are unrestricted numerical L-infinity feature-space changes.
Metrics are fractions; SD across training seeds uses ddof=1. Fixed test population.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import platform
import shutil
import sys
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PyCharm configuration: original run path from your supplied output log.
RUN_DIR = Path('/Users/almaleki/Desktop/Research/Research_OCT2025/Codes/PhD_Research_1_CIC2018Dataset/.venv/bin/experiment1_multiple_runs/20260919_164823_076724')
DATASET_OVERRIDE = None  # None reuses dataset_path saved by the original training script.
OUTPUT_PARENT = RUN_DIR.parent / 'experiment1_stronger_attack_outputs'
EVALUATION_BATCH_SIZE = 512
MAX_SAMPLES = 0  # Full held-out set; no subsampling.
RANDOM_SEARCH_QUERIES = 500

SEEDS = [42, 43, 44, 45, 46]
MODELS = ['baseline', 'FGSM_AT', 'PGD_AT']
EPSILONS = [.01, .05, .10, .20, .30]

class IDSModel(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Linear(256,128),
                                 nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,c))
    def forward(self,x):
        return self.net(x)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def scores(model,x,y):
    with torch.no_grad():
        logits=model(x)
        return nn.functional.cross_entropy(logits,y,reduction='none'),logits.argmax(1)

def choose(old_loss,old_pred,new_loss,new_pred,y):
    # Any successful misclassification takes priority over correct candidates.
    # Among equal correctness states, retain the larger cross-entropy loss.
    old_bad=old_pred.ne(y); new_bad=new_pred.ne(y)
    take=(new_bad & ~old_bad) | ((new_bad==old_bad) & (new_loss>old_loss))
    return torch.where(take,new_loss,old_loss),torch.where(take,new_pred,old_pred)

def project(z,x,eps):
    return torch.maximum(torch.minimum(z,x+eps),x-eps).detach()

def batch_attacks(model,x,y,eps,seed,queries):
    torch.manual_seed(seed)
    if x.is_cuda: torch.cuda.manual_seed_all(seed)
    clean_loss,clean_pred=scores(model,x,y)
    results={'clean':clean_pred.cpu().numpy()}
    union_loss,union_pred=clean_loss.clone(),clean_pred.clone()
    traces=[]
    for divisor in [5,10]:
        best_loss,best_pred=clean_loss.clone(),clean_pred.clone()
        # Separate retained candidate pools for each iteration budget.
        pools={k:(clean_loss.clone(),clean_pred.clone()) for k in [10,50,100]}
        for restart in range(6):
            z=x.clone() if restart==0 else project(x+torch.empty_like(x).uniform_(-eps,eps),x,eps)
            trajectory_loss,trajectory_pred=clean_loss.clone(),clean_pred.clone()
            max_loss=clean_loss.clone()
            for step in range(101):
                loss,pred=scores(model,z,y)
                trajectory_loss,trajectory_pred=choose(trajectory_loss,trajectory_pred,loss,pred,y)
                best_loss,best_pred=choose(best_loss,best_pred,loss,pred,y)
                max_loss=torch.maximum(max_loss,loss)
                traces.append(dict(alpha_divisor=divisor,restart=restart,iteration=step,
                    current_ce_sum=loss.sum().item(),trajectory_max_ce_sum=max_loss.sum().item(),
                    cumulative_correct=int(best_pred.eq(y).sum()),support=len(y)))
                if step in pools:
                    pl,pp=pools[step]
                    pools[step]=choose(pl,pp,trajectory_loss,trajectory_pred,y)
                    if restart in [0,1,5]:
                        key=f'PGD_best_steps{step}_random_restarts{restart}_alpha_eps_div{divisor}'
                        results[key]=pools[step][1].cpu().numpy()
                if divisor==5 and restart==0 and step==10:
                    results['original_PGD10_endpoint']=pred.cpu().numpy()
                if step==100: break
                z=z.detach().requires_grad_(True)
                grad,=torch.autograd.grad(nn.functional.cross_entropy(model(z),y),z)
                z=project(z.detach()+(eps/divisor)*grad.sign(),x,eps)
            union_loss,union_pred=choose(union_loss,union_pred,best_loss,best_pred,y)
    results['PGD_union']=union_pred.cpu().numpy()
    # Independent gradient-free score-based attack: maximize incorrect-vs-true margin.
    torch.manual_seed(seed+100000)
    if x.is_cuda: torch.cuda.manual_seed_all(seed+100000)
    def margin(z):
        with torch.no_grad():
            logits=model(z); true=logits.gather(1,y[:,None]).squeeze(1)
            other=logits.clone(); other.scatter_(1,y[:,None],float('-inf'))
            return other.max(1).values-true
    z=x.clone(); incumbent=margin(z)
    rloss,rpred=clean_loss.clone(),clean_pred.clone()
    for q in range(1,queries+1):
        # Decrease the changed coordinate fraction over the search.
        fraction=max(1/x.shape[1],.5*(1-q/(queries+1)))
        mask=torch.rand_like(x)<fraction
        coord=torch.randint(x.shape[1],(len(x),),device=x.device)
        mask[torch.arange(len(x),device=x.device),coord]=True
        signs=torch.where(torch.rand_like(x)<.5,-1.,1.)
        candidate=project(torch.where(mask,x+eps*signs,z),x,eps)
        value=margin(candidate)
        take=value>incumbent
        z=torch.where(take[:,None],candidate,z); incumbent=torch.maximum(incumbent,value)
        loss,pred=scores(model,candidate,y)
        rloss,rpred=choose(rloss,rpred,loss,pred,y)
        if q in [100,queries]: results[f'random_coordinate_search_queries{q}']=rpred.cpu().numpy()
    union_loss,union_pred=choose(union_loss,union_pred,rloss,rpred,y)
    results['all_attacks_union']=union_pred.cpu().numpy()
    return results,traces

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir',type=Path,default=RUN_DIR)
    p.add_argument('--dataset',type=Path,default=DATASET_OVERRIDE)
    p.add_argument('--batch-size',type=int,default=EVALUATION_BATCH_SIZE)
    p.add_argument('--max-samples',type=int,default=MAX_SAMPLES,help='0 = full test; positive = fixed stratified diagnostic subset')
    p.add_argument('--queries',type=int,default=RANDOM_SEARCH_QUERIES)
    p.add_argument('--output-parent',type=Path,default=OUTPUT_PARENT)
    a=p.parse_args()
    if a.batch_size<1 or a.queries<1 or a.max_samples<0: p.error('Invalid numerical argument')
    root = a.run_dir
    model_root = root / "exp1"
    prep_path=root/'datasets/balanced/preprocessing.joblib'
    split_path=root/'datasets/balanced/split_indices.npz'
    if not prep_path.is_file():
        raise FileNotFoundError(f'Original preprocessing file not found: {prep_path}. Restore the original output folder or update RUN_DIR at the top.')
    prep=joblib.load(prep_path)
    if a.dataset is None:
        a.dataset=Path(prep['dataset_path'])
    required=[a.dataset,prep_path,split_path]+[model_root/'balanced'/f'seed_{s}'/m/'model.pt' for s in SEEDS for m in MODELS]
    missing=[str(f) for f in required if not f.is_file()]
    if missing: raise FileNotFoundError('Missing original artifacts (no retraining attempted):\n'+'\n'.join(missing))
    if sha256(a.dataset)!=prep['dataset_sha256']:
        raise ValueError('Dataset hash differs from training input. Use the exact original CSV.')
    splits=np.load(split_path); idx=splits['test_indices']
    df=pd.read_csv(a.dataset,low_memory=False)
    if len(df)!=len(splits['source_row_indices']): raise ValueError('Source row count mismatch')
    labels={v:i for i,v in enumerate(prep['classes'])}
    y=df.iloc[idx]['AttackCategory'].map(labels)
    if y.isna().any(): raise ValueError('Unrecognized labels')
    y=y.to_numpy(dtype=np.int64)
    if 0<a.max_samples<len(idx):
        chosen,_=train_test_split(np.arange(len(idx)),train_size=a.max_samples,stratify=y,random_state=42)
        chosen=np.sort(chosen); idx=idx[chosen]; y=y[chosen]
    features=df.iloc[idx][prep['features']].replace([np.inf,-np.inf],np.nan).fillna(0)
    x=prep['scaler'].transform(features.to_numpy(dtype=float)).astype(np.float32)
    del df,features
    if not np.isfinite(x).all(): raise ValueError('Nonfinite standardized features')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out=a.output_parent/datetime.now().strftime('%Y%m%d_%H%M%S_%f'); out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    np.savez_compressed(out/'evaluation_indices.npz',source_row_indices=idx,y_true=y)
    config={**{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            'device':str(device),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
            'training_seeds':SEEDS,'epsilons':EPSILONS,'test_support':len(y),
            'scope':'full held-out set' if len(idx)==len(splits['test_indices']) else 'fixed stratified diagnostic subset',
            'dataset_sha256':prep['dataset_sha256'],'preprocessing_sha256':sha256(prep_path),
            'pgd_steps':[10,50,100],'random_restarts':[0,1,5],'alpha_divisors':[5,10],
            'attack_seed_rule':'1000*training_seed + batch_start; reused across models/epsilons',
            'classes':prep['classes'],'features':prep['features'],
            'independent_attack':'custom score-based random-coordinate search; not a standardized attack suite'}
    (out/'configuration.json').write_text(json.dumps(config,indent=2))
    print(f'Device={device}; test support={len(y)}; outputs={out.resolve()}',flush=True)
    rows=[]
    for s in SEEDS:
        for name in MODELS:
            checkpoint=model_root/'balanced'/f'seed_{s}'/name/'model.pt'
            model=IDSModel(len(prep['features']),len(prep['classes'])).to(device)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
            for param in model.parameters(): param.requires_grad_(False)
            target=out/f'seed_{s}'/name; target.mkdir(parents=True)
            (target/'checkpoint_sha256.txt').write_text(sha256(checkpoint))
            for eps in EPSILONS:
                predictions={}; trace_parts=[]
                for start in range(0,len(y),a.batch_size):
                    xb=torch.from_numpy(x[start:start+a.batch_size]).to(device)
                    yb=torch.from_numpy(y[start:start+a.batch_size]).to(device)
                    result,trace=batch_attacks(model,xb,yb,eps,1000*s+start,a.queries)
                    for k,v in result.items(): predictions.setdefault(k,[]).append(v)
                    trace_parts.append(pd.DataFrame(trace))
                    print(f'seed={s} model={name} epsilon={eps} rows={min(start+a.batch_size,len(y))}/{len(y)}',flush=True)
                predictions={k:np.concatenate(v) for k,v in predictions.items()}
                clean_ok=predictions['clean']==y
                pgd_ok=predictions['PGD_union']==y
                union_ok=predictions['all_attacks_union']==y
                if np.any(pgd_ok & ~clean_ok) or np.any(union_ok & ~pgd_ok):
                    raise RuntimeError('Candidate-union consistency check failed')
                reference=model_root/'balanced'/f'seed_{s}'/name/'clean_predictions.npz'
                if reference.exists():
                    with np.load(reference) as saved:
                        positions={int(v):i for i,v in enumerate(splits['test_indices'])}
                        selected=np.array([positions[int(v)] for v in idx])
                        if not np.array_equal(saved['y_true'][selected],y):
                            raise ValueError('Saved evaluation label alignment mismatch')
                        if not np.array_equal(saved['y_pred'][selected],predictions['clean']):
                            raise ValueError('Clean predictions differ from original run; check environment/checkpoints before comparing results')
                np.savez_compressed(target/f'eps_{eps:.2f}_predictions.npz',y_true=y,**predictions)
                traces=pd.concat(trace_parts).groupby(['alpha_divisor','restart','iteration'],as_index=False).sum()
                traces['mean_current_ce']=traces.current_ce_sum/traces.support
                traces['mean_trajectory_max_ce']=traces.trajectory_max_ce_sum/traces.support
                traces['cumulative_accuracy']=traces.cumulative_correct/traces.support
                traces.to_csv(target/f'eps_{eps:.2f}_convergence.csv',index=False)
                fig,axes=plt.subplots(1,2,figsize=(11,4))
                for (d,r),g in traces.groupby(['alpha_divisor','restart']):
                    axes[0].plot(g.iteration,g.mean_trajectory_max_ce,label=f'eps/{d}, restart {r}')
                    axes[1].plot(g.iteration,g.cumulative_accuracy)
                axes[0].set(xlabel='Iteration',ylabel='Mean best-so-far trajectory CE')
                axes[1].set(xlabel='Iteration',ylabel='Cumulative candidate accuracy',ylim=(0,1))
                axes[0].legend(fontsize=6); fig.tight_layout()
                fig.savefig(target/f'eps_{eps:.2f}_convergence.png',dpi=160); plt.close(fig)
                for attack,pred in predictions.items():
                    rows.append(dict(seed=s,model=name,epsilon=eps,attack=attack,test_support=len(y),
                        accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro',zero_division=0)))
                pd.DataFrame(rows).to_csv(out/'all_run_metrics.csv',index=False)
                summary=pd.DataFrame(rows).groupby(['model','epsilon','attack'])[['accuracy','macro_f1']].agg(['count','mean','std'])
                summary.columns=['_'.join(c) for c in summary.columns]
                summary.reset_index().to_csv(out/'mean_sd_metrics.csv',index=False)
            del model
    print(f'COMPLETE: {out.resolve()}',flush=True)

if __name__=='__main__': main()
