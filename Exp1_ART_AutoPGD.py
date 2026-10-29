#!/usr/bin/env python3
"""Standard-library Auto-PGD evaluation; no training or changes to original files.
Install in the SAME PyCharm interpreter:
    python -m pip install adversarial-robustness-toolbox==1.20.1
This is ART Auto-PGD (CE and DLR), NOT the complete AutoAttack suite.
Uses all supplementary test rows, all five training seeds, all three models.
Resume by rerunning: completed, fingerprint-checked chunks are reused.
"""
from pathlib import Path
import hashlib, inspect, json, random, sys, time, zipfile
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
try:
    import art
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import AutoProjectedGradientDescent
except ImportError:
    raise SystemExit('Install in this interpreter: '+sys.executable+' -m pip install adversarial-robustness-toolbox==1.20.1')

BASE = Path('/Users/almaleki/Desktop/Research/Research_OCT2025/Codes/PhD_Research_1_CIC2018Dataset/.venv/bin')
RUN = BASE/'experiment1_grouped_overlap_control/20261028_001334_956812'
CSV = BASE/'multiclass_balanced.csv'
OUT = RUN/'standard_ART_AutoPGD_eps030_v1'
SEEDS = [42,43,44,45,46]
MODELS = ['baseline','FGSM_AT','PGD_AT']
LOSSES = ['cross_entropy','difference_logits_ratio']
EPS = 0.30
STEPS = 100
STARTS = 5
BATCH = 256
CLASSES = ['Benign','Botnet','BruteForce','DDoS','DoS','Infiltration']
DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False


class IDSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(79,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,6))
    def forward(self,x): return self.net(x)


def export(y,p,clean,folder,tag):
    report=classification_report(y,p,labels=list(range(6)),target_names=CLASSES,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(folder/(tag+'_class_metrics.csv'))
    pd.DataFrame(confusion_matrix(y,p,labels=list(range(6))),index=CLASSES,columns=CLASSES).to_csv(folder/(tag+'_confusion_matrix.csv'))
    cc=clean==y; benign=y==0; malicious=~benign
    return dict(attack=tag,accuracy=float(np.mean(y==p)),macro_f1=report['macro avg']['f1-score'],
        clean_correct_count=int(cc.sum()),attack_induced_error_rate=float(np.mean(p[cc]!=y[cc])) if cc.any() else None,
        benign_false_positive_rate=float(np.mean(p[benign]!=0)),malicious_false_negative_rate=float(np.mean(p[malicious]==0)),
        wrong_attack_family_rate=float(np.mean((p[malicious]!=0)&(p[malicious]!=y[malicious]))),test_support=len(y))


def main():
    if art.__version__!='1.20.1':
        raise RuntimeError('Use pinned ART 1.20.1 for reproducibility; installed '+art.__version__)
    prep=RUN/'datasets/balanced/preprocessing.joblib'
    split=RUN/'datasets/balanced/split_indices.npz'
    checkpoints={(s,m):RUN/'balanced'/f'seed_{s}'/m/'model.pt' for s in SEEDS for m in MODELS}
    required=[CSV,prep,split]+list(checkpoints.values())
    missing=[str(p) for p in required if not p.is_file()]
    if missing: raise FileNotFoundError('Missing saved artifacts; no retraining attempted:\n'+'\n'.join(missing))
    saved=joblib.load(prep)
    if digest(CSV)!=saved['dataset_sha256']: raise RuntimeError('Dataset SHA256 mismatch.')
    if saved['classes']!=CLASSES or len(saved['features'])!=79: raise RuntimeError('Unexpected feature/class mapping.')
    idx=np.load(split)['test_indices']
    df=pd.read_csv(CSV,low_memory=False)
    features=df[saved['features']].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(dtype=np.float64)
    x=saved['scaler'].transform(features[idx]).astype(np.float32)
    y=df['AttackCategory'].map({c:i for i,c in enumerate(CLASSES)}).to_numpy()[idx].astype(np.int64)
    del df,features
    if not np.isfinite(x).all(): raise RuntimeError('Nonfinite inputs.')
    cfg=dict(run=str(RUN),dataset_sha256=saved['dataset_sha256'],split_sha256=digest(split),preprocessing_sha256=digest(prep),
        checkpoint_sha256={f'{s}/{m}':digest(p) for (s,m),p in checkpoints.items()},
        epsilon=EPS,norm='Linf',max_iter=STEPS,random_initialisations=STARTS,initial_step=2*EPS,
        losses=LOSSES,batch_size=BATCH,seeds=SEEDS,classes=CLASSES,test_support=len(y),
        attack_seed_rule='100000 + training_seed*1000 + loss_index*1000000 + chunk_index; shared across models',
        clipping=None,scope='Unconstrained standardized numerical features; not executable traffic',
        implementation='ART AutoProjectedGradientDescent, not full AutoAttack',
        stopping='ART iterations and adaptive step schedule; successful samples omitted from later restarts; no black-box query cap',
        versions={p:version(p) for p in ['adversarial-robustness-toolbox','torch','numpy','pandas','scikit-learn','scipy','joblib']},device=DEVICE,
        attack_source_sha256=digest(inspect.getsourcefile(AutoProjectedGradientDescent)))
    OUT.mkdir(parents=True,exist_ok=True)
    cp=OUT/'configuration.json'
    if cp.exists() and json.loads(cp.read_text())!=cfg: raise RuntimeError('Configuration differs from saved output. Select a NEW OUT folder.')
    cp.write_text(json.dumps(cfg,indent=2))
    (OUT/'evaluation_script.py').write_bytes(Path(__file__).read_bytes())
    (OUT/'ART_AutoPGD_source.py').write_text(inspect.getsource(inspect.getmodule(AutoProjectedGradientDescent)))
    np.savez_compressed(OUT/'evaluation_indices.npz',source_row_indices=idx,y_true=y)
    allrows=[]
    for s in SEEDS:
        for m in MODELS:
            folder=OUT/f'seed_{s}'/m; folder.mkdir(parents=True,exist_ok=True)
            seed_all(s); model=IDSModel()
            model.load_state_dict(torch.load(checkpoints[s,m],map_location='cpu',weights_only=True),strict=True)
            model.eval()
            clf=PyTorchClassifier(model=model,loss=nn.CrossEntropyLoss(),input_shape=(79,),nb_classes=6,optimizer=None,
                                  clip_values=None,preprocessing=(0.,1.),device_type=DEVICE)
            clean=clf.predict(x,batch_size=BATCH).argmax(1)
            original=np.load(RUN/'balanced'/f'seed_{s}'/m/'clean_predictions.npz')
            if not np.array_equal(original['y_true'],y) or not np.array_equal(original['y_pred'],clean):
                raise RuntimeError('Clean predictions do not reproduce saved run: '+str(folder))
            predictions={'clean':clean}; union=clean.copy()
            for li,loss in enumerate(LOSSES):
                tag='APGD_CE' if li==0 else 'APGD_DLR'
                attack=AutoProjectedGradientDescent(estimator=clf,norm=np.inf,eps=EPS,eps_step=2*EPS,
                    max_iter=STEPS,nb_random_init=STARTS,batch_size=BATCH,targeted=False,loss_type=loss,verbose=False)
                pred=clean.copy(); elapsed=0.; max_norm=0.
                chunks=folder/tag; chunks.mkdir(exist_ok=True)
                for ci,start in enumerate(range(0,len(y),BATCH)):
                    end=min(start+BATCH,len(y)); dest=chunks/f'chunk_{ci:05d}.npz'
                    if not dest.exists():
                        seed_all(100000+s*1000+li*1000000+ci)
                        xx=x[start:end]; yy=y[start:end]; candidate=xx.copy()
                        correct=clean[start:end]==yy
                        began=time.perf_counter()
                        if correct.any():
                            candidate[correct]=attack.generate(x=xx[correct].copy(),y=yy[correct])
                        seconds=time.perf_counter()-began
                        delta=np.max(np.abs(candidate.astype(np.float64)-xx.astype(np.float64)),axis=1)
                        # Float32 rounding tolerance only; no post-hoc clipping or attack modification.
                        tol=8*np.finfo(np.float32).eps*np.maximum(1.,np.max(np.abs(xx),axis=1))
                        if not np.isfinite(candidate).all() or np.any(delta>EPS+tol): raise RuntimeError('Invalid adversarial bounds.')
                        pp=clf.predict(candidate,batch_size=BATCH).argmax(1)
                        # Clean errors are retained; attack success measured among clean-correct rows.
                        pp[~correct]=clean[start:end][~correct]
                        tmp=dest.with_suffix('.tmp')
                        with open(tmp,'wb') as f:
                            np.savez_compressed(f,y_pred=pp,max_linf=delta,seconds=seconds,source_row_indices=idx[start:end])
                        tmp.replace(dest)
                    with np.load(dest) as a:
                        if not np.array_equal(a['source_row_indices'],idx[start:end]): raise RuntimeError('Chunk row mismatch.')
                        pred[start:end]=a['y_pred']; elapsed+=float(a['seconds']); max_norm=max(max_norm,float(a['max_linf'].max()))
                    if ci%10==0 or end==len(y): print(f'{s} {m} {tag}: {end:,}/{len(y):,}',flush=True)
                predictions[tag]=pred
                success=(clean==y)&(pred!=y); union[success]=pred[success]
                row=export(y,pred,clean,folder,tag)
                row.update(seed=s,model=m,epsilon=EPS,attack_seconds=elapsed,max_observed_linf=max_norm)
                allrows.append(row)
            predictions['APGD_union']=union
            for tag in ['clean','APGD_union']:
                row=export(y,predictions[tag],clean,folder,tag)
                row.update(seed=s,model=m,epsilon=0. if tag=='clean' else EPS)
                allrows.append(row)
            # Union errors use DLR's label if both attacks succeed; family confusion is selection-dependent.
            np.savez_compressed(folder/'predictions.npz',y_true=y,source_row_indices=idx,**predictions)
            pd.DataFrame(allrows).to_csv(OUT/'all_run_metrics.csv',index=False)
            print(f'Completed {s} {m}: union accuracy={np.mean(union==y):.6f}',flush=True)
    raw=pd.DataFrame(allrows)
    metrics=['accuracy','macro_f1','attack_induced_error_rate','benign_false_positive_rate','malicious_false_negative_rate']
    summary=raw.groupby(['model','attack'])[metrics].agg(['count','mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(OUT/'mean_sd_metrics.csv')
    archive=OUT/'review_outputs.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file() and p!=archive and not p.name.startswith('chunk_') and p.suffix!='.tmp': z.write(p,p.relative_to(OUT))
    print('Completed. Upload:',archive,flush=True)


if __name__=='__main__': main()
