"""Tabular baselines under both protocols; all preprocessing fitted on the training fold only. Saves out-of-fold probabilities."""
import sys,time,json; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
if __name__=='__main__':
    d,X,y,classes,names,groups=load(sys.argv[1])
    os.makedirs('res/oof',exist_ok=True)
    out='res/baselines.jsonl'; done=set()
    if os.path.exists(out):
        for l in open(out): r=json.loads(l); done.add((r['protocol'],r['fold'],r['model'],r['feats']))
    for prot in ['random','grouped']:
        for f,(tr,te) in enumerate(make_splits(y,groups,prot)):
            Z24,Z44,feat,table=build_features(X,y,names,tr,f'cache/{prot}_{f}.npz')
            for fs,Z in [('44',Z44),('24',Z24)]:
                for name,mk in models().items():
                    if fs=='24' and name not in ('LightGBM','SVM (RBF)','KNN','Extra Trees'): continue
                    if (prot,f,name,fs) in done: continue
                    t=time.time(); m=mk().fit(Z[tr],y[tr]); prob=m.predict_proba(Z[te])
                    np.save(f'res/oof/{prot}_{f}_{name.replace(" ","_")}_{fs}.npy',prob)
                    r=dict(protocol=prot,fold=f,model=name,feats=fs,time=time.time()-t,**scores(y[te],prob))
                    open(out,'a').write(json.dumps(r)+'\n'); print(prot,f,name,fs,round(r['acc'],4),round(r['bacc'],4),round(r['time']),flush=True)
