"""Training experiments that need model fitting: augmentation (several seeds and targets), coarse-label training,
LightGBM hyperparameter sensitivity. Resumable; results in res/exp.jsonl and res/oof/."""
import sys,time,json; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
from lightgbm import LGBMClassifier
d,X,y,classes,names,groups=load(sys.argv[1]); L=np.log(X); SIG=np.array([0.02]*8+[0.08]*16)
def synth(method,tr,rng,target):
    Ls=[];ys=[]
    if method=='noise_all': return L[tr]+rng.normal(0,1,(len(tr),24))*SIG, y[tr]      # one noisy copy of every training sample
    for c in range(8):
        idx=tr[y[tr]==c]; n=target-len(idx)
        if n<=0: continue
        a=rng.choice(idx,n)
        if method=='noise': new=L[a]+rng.normal(0,1,(n,24))*SIG
        elif method=='dup': new=L[a].copy()                                           # control: plain duplication
        elif method=='within':                                                         # control: partner from the SAME reference code
            b=np.array([rng.choice(idx[groups[idx]==groups[i]]) for i in a]); lam=rng.uniform(0,1,(n,1)); new=lam*L[a]+(1-lam)*L[b]
        else:
            if method=='smote':
                nn=NearestNeighbors(n_neighbors=6).fit(L[idx]); _,ind=nn.kneighbors(L[a]); b=idx[ind[np.arange(n),rng.integers(1,6,n)]]
            else:   # xsource: partner of the same class from a different reference code
                b=rng.choice(idx,n)
                for _ in range(20):
                    same=groups[a]==groups[b]
                    if not same.any(): break
                    b[same]=rng.choice(idx,same.sum())
            lam=rng.uniform(0,1,(n,1)); new=lam*L[a]+(1-lam)*L[b]
        Ls.append(new); ys.append(np.full(n,c))
    return np.vstack(Ls),np.concatenate(ys)
out='res/exp.jsonl'; done=set()
if os.path.exists(out):
    for l in open(out): done.add(json.loads(l)['tag'])
def log(tag,prob,yt,extra):
    np.save(f'res/oof/x_{tag.replace(" ","_")}.npy',prob); r=dict(tag=tag,**extra,**scores(yt,prob)); open(out,'a').write(json.dumps(r)+'\n'); print(tag,round(r['acc'],4),round(r['bacc'],4),flush=True)
COARSE3={'IAB':0,'CAB':0,'IOAB':0,'MORB':1,'BABB':1,'OIB':2,'OPB':2,'CFB':2}; COARSE4={'IAB':0,'CAB':0,'IOAB':0,'MORB':1,'BABB':1,'OIB':2,'OPB':2,'CFB':3}
LGB={'small':dict(n_estimators=300,num_leaves=7),'medium':dict(n_estimators=300,num_leaves=15),'large':dict(n_estimators=1000,num_leaves=63),'slow':dict(n_estimators=1500,num_leaves=31,learning_rate=0.02)}
for prot in ['grouped','random']:
    for f,(tr,te) in enumerate(make_splits(y,groups,prot)):
        z=np.load(f'cache/{prot}_{f}.npz',allow_pickle=True); feat=list(z['feat']); pairs=[(names.index(r.split('/')[0]),names.index(r.split('/')[1])) for r in feat[24:]]
        F=np.column_stack([X]+[L[:,i]-L[:,j] for i,j in pairs]); pt=PowerTransformer().fit(F[tr]); Z=pt.transform(F)
        def aug(method,seed,target):
            rng=np.random.default_rng(seed*1000+f); Ls,ys=synth(method,tr,rng,target); Zs=pt.transform(np.column_stack([np.exp(Ls)]+[Ls[:,i]-Ls[:,j] for i,j in pairs])); return np.vstack([Z[tr],Zs]),np.concatenate([y[tr],ys])
        jobs=[(m,s,1500,mn) for m in ['xsource','noise','smote'] for s in [1,2,3] for mn in ['LightGBM']]+[(m,s,1500,'SVM (RBF)') for m in ['xsource','noise'] for s in [1,2,3]]+[('xsource',1,1500,'CatBoost')]
        jobs+=[(m,s,1500,'LightGBM') for m in ['dup','within'] for s in [1,2,3]]
        if prot=='grouped': jobs+=[('xsource',s,t,'LightGBM') for t in [750,3000] for s in [1,2,3]]+[('noise_all',s,1500,'LightGBM') for s in [1,2,3]]+[('smote',s,1500,'SVM (RBF)') for s in [1,2,3]]+[('xsource',s,1500,'CatBoost') for s in [2,3]]
        for s in [1,2,3]:      # model-seed variability of the unaugmented LightGBM
            tag=f'{prot}_{f}_seed{s}_LightGBM'
            if tag in done or prot!='grouped': continue
            m=models()['LightGBM'](); m.set_params(random_state=s); log(tag,m.fit(Z[tr],y[tr]).predict_proba(Z[te]),y[te],dict(protocol=prot,fold=f,kind='seed',seed=s))
        for method,seed,target,mn in jobs:
            tag=f'{prot}_{f}_aug_{method}_s{seed}_t{target}_{mn}'
            if tag in done: continue
            Ztr,ytr=aug(method,seed,target); log(tag,models()[mn]().fit(Ztr,ytr).predict_proba(Z[te]),y[te],dict(protocol=prot,fold=f,kind='aug',method=method,seed=seed,target=target,model=mn))
        for nm,mp in [('coarse3',COARSE3),('coarse4',COARSE4)]:
            tag=f'{prot}_{f}_{nm}_LightGBM'
            if tag in done: continue
            yc=np.array([mp[classes[i]] for i in y]); k=len(set(mp.values())); p=models()['LightGBM']().fit(Z[tr],yc[tr]).predict_proba(Z[te]); np.save(f'res/oof/x_{tag}.npy',p)
            pp=p.argmax(1); r=dict(tag=tag,protocol=prot,fold=f,kind=nm,acc=float(M.accuracy_score(yc[te],pp)),bacc=float(M.balanced_accuracy_score(yc[te],pp))); open(out,'a').write(json.dumps(r)+'\n'); print(tag,round(r['acc'],4),round(r['bacc'],4),flush=True)
        for nm,kw in LGB.items():
            tag=f'{prot}_{f}_lgb_{nm}'
            if tag in done: continue
            base=dict(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,class_weight='balanced',random_state=SEED,n_jobs=2,verbose=-1); base.update(kw)
            log(tag,LGBMClassifier(**base).fit(Z[tr],y[tr]).predict_proba(Z[te]),y[te],dict(protocol=prot,fold=f,kind='lgb',config=nm))
print('ALL DONE')
