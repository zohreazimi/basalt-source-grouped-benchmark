"""Repeated grouped cross-validation with random assignment of reference codes to five folds."""
import sys; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
d,X,y,classes,names,groups=load(sys.argv[1]); ug=np.unique(groups); res=[]
for rep in range(int(sys.argv[2]) if len(sys.argv)>2 else 5):
    r2=np.random.default_rng(100+rep); fold_of=r2.integers(0,5,len(ug)); fo=fold_of[groups]; P={m:np.zeros(len(y),int) for m in ['LightGBM','SVM (RBF)']}
    for f in range(5):
        te=np.where(fo==f)[0]; tr=np.where(fo!=f)[0]; Z=build_features(X,y,names,tr,f'cache/rg{rep}_{f}.npz')[1]
        for m in P: P[m][te]=models()[m]().fit(Z[tr],y[tr]).predict(Z[te])
    for m in P:
        rc=M.recall_score(y,P[m],average=None); res.append(dict(rep=rep,model=m,acc=float(M.accuracy_score(y,P[m])),bacc=float(rc.mean()),**{classes[i]:float(rc[i]) for i in range(8)})); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in res[-1].items()},flush=True)
    json.dump(res,open('res/repeated_grouped.json','w'),indent=1)
