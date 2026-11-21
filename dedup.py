"""Sensitivity to near-duplicate analyses: one member of each same-label pair removed, both members of pairs with
conflicting labels removed, and reference codes linked by a duplicate pair merged into one group before splitting."""
import sys,json; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
d,X,y,classes,names,groups=load(sys.argv[1]); pairs=np.load('res/dup_pairs.npy'); n=len(y)
drop=np.zeros(n,bool)
for i,j in pairs:
    if y[i]!=y[j]: drop[i]=drop[j]=True
for i,j in pairs:
    if y[i]==y[j] and not drop[i] and not drop[j]: drop[j]=True
G=groups.max()+1; A=coo_matrix((np.ones(len(pairs)),(groups[pairs[:,0]],groups[pairs[:,1]])),shape=(G,G)); _,lab=connected_components(A,directed=False); g2=lab[groups]
keep=np.where(~drop)[0]; out=dict(n_drop=int(drop.sum()),n_keep=int(len(keep)),n_groups=int(len(np.unique(g2[keep]))),drop_by_class={classes[c]:int(v) for c,v in enumerate(np.bincount(y[drop],minlength=8))})
np.save('res/dedup_keep.npy',keep); np.save('res/dedup_groups.npy',g2)
sp=[(keep[a],keep[b]) for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(keep,y[keep],g2[keep])]
for mn in ['LightGBM','SVM (RBF)','CatBoost']:
    P=np.full((n,8),np.nan)
    for f,(tr,te) in enumerate(sp):
        Z=build_features(X,y,names,tr,f'cache/dedup_{f}.npz')[1]; P[te]=models()[mn]().fit(Z[tr],y[tr]).predict_proba(Z[te])
    np.save(f'res/oof/dedup_{mn.replace(" ","_")}.npy',P); p=P[keep].argmax(1); rc=M.recall_score(y[keep],p,average=None)
    out[mn]=dict(acc=float(M.accuracy_score(y[keep],p)),bacc=float(rc.mean()),recall={classes[i]:float(rc[i]) for i in range(8)}); print(mn,out[mn],flush=True)
json.dump(out,open('res/dedup.json','w'),indent=1)
