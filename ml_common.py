"""Shared utilities: data loading, fold-internal feature construction, splits and metrics."""
import os, json, time, numpy as np, pandas as pd, warnings
from itertools import combinations
warnings.filterwarnings('ignore')
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics as M

PRETTY={'SIO2':'SiO2','TIO2':'TiO2','AL2O3':'Al2O3','FE2O3t':'Fe2O3t','CAO':'CaO','MGO':'MgO','K2O':'K2O','NA2O':'Na2O',
 'RB':'Rb','SR':'Sr','Y':'Y','ZR':'Zr','NB':'Nb','BA':'Ba','LA':'La','CE':'Ce','ND':'Nd','SM':'Sm','GD':'Gd','YB':'Yb','LU':'Lu','HF':'Hf','TA':'Ta','TH':'Th'}
SEED=42
N_RATIOS=20

def load(path):
    d=pd.read_excel(path)
    cols=list(d.columns[4:])
    names=[PRETTY[c.split('(')[0]] for c in cols]
    X=d[cols].values.astype(float)
    lab=d['Tectonic Setting'].values
    classes=sorted(set(lab)); y=np.array([classes.index(v) for v in lab])
    groups=pd.factorize(d['References'])[0]
    return d,X,y,classes,names,groups

def log_ratios(X,names):
    L=np.log(X)
    idx=list(combinations(range(X.shape[1]),2))
    R=np.stack([L[:,i]-L[:,j] for i,j in idx],1)
    rn=[f'{names[i]}/{names[j]}' for i,j in idx]
    return R,rn

def select_ratios(Rtr,ytr,rn,k=N_RATIOS,corr_max=0.98,seed=SEED):
    """Consensus ranking (ANOVA F, mutual information, RF importance) computed on TRAINING data only."""
    F,_=f_classif(Rtr,ytr)
    mi=mutual_info_classif(Rtr,ytr,random_state=seed,n_neighbors=3)
    rf=RandomForestClassifier(n_estimators=300,n_jobs=2,random_state=seed,class_weight='balanced').fit(Rtr,ytr)
    imp=rf.feature_importances_
    nz=lambda v:(v-v.min())/(v.max()-v.min())
    S=np.column_stack([nz(F),nz(mi),nz(imp)]); score=S.mean(1)
    order=np.argsort(-score); C=np.corrcoef(Rtr.T); chosen=[]
    for j in order:
        if all(abs(C[j,c])<=corr_max for c in chosen): chosen.append(j)
        if len(chosen)==k: break
    table=[dict(ratio=rn[j],anova=float(S[j,0]),mi=float(S[j,1]),rf=float(S[j,2]),score=float(score[j])) for j in chosen]
    return chosen,table

def make_splits(y,groups,protocol,n_splits=5,seed=SEED):
    if protocol=='random':
        return list(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(y,y))
    return list(StratifiedGroupKFold(n_splits,shuffle=True,random_state=seed).split(y,y,groups))

def inner_val(tr,y,groups,protocol,seed=SEED):
    """Split the training indices into fit / validation parts (same logic as the outer protocol)."""
    if protocol=='random':
        a,b=next(StratifiedKFold(6,shuffle=True,random_state=seed+1).split(tr,y[tr]))
    else:
        a,b=next(StratifiedGroupKFold(6,shuffle=True,random_state=seed+1).split(tr,y[tr],groups[tr]))
    return tr[a],tr[b]

def build_features(X,y,names,tr,cache,use_ratios=True):
    """Return standardised feature matrix for ALL samples, with every data-dependent step fitted on `tr` only."""
    if os.path.exists(cache):
        z=np.load(cache,allow_pickle=True); return z['Z24'],z['Z44'],list(z['feat']),json.loads(str(z['table']))
    R,rn=log_ratios(X,names)
    chosen,table=select_ratios(R[tr],y[tr],rn)
    pt24=PowerTransformer().fit(X[tr]); Z24=pt24.transform(X)
    F=np.column_stack([X,R[:,chosen]])
    pt=PowerTransformer().fit(F[tr]); Z44=pt.transform(F)
    feat=names+[rn[j] for j in chosen]
    np.savez(cache,Z24=Z24,Z44=Z44,feat=np.array(feat),table=json.dumps(table))
    return Z24,Z44,feat,table

def scores(y,prob):
    p=prob.argmax(1); nc=prob.shape[1]
    pr,rc,f1,_=M.precision_recall_fscore_support(y,p,average='weighted',zero_division=0)
    try:
        auc_w=M.roc_auc_score(y,prob,multi_class='ovr',average='weighted'); auc_m=M.roc_auc_score(y,prob,multi_class='ovr',average='macro')
    except Exception: auc_w=auc_m=float('nan')
    return dict(acc=M.accuracy_score(y,p),bacc=M.balanced_accuracy_score(y,p),f1_macro=M.f1_score(y,p,average='macro'),
                prec_w=pr,rec_w=rc,f1_w=f1,kappa=M.cohen_kappa_score(y,p),mcc=M.matthews_corrcoef(y,p),auc_w=auc_w,auc_m=auc_m)

def boot_cm(yt,preds,groups,k,B=10000,seed=0):
    """Cluster bootstrap: resample groups with replacement (B replicates) and return the confusion matrices of every
    prediction vector in `preds`, shape (B, len(preds), k, k). The same resamples are used for all vectors (paired)."""
    ug,inv=np.unique(groups,return_inverse=True); G=len(ug); C=np.zeros((len(preds),G,k*k))
    for p,pr in enumerate(preds): np.add.at(C[p],(inv,yt*k+pr),1)
    W=np.random.default_rng(seed).multinomial(G,np.ones(G)/G,size=B).astype(float)
    return np.einsum('bg,pgc->bpc',W,C).reshape(B,len(preds),k,k)
def cm_acc(cm): return np.trace(cm,axis1=-2,axis2=-1)/cm.sum((-2,-1))
def cm_rec(cm): return np.diagonal(cm,axis1=-2,axis2=-1)/np.maximum(cm.sum(-1),1e-12)
def cm_bacc(cm):
    r=cm_rec(cm); ok=cm.sum(-1)>0; return (r*ok).sum(-1)/ok.sum(-1)
def q3(full,boot): return [float(full),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]

def component_groups(refs):
    """Group label per sample after merging reference codes that share any constituent reference (connected components)."""
    import re
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    parts=lambda s:[p.strip() for p in re.split(r';|/',s) if p.strip()]
    ucodes=pd.unique(refs); cons=sorted({p for s in ucodes for p in parts(s)}); ci={c:i for i,c in enumerate(cons)}; n=len(ucodes); r=[];c=[]
    for i,s in enumerate(ucodes):
        for p in parts(s): r.append(i); c.append(n+ci[p])
    A=coo_matrix((np.ones(len(r)),(r,c)),shape=(n+len(cons),n+len(cons))); _,lab=connected_components(A,directed=False)
    code2comp={s:lab[i] for i,s in enumerate(ucodes)}; return pd.factorize(np.array([code2comp[s] for s in refs]))[0]
