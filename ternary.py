"""Data-driven ternary diagrams. Each triple of elements (a, b, c) defines a ternary plot after centring each element by its
training geometric mean; the two isometric log-ratio (ilr) coordinates of the triple are classified with class-balanced LDA.
Triples are selected by inner source-grouped CV on the training folds only (nested), and evaluated on the outer test folds."""
import sys,json,itertools; sys.path.insert(0, '.')
from ml_common import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
d,X,y,classes,names,groups=load(sys.argv[1]); L=np.log(X)
SCHEMES={'3 groups':{'IAB':'Arc','CAB':'Arc','IOAB':'Arc','MORB':'Ridge','BABB':'Ridge','OIB':'Intraplate','OPB':'Intraplate','CFB':'Intraplate'},
         '4 groups':{'IAB':'Arc','CAB':'Arc','IOAB':'Arc','MORB':'Ridge','BABB':'Ridge','OIB':'Oceanic intraplate','OPB':'Oceanic intraplate','CFB':'CFB'},
         }
IMMOBILE=['TiO2','Zr','Hf','Nb','Ta','Th','Y','La','Ce','Nd','Sm','Gd','Yb','Lu']

def ilr(tri):
    a,b,c=[L[:,names.index(e)] for e in tri]; return np.column_stack([(a-b)/np.sqrt(2),(a+b-2*c)/np.sqrt(6)])   # centring only shifts these coordinates
def cv_bacc(F,yt,idx,g,clf,n=3,seed=0):
    P=np.zeros(len(idx),int)
    for tr,te in StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(idx,yt[idx],g[idx]): P[te]=clf().fit(F[idx[tr]],yt[idx[tr]]).predict(F[idx[te]])
    return M.balanced_accuracy_score(yt[idx],P)
mk_lda=lambda:LDA(priors=None,solver="lsqr"); 
def mk(yt): 
    k=len(np.unique(yt)); return lambda:LDA(priors=np.full(k,1/k),solver="lsqr")
out={}
CLASSIC={'Th–Hf–Ta (Wood, 1980)':('Th','Hf','Ta'),'Nb–Zr–Y (Meschede, 1986)':('Nb','Zr','Y'),'Ti–Zr–Y (Pearce and Cann, 1973)':('TiO2','Zr','Y'),'La–Y–Nb (Cabanis and Lecolle, 1989)':('La','Y','Nb')}
outer=make_splits(y,groups,'grouped')
for sn,mp in SCHEMES.items():
    sup=sorted(set(mp.values())); yt=np.array([sup.index(mp[classes[i]]) for i in y]); res={}
    for pool_name,pool in [('immobile',IMMOBILE)]:
        triples=list(itertools.combinations(pool,3)); feats={t:ilr(t) for t in triples}
        P_l=np.zeros(len(y),int); P_s=np.zeros(len(y),int); chosen=[]
        for tr,te in outer:
            sc={t:cv_bacc(feats[t],yt,tr,groups,mk(yt)) for t in triples}; best=max(sc,key=sc.get); chosen.append(best)
            P_l[te]=mk(yt)().fit(feats[best][tr],yt[tr]).predict(feats[best][te])
        rc=M.recall_score(yt,P_l,average=None)
        res[pool_name]=dict(chosen=['–'.join(t) for t in chosen],lda_acc=float(M.accuracy_score(yt,P_l)),lda_bacc=float(rc.mean()),lda_recall={s:float(r) for s,r in zip(sup,rc)})
        # full-data ranking (descriptive): outer grouped CV score of every triple
        full={ '–'.join(t):cv_bacc(feats[t],yt,np.arange(len(y)),groups,mk(yt),n=5,seed=42) for t in triples}
        res[pool_name]['top10']=sorted(full.items(),key=lambda kv:-kv[1])[:10]
        print(sn,pool_name,'nested LDA acc %.1f bacc %.1f'%(100*res[pool_name]['lda_acc'],100*res[pool_name]['lda_bacc']),'chosen',res[pool_name]['chosen'],{k:round(100*v) for k,v in res[pool_name]['lda_recall'].items()},flush=True)
        print('   top5 (non-nested):',[(k,round(100*v,1)) for k,v in res[pool_name]['top10'][:5]],flush=True)
    out[sn]=res
json.dump(out,open('res/ternary.json','w'),indent=1)
