"""Two-step ternary diagrams: step 1 arc vs non-arc; step 2 ridge vs intraplate (non-arc samples only). Nested selection as in ternary.py."""
import sys,json,itertools; sys.path.insert(0, '.')
from ml_common import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
d,X,y,classes,names,groups=load(sys.argv[1]); L=np.log(X)
ARC=[classes.index(c) for c in ['IAB','CAB','IOAB']]; RIDGE=[classes.index(c) for c in ['MORB','BABB']]
IMMOBILE=['TiO2','Zr','Hf','Nb','Ta','Th','Y','La','Ce','Nd','Sm','Gd','Yb','Lu']; ALL17=IMMOBILE+['Sr','Ba','Rb']
def ilr(tri):
    a,b,c=[L[:,names.index(e)] for e in tri]; return np.column_stack([(a-b)/np.sqrt(2),(a+b-2*c)/np.sqrt(6)])
mk=lambda:LDA(priors=[0.5,0.5],solver="lsqr")
def cvb(F,yt,idx,n=3,seed=0):
    P=np.zeros(len(idx),int)
    for tr,te in StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(idx,yt[idx],groups[idx]): P[te]=mk().fit(F[idx[tr]],yt[idx[tr]]).predict(F[idx[te]])
    return M.balanced_accuracy_score(yt[idx],P)
outer=make_splits(y,groups,'grouped'); out={}
y1=np.isin(y,ARC).astype(int); nonarc=~np.isin(y,ARC); y2=np.isin(y,RIDGE).astype(int)
for pool_name,pool in [('immobile',IMMOBILE)]:
    triples=list(itertools.combinations(pool,3)); feats={t:ilr(t) for t in triples}
    P1=np.zeros(len(y),int); P2=np.zeros(len(y),int); ch1=[];ch2=[]
    for tr,te in outer:
        s1={t:cvb(feats[t],y1,tr) for t in triples}; b1=max(s1,key=s1.get); ch1.append('–'.join(b1)); m1=mk().fit(feats[b1][tr],y1[tr]); P1[te]=m1.predict(feats[b1][te])
        tr2=tr[nonarc[tr]]; s2={t:cvb(feats[t],y2,tr2) for t in triples}; b2=max(s2,key=s2.get); ch2.append('–'.join(b2)); m2=mk().fit(feats[b2][tr2],y2[tr2]); P2[te]=m2.predict(feats[b2][te])
    # combined 3-group prediction: arc if step1 says arc, else ridge/intraplate from step 2
    y3=np.where(np.isin(y,ARC),0,np.where(np.isin(y,RIDGE),1,2)); p3=np.where(P1==1,0,np.where(P2==1,1,2)); rc=M.recall_score(y3,p3,average=None)
    r=dict(step1=dict(chosen=ch1,acc=float(M.accuracy_score(y1,P1)),bacc=float(M.balanced_accuracy_score(y1,P1)),sens_arc=float(M.recall_score(y1,P1)),spec=float(M.recall_score(1-y1,1-P1))),
           step2=dict(chosen=ch2,acc=float(M.accuracy_score(y2[nonarc],P2[nonarc])),bacc=float(M.balanced_accuracy_score(y2[nonarc],P2[nonarc])),rec_ridge=float(M.recall_score(y2[nonarc],P2[nonarc])),rec_intraplate=float(M.recall_score(1-y2[nonarc],1-P2[nonarc]))),
           combined=dict(acc=float(M.accuracy_score(y3,p3)),bacc=float(rc.mean()),recall=dict(Arc=float(rc[0]),Ridge=float(rc[1]),Intraplate=float(rc[2]))))
    r['top1']=sorted({'–'.join(t):cvb(feats[t],y1,np.arange(len(y)),n=5,seed=42) for t in triples}.items(),key=lambda kv:-kv[1])[:8]
    idx2=np.where(nonarc)[0]; r['top2']=sorted({'–'.join(t):cvb(feats[t],y2,idx2,n=5,seed=42) for t in triples}.items(),key=lambda kv:-kv[1])[:8]
    out[pool_name]=r; print(pool_name,json.dumps({k:r[k] for k in ['step1','step2','combined']}),flush=True); print('  top step1',[(k,round(100*v,1)) for k,v in r['top1'][:5]]); print('  top step2',[(k,round(100*v,1)) for k,v in r['top2'][:5]],flush=True)
    # per fine class: where do they go in step 1 / step 2
    r['arc_share']={c:float(P1[y==i].mean()) for i,c in enumerate(classes)}; print('  step1 predicted-arc share by setting',{c:round(100*P1[y==i].mean()) for i,c in enumerate(classes)}); print('  step2 predicted-ridge share by setting',{c:round(100*P2[y==i].mean()) for i,c in enumerate(classes) if i not in ARC},flush=True)
json.dump(out,open('res/ternary2.json','w'),indent=1)
