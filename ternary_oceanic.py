"""Ternary diagram for oceanic basalts (oceanic arc = IAB + IOAB, MORB, OIB).
Nested selection among 364 triples (three scope definitions), the fixed Nb-La-Yb diagram, comparison with classical
coordinates and with multivariate models on identical folds, subgroup recalls, out-of-scope settings, (figures: fig7.py).
One classifier throughout: linear discriminant analysis, equal priors, solver 'lsqr'."""
import sys,json,itertools; sys.path.insert(0, '.')
from ml_common import *
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from lightgbm import LGBMClassifier
d,X,y,classes,names,groups=load(sys.argv[1]); L=np.log(X); g=lambda e:L[:,names.index(e)]; out={}
IMMOBILE=['TiO2','Zr','Hf','Nb','Ta','Th','Y','La','Ce','Nd','Sm','Gd','Yb','Lu']
def feats(t): a,b,c=[g(e) for e in t]; return np.column_stack([a-b,b-c])
triples=list(itertools.combinations(IMMOBILE,3)); FE={t:feats(t) for t in triples}; mk=lambda:LDA(priors=[1/3]*3,solver='lsqr')
SC={'oceanic arc / MORB / OIB':{'IAB':0,'IOAB':0,'MORB':1,'OIB':2},
    'oceanic arc / MORB / OIB + OPB':{'IAB':0,'IOAB':0,'MORB':1,'OIB':2,'OPB':2},
    'oceanic arc / MORB + OPB / OIB':{'IAB':0,'IOAB':0,'MORB':1,'OPB':1,'OIB':2}}
def summ(yt,idx,P,others=()):
    B=boot_cm(yt[idx],[P[idx]]+[o[idx] for o in others],groups[idx],3); cm=M.confusion_matrix(yt[idx],P[idx],labels=[0,1,2])
    r=dict(acc=q3(cm_acc(cm),cm_acc(B[:,0])),bacc=q3(cm_bacc(cm),cm_bacc(B[:,0])),recall=cm_rec(cm).tolist(),precision=(cm.diagonal()/cm.sum(0)).tolist(),cm=cm.tolist())
    return r,B
NEST={}
for sn,mp in SC.items():
    use=np.array([classes[i] in mp for i in y]); yt=np.array([mp.get(classes[i],0) for i in y]); idx=np.where(use)[0]
    def cvb(F,ii):
        P=np.zeros(len(ii),int)
        for tr,te in StratifiedGroupKFold(3,shuffle=True,random_state=0).split(ii,yt[ii],groups[ii]): P[te]=mk().fit(F[ii[tr]],yt[ii[tr]]).predict(F[ii[te]])
        return M.balanced_accuracy_score(yt[ii],P)
    P=np.zeros(len(y),int); ch=[]
    for tr_,te_ in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(idx,yt[idx],groups[idx]):
        tr,te=idx[tr_],idx[te_]; s={t:cvb(FE[t],tr) for t in triples}; b=max(s,key=s.get); ch.append('–'.join(b)); P[te]=mk().fit(FE[b][tr],yt[tr]).predict(FE[b][te])
    r,_=summ(yt,idx,P); r['chosen']=ch; r['n']=int(len(idx)); out['nested|'+sn]=r; NEST[sn]=P
    print(sn,len(idx),[round(100*v,1) for v in r['acc']+r['bacc']],ch,flush=True)
# ---- main scope ----
mp=SC['oceanic arc / MORB / OIB']; lab=['Oceanic arc (IAB, IOAB)','MORB','OIB']; use=np.array([classes[i] in mp for i in y]); yt=np.array([mp.get(classes[i],0) for i in y]); idx=np.where(use)[0]
SPG=[(idx[a],idx[b]) for a,b in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(idx,yt[idx],groups[idx])]
SPR=[(idx[a],idx[b]) for a,b in StratifiedKFold(5,shuffle=True,random_state=42).split(idx,yt[idx])]
def cv(F,sp,model=mk):
    P=np.zeros(len(y),int)
    for tr,te in sp: P[te]=model().fit(F[tr],yt[tr]).predict(F[te])
    return P
T=('Nb','La','Yb'); Pn=NEST['oceanic arc / MORB / OIB']; Pf=cv(feats(T),SPG); Pr=cv(feats(T),SPR)
CL={'Nb–La–Yb (triple fixed)':T,'La–Y–Nb':('La','Y','Nb'),'Th–Nb–Yb':('Th','Nb','Yb'),'Th–Hf–Ta':('Th','Hf','Ta'),'Nb–Zr–Y':('Nb','Zr','Y'),'Ti–Zr–Y':('TiO2','Zr','Y')}
PC={nm:cv(feats(t),SPG) for nm,t in CL.items()}
lgb=lambda:LGBMClassifier(n_estimators=600,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,class_weight='balanced',random_state=SEED,n_jobs=2,verbose=-1)
im=[names.index(e) for e in IMMOBILE]
PM={'LDA, 14 immobile elements':cv(L[:,im],SPG),'LightGBM, 14 immobile elements':cv(L[:,im],SPG,lgb),'LightGBM, 24 concentrations':cv(L,SPG,lgb),'QDA, Nb–La–Yb':cv(feats(T),SPG,lambda:QDA(priors=[1/3]*3))}
allp={'Nested selection':Pn,**PC,**PM,'Nb–La–Yb, random splits':Pr}; keys=list(allp); B=boot_cm(yt[idx],[allp[k][idx] for k in keys],groups[idx],3)
res={}
for i,k in enumerate(keys):
    cm=M.confusion_matrix(yt[idx],allp[k][idx],labels=[0,1,2]); res[k]=dict(acc=q3(cm_acc(cm),cm_acc(B[:,i])),bacc=q3(cm_bacc(cm),cm_bacc(B[:,i])),recall=cm_rec(cm).tolist(),precision=(cm.diagonal()/cm.sum(0)).tolist(),cm=cm.tolist())
    a0=cm_acc(M.confusion_matrix(yt[idx],Pn[idx])); b0=cm_bacc(M.confusion_matrix(yt[idx],Pn[idx]))
    res[k]['minus_nested']=dict(acc=q3(res[k]['acc'][0]-a0,cm_acc(B[:,i])-cm_acc(B[:,0])),bacc=q3(res[k]['bacc'][0]-b0,cm_bacc(B[:,i])-cm_bacc(B[:,0])))
    print(k,[round(100*v,1) for v in res[k]['acc']+res[k]['bacc']],'| minus nested',[round(100*v,1) for v in res[k]['minus_nested']['acc']+res[k]['minus_nested']['bacc']],flush=True)
out['same_folds']=res; i1,i2=keys.index('Nb–La–Yb, random splits'),keys.index('Nb–La–Yb (triple fixed)')
out['random_minus_grouped']=dict(acc=q3(res[keys[i1]]['acc'][0]-res[keys[i2]]['acc'][0],cm_acc(B[:,i1])-cm_acc(B[:,i2])),bacc=q3(res[keys[i1]]['bacc'][0]-res[keys[i2]]['bacc'][0],cm_bacc(B[:,i1])-cm_bacc(B[:,i2])))
# subgroup recalls (nested predictions and fixed diagram)
LaSmN=(X[:,names.index('La')]/X[:,names.index('Sm')])/(0.237/0.148); sub={}
for nm,m in [('IAB',y==classes.index('IAB')),('IOAB',y==classes.index('IOAB')),('MORB, (La/Sm)N < 1',(y==classes.index('MORB'))&(LaSmN<1)),('MORB, (La/Sm)N ≥ 1',(y==classes.index('MORB'))&(LaSmN>=1)),('OIB',y==classes.index('OIB'))]:
    sub[nm]=dict(n=int(m.sum()),recall_nested=float((Pn[m]==yt[m]).mean()),recall_fixed=float((Pf[m]==yt[m]).mean()),fields_fixed=[float((Pf[m]==j).mean()) for j in range(3)])
out['subgroups']=sub; print(json.dumps(sub))
F=feats(T); m2=mk().fit(F[idx],yt[idx]); coef,inter=m2.coef_,m2.intercept_
assert (np.argmax(F[idx]@coef.T+inter,1)==m2.predict(F[idx])).all()
out.update(n=int(len(idx)),n_by_class=np.bincount(yt[idx]).tolist(),coef=coef.tolist(),intercept=inter.tolist(),apparent_acc=float((m2.predict(F[idx])==yt[idx]).mean()))
out['excluded']={c:[float((m2.predict(F[y==classes.index(c)])==j).mean()) for j in range(3)] for c in ['BABB','OPB','CFB','CAB']}
MULT={'Nb':2.0,'La':1.0,'Yb':3.0}; out['mult']=MULT; json.dump(out,open('res/ternary_oceanic.json','w'),indent=1); print('excluded',out['excluded'])
