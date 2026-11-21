"""Coherent element-level permutation importance for LightGBM (ratios recomputed from the shuffled concentrations)."""
import sys; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
d,X,y,classes,names,groups=load(sys.argv[1]); out={}
R,rn=log_ratios(X,names); rng=np.random.default_rng(0); rec=[]; REP=3
BL={'8 major oxides':list(range(8)),'16 trace elements':list(range(8,24)),'all 24 concentrations':list(range(24)),'HFSE (TiO2, Zr, Hf, Nb, Ta)':[names.index(k) for k in ['TiO2','Zr','Hf','Nb','Ta']],
    'LILE (K2O, Rb, Ba, Sr)':[names.index(k) for k in ['K2O','Rb','Ba','Sr']],'REE and Y':[names.index(k) for k in ['La','Ce','Nd','Sm','Gd','Yb','Lu','Y']],'Th':[names.index('Th')]}
for f,(tr,te) in enumerate(make_splits(y,groups,'grouped')):
    z=np.load(f'cache/grouped_{f}.npz',allow_pickle=True); feat=list(z['feat']); sel=[rn.index(r) for r in feat[24:]]; pairs=[(names.index(r.split('/')[0]),names.index(r.split('/')[1])) for r in feat[24:]]
    F=np.column_stack([X,R[:,sel]]); pt=PowerTransformer().fit(F[tr]); assert np.allclose(pt.transform(F),z['Z44'],atol=1e-6)
    def Zperm(cols):
        Xs=X[te].copy(); pm=rng.permutation(len(te)); Xs[:,cols]=Xs[pm][:,cols]; L=np.log(Xs)
        return pt.transform(np.column_stack([Xs]+[L[:,i]-L[:,j] for i,j in pairs]))
    lgb=models()['LightGBM']().fit(z['Z44'][tr],y[tr])
    for mn,pf in [('LightGBM',lgb.predict)]:
        base=M.balanced_accuracy_score(y[te],pf(z['Z44'][te]))
        for j,nm in enumerate(names): rec.append(dict(model=mn,fold=f,item=nm,kind='element',drop=float(np.mean([base-M.balanced_accuracy_score(y[te],pf(Zperm([j]))) for _ in range(REP)]))))
        for nm,cols in BL.items(): rec.append(dict(model=mn,fold=f,item=nm,kind='block',drop=float(np.mean([base-M.balanced_accuracy_score(y[te],pf(Zperm(cols))) for _ in range(REP)]))))
    print('perm fold',f,flush=True)
pd.DataFrame(rec).to_csv('res/perm_coherent.csv',index=False)
a=pd.DataFrame(rec).groupby(['kind','item'])['drop'].agg(['mean','std']); print((100*a).round(1).sort_values('mean',ascending=False).to_string())
