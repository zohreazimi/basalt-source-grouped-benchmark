"""Revision experiments: (a) grouping by connected components of shared constituent references, (b) immobile-element feature set,
(c) OPB by database, (d) BABB subduction indices by predicted class, (e) per-code table, (f) class medians, (g) MgO window."""
import sys,re; sys.path.insert(0, '.')
from ml_common import *
from models_ml import models
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
d,X,y,classes,names,groups=load(sys.argv[1]); out={}
refs=d['References'].astype(str).values
def parts(s): return [p.strip() for p in re.split(r';|/',s) if p.strip()]
comp_flag=np.array([len(parts(s))>1 for s in refs]); ucodes=pd.unique(refs)
out['composite']=dict(n_codes=int(len(ucodes)),n_composite_codes=int(sum(len(parts(s))>1 for s in ucodes)),n_rows_composite=int(comp_flag.sum()))
# connected components over constituents
cons=sorted({p for s in ucodes for p in parts(s)}); ci={c:i for i,c in enumerate(cons)}; n=len(ucodes); r=[];c=[]
for i,s in enumerate(ucodes):
    for p in parts(s): r.append(i); c.append(n+ci[p])
A=coo_matrix((np.ones(len(r)),(r,c)),shape=(n+len(cons),n+len(cons))); _,labc=connected_components(A,directed=False)
code2comp={s:labc[i] for i,s in enumerate(ucodes)}; g2=pd.factorize(np.array([code2comp[s] for s in refs]))[0]
sz=pd.Series(g2).value_counts(); big=sz.index[0]
out['components']=dict(n=int(len(sz)),largest=int(sz.iloc[0]),largest_classes={classes[k]:int(v) for k,v in pd.Series(y[g2==big]).value_counts().items()},rows_sharing=int(sum(sz[g]>pd.Series(refs[g2==g]).value_counts().iloc[0] for g in sz.index if pd.Series(refs[g2==g]).nunique()>1 for _ in [0]) if False else int(np.sum([pd.Series(refs[g2==g]).nunique()>1 for g in g2]))))
def run(prot_name,splits,Zsel,mnames,tag):
    res={}
    for mn in mnames:
        P=np.zeros((len(y),8)); fa=[];fb=[]
        for f,(tr,te) in enumerate(splits):
            Z=Zsel(f,tr); m=models()[mn]().fit(Z[tr],y[tr]); P[te]=m.predict_proba(Z[te]); p=P[te].argmax(1); fa.append(M.accuracy_score(y[te],p)); fb.append(M.balanced_accuracy_score(y[te],p))
        p=P.argmax(1); pr,rc,f1,_=M.precision_recall_fscore_support(y,p,zero_division=0)
        res[mn]=dict(acc=[float(np.mean(fa)),float(np.std(fa,ddof=1))],bacc=[float(np.mean(fb)),float(np.std(fb,ddof=1))],recall={classes[i]:float(rc[i]) for i in range(8)},precision={classes[i]:float(pr[i]) for i in range(8)},cm=M.confusion_matrix(y,p).tolist())
        np.save(f'res/oof/rev_{tag}_{mn.replace(" ","_")}.npy',P); print(tag,mn,[round(100*v,1) for v in res[mn]['acc']+res[mn]['bacc']],{k:round(v,2) for k,v in res[mn]['recall'].items()},flush=True)
    return res
MN=['LightGBM','SVM (RBF)','CatBoost']
# (a) component-grouped
sp_c=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(y,y,g2))
out['comp_fold_classes']=[np.bincount(y[te],minlength=8).tolist() for tr,te in sp_c]
def Zc(f,tr): return build_features(X,y,names,tr,f'cache/comp_{f}.npz')[1]
out['component_grouped']=run('comp',sp_c,Zc,MN,'comp')
np.save('res/groups_components.npy',g2)
# (b) immobile-element set
IM=['TiO2','Zr','Hf','Nb','Ta','Th','Y','La','Ce','Nd','Sm','Gd','Yb','Lu']; ii=[names.index(k) for k in IM]
def mkZim(tr):
    L=np.log(X[:,ii]); R,rn=log_ratios(X[:,ii],IM); F=np.column_stack([L,R]); return PowerTransformer().fit(F[tr]).transform(F)
for prot in ['random','grouped']:
    sp=make_splits(y,groups,prot); out[f'immobile_{prot}']=run(prot,sp,lambda f,tr:mkZim(tr),['LightGBM','SVM (RBF)'],f'immobile_{prot}')
out['immobile_comp']=run('comp',sp_c,lambda f,tr:mkZim(tr),['LightGBM','SVM (RBF)'],'immobile_comp')
# (g) MgO 5-10 window, grouped
mg=X[:,names.index('MgO')]; keep=(mg>=5)&(mg<=10); out['mgo']=dict(n_keep=int(keep.sum()),gt12=int((mg>12).sum()),lt4=int((mg<4).sum()))
sp=make_splits(y,groups,'grouped'); Pw=np.zeros((len(y),8))
fa=[];fb=[]
for f,(tr,te) in enumerate(sp):
    Z=np.load(f'cache/grouped_{f}.npz')['Z44']; tr2=tr[keep[tr]]; te2=te[keep[te]]; m=models()['LightGBM']().fit(Z[tr2],y[tr2]); p=m.predict(Z[te2]); fa.append(M.accuracy_score(y[te2],p)); fb.append(M.balanced_accuracy_score(y[te2],p))
out['mgo']['LightGBM']=dict(acc=[float(np.mean(fa)),float(np.std(fa,ddof=1))],bacc=[float(np.mean(fb)),float(np.std(fb,ddof=1))])
# reference: full-model grouped predictions restricted to the window
P=np.zeros((len(y),8))
for f,(tr,te) in enumerate(sp): P[te]=np.load(f'res/oof/grouped_{f}_LightGBM_44.npy')
p=P.argmax(1); out['mgo']['full_model_on_window']=dict(acc=float(M.accuracy_score(y[keep],p[keep])),bacc=float(M.balanced_accuracy_score(y[keep],p[keep])))
# (c) OPB by database
db=d['Database'].values; o=y==classes.index('OPB'); res={}
for D_ in ['GeoROC','PetDB']:
    m=o&(db==D_); res[D_]=dict(n=int(m.sum()),recall=float((p[m]==y[m]).mean()),pred={classes[k]:int(v) for k,v in pd.Series(p[m]).value_counts().items()})
out['opb_db']=res
# (e) per-code table for OPB, BABB, IOAB
tab=[]
for cl in ['OPB','BABB','IOAB']:
    m=y==classes.index(cl); s=pd.DataFrame(dict(code=refs[m],ok=(p[m]==y[m]),pred=p[m],db=db[m]))
    for code,gdf in s.groupby('code'):
        if len(gdf)>=15:
            wrong=gdf[~gdf.ok].pred; tab.append(dict(cls=cl,code=code,db=gdf.db.iloc[0],n=int(len(gdf)),recall=float(gdf.ok.mean()),main_wrong=classes[wrong.mode()[0]] if len(wrong) else '–'))
out['per_code']=tab
# (d) BABB subduction indices by predicted class
b=y==classes.index('BABB'); gi=lambda k:X[:,names.index(k)]
idx=dict(BaNb=gi('Ba')/gi('Nb'),ThNb=gi('Th')/gi('Nb'),NbLa=gi('Nb')/gi('La'))
res={}
for nm,pc in [('correct BABB',['BABB']),('predicted MORB',['MORB']),('predicted arc (IAB, IOAB, CAB)',['IAB','IOAB','CAB'])]:
    m=b&np.isin(p,[classes.index(k) for k in pc]); res[nm]=dict(n=int(m.sum()),**{k:float(np.median(v[m])) for k,v in idx.items()})
out['babb_idx']=res
from scipy.stats import mannwhitneyu
m1=b&(p==classes.index('MORB')); m2=b&np.isin(p,[classes.index(k) for k in ['IAB','IOAB','CAB']]); out['babb_test_p']={k:float(mannwhitneyu(v[m1],v[m2]).pvalue) for k,v in idx.items()}
# (f) class medians
med={}
for i,cl in enumerate(classes):
    m=y==i; med[cl]=dict(NbLa=float(np.median(idx['NbLa'][m])),ThNb=float(np.median(idx['ThNb'][m])),BaNb=float(np.median(idx['BaNb'][m])),TiO2=float(np.median(gi('TiO2')[m])),MgO=float(np.median(mg[m])),Fe2O3t=float(np.median(gi('Fe2O3t')[m])),K2O=float(np.median(gi('K2O')[m])))
out['medians']=med
json.dump(out,open('res/revision.json','w'),indent=1)
for k in ['composite','components','comp_fold_classes','mgo','opb_db','babb_idx','babb_test_p','medians']: print(k,json.dumps(out[k]))
print(pd.DataFrame(tab).to_string())
