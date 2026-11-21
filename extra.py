"""Small post-hoc items: BABB recall without three codes of doubtful back-arc affinity, MORB field share by enrichment,
duplicates by class, triple point of the Nb-La-Yb diagram. Writes res/extra.json."""
import sys,json; sys.path.insert(0, '.')
from ml_common import *
d,X,y,classes,names,groups=load(sys.argv[1]); refs=d['References'].astype(str).values; out={}; gi=lambda k:X[:,names.index(k)]
P=np.zeros((len(y),8))
for f,(tr,te) in enumerate(make_splits(y,groups,'grouped')): P[te]=np.load(f'res/oof/grouped_{f}_LightGBM_44.npy')
p=P.argmax(1); b=y==classes.index('BABB'); doubt=np.array([any(k in s for k in ['TAMURA, 2011','HICKEY-VARGAS, 2018','ISHIZUKA, 2009']) for s in refs])
out['babb']=dict(n_doubt=int((b&doubt).sum()),recall_doubt=float((p[b&doubt]==y[b&doubt]).mean()),recall_rest=float((p[b&~doubt]==y[b&~doubt]).mean()),n_rest=int((b&~doubt).sum()))
pairs=np.load('res/dup_pairs.npy'); ind=np.unique(pairs); out['dups_by_class']={classes[c]:int((y[ind]==c).sum()) for c in range(8)}; out['dups_n']=int(len(ind))
T=json.load(open('res/ternary_oceanic.json')); coef=np.array(T['coef']); ic=np.array(T['intercept']); F=np.column_stack([np.log(gi('Nb')/gi('La')),np.log(gi('La')/gi('Yb'))]); z=np.argmax(F@coef.T+ic,1)
r=(gi('La')/gi('Sm'))/(0.237/0.148); m=y==classes.index('MORB'); out['morb_bins']={nm:dict(n=int((m&(r>=lo)&(r<hi)).sum()),morb_field=float((z[m&(r>=lo)&(r<hi)]==1).mean())) for nm,lo,hi in [('<1',0,1),('1-1.5',1,1.5),('1.5-2',1.5,2),('>=2',2,99)]}
A=np.array([coef[0]-coef[1],coef[1]-coef[2]]); bb=-np.array([ic[0]-ic[1],ic[1]-ic[2]]); tp=np.linalg.solve(A,bb); out['triple']=dict(NbLa=float(np.exp(tp[0])),LaYb=float(np.exp(tp[1])))
ia=(y==classes.index('IAB'))&(z==2); out['iab_in_oib']=dict(n=int(ia.sum()),NbLa=float(np.median(gi('Nb')[ia]/gi('La')[ia])),LaYb=float(np.median(gi('La')[ia]/gi('Yb')[ia])))
o=y==classes.index('OPB'); out['opb_morb_field_LaYb']=float(np.median((gi('La')/gi('Yb'))[o]))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
mp={'IAB':0,'IOAB':0,'MORB':1,'OIB':2}; yt=np.array([mp.get(classes[i],0) for i in y]); idx=np.where(np.array([classes[i] in mp for i in y]))[0]; Pq=np.zeros(len(y),int)
for a_,b_ in StratifiedGroupKFold(5,shuffle=True,random_state=42).split(idx,yt[idx],groups[idx]): Pq[idx[b_]]=LDA(priors=[1/3]*3,solver='svd').fit(F[idx[a_]],yt[idx[a_]]).predict(F[idx[b_]])
out['fixed_svd_acc']=float((Pq[idx]==yt[idx]).mean())
json.dump(out,open('res/extra.json','w'),indent=1); print(json.dumps(out))
