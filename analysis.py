"""All post-hoc statistics for the paper, computed from saved out-of-fold probabilities. Writes res/V.json.
Intervals: cluster bootstrap over reference codes, 10,000 replicates, one set of resamples for every quantity (seed 0)."""
import sys,re,itertools; sys.path.insert(0, '.')
from ml_common import *
d,X,y,classes,names,groups=load(sys.argv[1]); V={}; refs=d['References'].astype(str).values; db=d['Database'].values; allc=np.arange(8); A=np.arange(len(y)); ci_=classes.index
SPL={p:make_splits(y,groups,p) for p in ['random','grouped']}
def oof(prot,fmt,k=8):
    P=np.zeros((len(y),k))
    for f,(tr,te) in enumerate(SPL[prot]): P[te]=np.load('res/oof/'+fmt.format(prot=prot,f=f))
    return P
MODELS=['LightGBM','XGBoost','CatBoost','Extra Trees','Random Forest','KNN','SVM (RBF)','Logistic regression','LDA','SVM (linear)','Decision tree','AdaBoost','Naive Bayes','Dummy (majority)']
fn=lambda m,fs='44':'{prot}_{f}_'+m.replace(' ','_')+f'_{fs}.npy'
PR={p:{m:oof(p,fn(m)) for m in MODELS} for p in SPL}
for p in SPL:
    for m in ['LightGBM','SVM (RBF)','Extra Trees','KNN']: PR[p][m+' [24]']=oof(p,fn(m,'24'))
    PR[p]['Ensemble']=(PR[p]['LightGBM']+PR[p]['SVM (RBF)']+PR[p]['CatBoost'])/3
def foldstats(prot,P):
    r={k:[] for k in ['acc','bacc','f1_macro','mcc','auc_m']}
    for tr,te in SPL[prot]:
        s=scores(y[te],P[te])
        for k in r: r[k].append(s[k])
    return {k:[float(np.mean(v)),float(np.std(v,ddof=1))] for k,v in r.items()}
V['models']={m:{p:foldstats(p,PR[p][m]) for p in SPL} for m in PR['random']}
CM=lambda p,yt=y,k=8:M.confusion_matrix(yt,p,labels=np.arange(k)).astype(float)
def BT(preds,yt=y,k=8,g=groups): return boot_cm(yt,preds,g,k)
# ---- pooled estimates with intervals for every model, and the loss between protocols ----
names_m=list(PR['random']); pg_={m:PR['grouped'][m].argmax(1) for m in names_m}; pr_={m:PR['random'][m].argmax(1) for m in names_m}
B=BT([pg_[m] for m in names_m]+[pr_[m] for m in names_m]); n=len(names_m); pooled={}
for i,m in enumerate(names_m):
    cg,cr=CM(pg_[m]),CM(pr_[m]); bg,br=B[:,i],B[:,n+i]
    pooled[m]=dict(acc=q3(cm_acc(cg),cm_acc(bg)),bacc=q3(cm_bacc(cg),cm_bacc(bg)),acc_r=q3(cm_acc(cr),cm_acc(br)),bacc_r=q3(cm_bacc(cr),cm_bacc(br)),
        gap_acc=q3(cm_acc(cr)-cm_acc(cg),cm_acc(br)-cm_acc(bg)),gap_bacc=q3(cm_bacc(cr)-cm_bacc(cg),cm_bacc(br)-cm_bacc(bg)),
        recall={classes[c]:q3(cm_rec(cg)[c],cm_rec(bg)[:,c]) for c in range(8)},recall_r={classes[c]:q3(cm_rec(cr)[c],cm_rec(br)[:,c]) for c in range(8)},
        f1_macro=float(M.f1_score(y,pg_[m],average='macro')),kappa=float(M.cohen_kappa_score(y,pg_[m])))
V['ci']=pooled; print({m:[round(100*x,1) for x in pooled[m]['acc']+pooled[m]['bacc']] for m in ['LightGBM','CatBoost','SVM (RBF)','Ensemble']},flush=True)
V['paired']={}
for p,PP in [('grouped',pg_),('random',pr_)]:
    off=0 if p=='grouped' else n
    for a,b in [('CatBoost','XGBoost'),('CatBoost','LightGBM'),('SVM (RBF)','LightGBM'),('Ensemble','LightGBM'),('Ensemble','CatBoost'),('LightGBM','XGBoost'),('CatBoost','Random Forest'),('CatBoost','SVM (RBF)'),('LightGBM','KNN'),('LightGBM','LightGBM [24]'),('SVM (RBF)','SVM (RBF) [24]'),('LightGBM','LDA')]:
        ia,ib=off+names_m.index(a),off+names_m.index(b); ca,cb=CM(PP[a]),CM(PP[b])
        V['paired'][f'{p}|{a}|{b}']=dict(acc=q3(cm_acc(ca)-cm_acc(cb),cm_acc(B[:,ia])-cm_acc(B[:,ib])),bacc=q3(cm_bacc(ca)-cm_bacc(cb),cm_bacc(B[:,ia])-cm_bacc(B[:,ib])))
# intervals with merged reference components as the resampling unit
g2=component_groups(refs); B2=BT([pg_['LightGBM'],pr_['LightGBM']],g=g2); cg,cr=CM(pg_['LightGBM']),CM(pr_['LightGBM'])
V['ci_comp']=dict(acc=q3(cm_acc(cg),cm_acc(B2[:,0])),bacc=q3(cm_bacc(cg),cm_bacc(B2[:,0])),gap_bacc=q3(cm_bacc(cr)-cm_bacc(cg),cm_bacc(B2[:,1])-cm_bacc(B2[:,0])),gap_acc=q3(cm_acc(cr)-cm_acc(cg),cm_acc(B2[:,1])-cm_acc(B2[:,0])))
# ---- class-wise, confusion, per-code ----
V['per_class']={}; V['cm']={}
for p in SPL:
    for k in ['LightGBM','SVM (RBF)','CatBoost','Ensemble']:
        pp=PR[p][k].argmax(1); a_,rc,f1,sup=M.precision_recall_fscore_support(y,pp,zero_division=0); V['per_class'][f'{p}|{k}']=dict(precision=a_.tolist(),recall=rc.tolist(),f1=f1.tolist(),n=sup.tolist()); V['cm'][f'{p}|{k}']=M.confusion_matrix(y,pp).tolist()
pg=pg_['LightGBM']; pr=pr_['LightGBM']; pub={}
for p,pp in [('random',pr),('grouped',pg)]:
    s=pd.DataFrame(dict(g=groups,ok=pp==y,y=y)).groupby('g').agg(n=('ok','size'),acc=('ok','mean'),c=('y',lambda v:v.mode()[0])); pub[p+'_macro']=float(s.acc.mean()); s=s[s.n>=10]
    pub[p]=dict(n=int(len(s)),above90=float((s.acc>=0.9).mean()),below50=float((s.acc<0.5).mean()),acc=s.acc.round(4).tolist(),by_class={classes[c]:[int((g_.acc<0.5).sum()),int(len(g_))] for c,g_ in s.groupby('c')})
V['pub']=pub
absent=np.zeros(len(y),bool)
for tr,te in SPL['random']: absent[te]=~np.isin(groups[te],groups[tr])
gsz=pd.Series(groups).map(pd.Series(groups).value_counts()).values
V['within_random']=dict(n_absent=int(absent.sum()),acc_absent=float((pr[absent]==y[absent]).mean()),acc_present=float((pr[~absent]==y[~absent]).mean()),acc_absent_grouped=float((pg[absent]==y[absent]).mean()))
V['by_size']={nm:dict(n=int(m.sum()),random=float((pr[m]==y[m]).mean()),grouped=float((pg[m]==y[m]).mean())) for nm,m in [('1',gsz==1),('2-9',(gsz>=2)&(gsz<10)),('10-49',(gsz>=10)&(gsz<50)),('50+',gsz>=50)]}
parts=lambda s:[q.strip() for q in re.split(r';|/',s) if q.strip()]; link=np.zeros(len(y),bool)
for tr,te in SPL['grouped']:
    S=set(q for s in pd.unique(refs[tr]) for q in parts(s)); link[te]=[any(q in S for q in parts(s)) for s in refs[te]]
V['linked']=dict(share=float(link.mean()),acc_linked=float((pg[link]==y[link]).mean()),acc_unlinked=float((pg[~link]==y[~link]).mean()),
    by_class={classes[c]:dict(n_linked=int((link&(y==c)).sum()),n_unlinked=int((~link&(y==c)).sum()),rec_linked=float((pg[link&(y==c)]==c).mean()) if (link&(y==c)).any() else None,rec_unlinked=float((pg[~link&(y==c)]==c).mean())) for c in range(8)},
    bacc_unlinked=float(M.balanced_accuracy_score(y[~link],pg[~link])))
# pooled values for the sensitivity partitions and the alteration-resistant feature set (from the saved confusion matrices)
RVJ=json.load(open('res/revision.json')); rp={}
for tag,key in [('comp','component_grouped'),('immobile_random','immobile_random'),('immobile_grouped','immobile_grouped'),('immobile_comp','immobile_comp')]:
    for mn,v in RVJ[key].items(): c=np.array(v['cm'],float); rp[f'{tag}|{mn}']=dict(acc=float(cm_acc(c)),bacc=float(cm_bacc(c)))
V['rev_pooled']=rp; print('rev_pooled',{k:(round(100*v['acc'],1),round(100*v['bacc'],1)) for k,v in rp.items()})
lkm=link; V['linked']['share_morb']=float((y[lkm]==ci_('MORB')).mean())
# ---- near-duplicate analyses ----
DU=json.load(open('res/dups.json')); pairs=np.load('res/dup_pairs.npy'); dupof={}
for i,j in pairs: dupof.setdefault(i,[]).append(j); dupof.setdefault(j,[]).append(i)
dt=np.zeros(len(y),bool)
for tr,te in SPL['grouped']:
    S=set(tr); dt[te]=[any(k in S for k in dupof.get(i,[])) for i in te]
conf=np.zeros(len(y),bool)
for i,j in pairs:
    if y[i]!=y[j]: conf[i]=conf[j]=True
DU.update(n_test_dup_in_train=int(dt.sum()),acc_dup=float((pg[dt]==y[dt]).mean()),acc_rest=float((pg[~dt]==y[~dt]).mean()),bacc_rest=float(M.balanced_accuracy_score(y[~dt],pg[~dt])),n_conflict=int(conf.sum()),conflict_by_class={classes[c]:int((conf&(y==c)).sum()) for c in range(8)})
o_=y==ci_('OPB'); DU['opb']=dict(n_dup=int((dt&o_).sum()),recall_dup=float((pg[dt&o_]==y[dt&o_]).mean()),n_rest=int((~dt&o_).sum()),recall_rest=float((pg[~dt&o_]==y[~dt&o_]).mean()))
ind_=np.unique(pairs); DU['by_class']={classes[c]:int((y[ind_]==c).sum()) for c in range(8)}
if os.path.exists('res/dedup.json'):
    DD=json.load(open('res/dedup.json')); keep=np.load('res/dedup_keep.npy'); gd=np.load('res/dedup_groups.npy')
    for mn in ['LightGBM','SVM (RBF)','CatBoost']:
        P=np.load(f'res/oof/dedup_{mn.replace(" ","_")}.npy')[keep].argmax(1); Bd=boot_cm(y[keep],[P],gd[keep],8); c=CM(P,y[keep]); DD[mn].update(acc_ci=q3(cm_acc(c),cm_acc(Bd[:,0])),bacc_ci=q3(cm_bacc(c),cm_bacc(Bd[:,0])))
    DU['dedup']=DD
V['dups']=DU
# BABB label noise: recall without the analyses that carry a conflicting duplicate, and without two codes of doubtful affinity
b=y==ci_('BABB'); doubt=np.array([any(k in s for k in ['HICKEY-VARGAS, 2018','ISHIZUKA, 2009']) for s in refs])
V['babb_noise']=dict(recall_all=float((pg[b]==y[b]).mean()),recall_noconflict=float((pg[b&~conf]==y[b&~conf]).mean()),n_doubt=int((b&doubt).sum()),recall_nodoubt=float((pg[b&~doubt&~conf]==y[b&~doubt&~conf]).mean()))
# OPB: chemistry of the analyses by predicted class
gi=lambda k:X[:,names.index(k)]; o=y==ci_('OPB'); r={}
for nm,pc in [('correct OPB',['OPB']),('predicted CFB',['CFB']),('predicted MORB',['MORB']),('predicted OIB',['OIB'])]:
    m=o&np.isin(pg,[ci_(k) for k in pc]); r[nm]=dict(n=int(m.sum()),LaYb=float(np.median((gi('La')/gi('Yb'))[m])),NbZr=float(np.median((gi('Nb')/gi('Zr'))[m])),ThNb=float(np.median((gi('Th')/gi('Nb'))[m])))
r['class medians']={c:float(np.median((gi('La')/gi('Yb'))[y==ci_(c)])) for c in ['OPB','CFB','MORB','OIB']}; V['opb_idx']=r
# ---- calibration and selective prediction ----
def ece(P):
    c=P.max(1); ok=P.argmax(1)==y; e=0
    for lo in np.linspace(0,1,10,endpoint=False):
        m=(c>lo)&(c<=lo+0.1)
        if m.any(): e+=m.mean()*abs(ok[m].mean()-c[m].mean())
    return float(e)
cal={}
for k in ['LightGBM','SVM (RBF)','CatBoost','XGBoost','Random Forest','Ensemble']:
    r={}
    for p in SPL:
        P=PR[p][k]; pp=P.argmax(1); r[p]=dict(ece=ece(P),conf=float(P.max(1).mean()),acc=float((pp==y).mean()),nll=float(M.log_loss(y,np.clip(P,1e-12,1),labels=allc)))
    P=PR['grouped'][k]; pp=P.argmax(1); cf=P.max(1); r['sel']={}
    for th in [0.5,0.7,0.9,0.95]:
        m=cf>=th; r['sel'][str(th)]=dict(coverage=float(m.mean()),acc=float((pp[m]==y[m]).mean()),bacc=float(M.balanced_accuracy_score(y[m],pp[m])))
    keep_=cf>=0.9; r['cov_by_class']={classes[i]:[float(keep_[y==i].mean()),float((pp[keep_&(y==i)]==i).mean()) if (keep_&(y==i)).any() else None] for i in range(8)}
    r['retained_cfb_morb']=float(np.isin(y[keep_],[ci_('CFB'),ci_('MORB')]).mean())
    order=np.argsort(-cf); r['risk_cov']=[(float((i+1)/len(y)),float((pp[order[:i+1]]==y[order[:i+1]]).mean())) for i in range(199,len(y),200)]
    cal[k]=r
V['cal']=cal
# ---- augmentation ----
E=pd.DataFrame([json.loads(l) for l in open('res/exp.jsonl')]); aug={}
def xo(prot,tag,k=8): return oof(prot,'x_{prot}_{f}_'+tag+'.npy',k)
def seeds_of(p,method,mn,target=1500): return [s for s in [1,2,3] if len(E[(E.kind=='aug')&(E.protocol==p)&(E.method==method)&(E.seed==s)&(E.target==target)&(E.model==mn)])==5]
def smean(fn_,cms): return np.mean([fn_(c) for c in cms],0)
STORE={}
for p in SPL:
    for mn in ['LightGBM','SVM (RBF)','CatBoost']:
        p0=PR[p][mn].argmax(1)
        for method,target in [(m_,1500) for m_ in ['xsource','noise','smote','dup','within','noise_all']]+[('xsource',750),('xsource',3000)]:
            ss=seeds_of(p,method,mn,target)
            if not ss: continue
            ps=[xo(p,f'aug_{method}_s{s}_t{target}_{mn.replace(" ","_")}').argmax(1) for s in ss]; Bb=BT([p0]+ps); c0=CM(p0); cs=[CM(q) for q in ps]; key=f'{p}|{mn}|{method}'+('' if target==1500 else f'|{target}'); STORE[key]=(ps,Bb)
            ba=np.mean([cm_acc(Bb[:,1+i]) for i in range(len(ps))],0); bb=np.mean([cm_bacc(Bb[:,1+i]) for i in range(len(ps))],0); brc=np.mean([cm_rec(Bb[:,1+i]) for i in range(len(ps))],0)
            aug[key]=dict(n_seeds=len(ss),acc=float(smean(cm_acc,cs)),bacc=float(smean(cm_bacc,cs)),bacc_seeds=[float(cm_bacc(c)) for c in cs],d_acc=q3(smean(cm_acc,cs)-cm_acc(c0),ba-cm_acc(Bb[:,0])),d_bacc=q3(smean(cm_bacc,cs)-cm_bacc(c0),bb-cm_bacc(Bb[:,0])),
                d_recall={classes[i]:q3(smean(cm_rec,cs)[i]-cm_rec(c0)[i],brc[:,i]-cm_rec(Bb[:,0])[:,i]) for i in range(8)},recall={classes[i]:float(smean(cm_rec,cs)[i]) for i in range(8)})
            print(key,len(ss),[round(100*x,1) for x in aug[key]['d_acc']+aug[key]['d_bacc']],flush=True)
con={}
def contrast(k1,k2):   # same resamples (seed 0) in both calls, so the difference is paired
    (p1,B1),(p2,B2_)=STORE[k1],STORE[k2]; f=lambda ps,Bb:(np.mean([cm_bacc(CM(q)) for q in ps]),np.mean([cm_bacc(Bb[:,1+i]) for i in range(len(ps))],0)); a,ba=f(p1,B1); b_,bb=f(p2,B2_); return q3(a-b_,ba-bb)
for a,b_ in [('xsource','noise'),('xsource','smote'),('xsource','dup'),('xsource','within'),('noise','dup'),('noise','smote')]:
    if f'grouped|LightGBM|{a}' in STORE and f'grouped|LightGBM|{b_}' in STORE: con[f'{a}|{b_}']=contrast(f'grouped|LightGBM|{a}',f'grouped|LightGBM|{b_}')
ps,Bb=STORE['grouped|LightGBM|xsource']; Bc=BT([pg_['CatBoost']]); con['xsource LightGBM|plain CatBoost']=q3(np.mean([cm_bacc(CM(q)) for q in ps])-cm_bacc(CM(pg_['CatBoost'])),np.mean([cm_bacc(Bb[:,1+i]) for i in range(3)],0)-cm_bacc(Bc[:,0]))
aug['contrasts']=con; print('contrasts',{k:[round(100*x,1) for x in v] for k,v in con.items()})
sd=[xo('grouped',f'seed{s}_LightGBM').argmax(1) for s in [1,2,3] if len(E[(E.kind=='seed')&(E.seed==s)])==5]
V['seedvar']=dict(bacc=[float(cm_bacc(CM(q))) for q in sd]+[float(cm_bacc(CM(pg)))],acc=[float(cm_acc(CM(q))) for q in sd]+[float(cm_acc(CM(pg)))])
V['aug']=aug
# ---- label hierarchy ----
SCH={'8':{c:c for c in classes},'6':{'IAB':'Arc','CAB':'Arc','IOAB':'Arc','BABB':'BABB','MORB':'MORB','OIB':'OIB','OPB':'OPB','CFB':'CFB'},
     '4':{'IAB':'Arc','CAB':'Arc','IOAB':'Arc','BABB':'Ridge','MORB':'Ridge','OIB':'Oceanic intraplate','OPB':'Oceanic intraplate','CFB':'Continental flood'},
     '3':{'IAB':'Arc','CAB':'Arc','IOAB':'Arc','BABB':'Ridge','MORB':'Ridge','OIB':'Intraplate','OPB':'Intraplate','CFB':'Intraplate'}}
hier={}
def coarse(P,mp):
    sup=sorted(set(mp.values())); yt=np.array([sup.index(mp[classes[i]]) for i in y]); Q=np.stack([P[:,[j for j,c in enumerate(classes) if mp[c]==s]].sum(1) for s in sup],1); return sup,yt,Q.argmax(1)
for mn in ['LightGBM','Ensemble']:
    for sn,mp in SCH.items():
        sup,yt,qg=coarse(PR['grouped'][mn],mp); _,_,qr=coarse(PR['random'][mn],mp); k=len(sup); Bh=BT([qg,qr],yt,k); cg,cr=CM(qg,yt,k),CM(qr,yt,k)
        hier[f'grouped|{mn}|{sn}']=dict(acc=q3(cm_acc(cg),cm_acc(Bh[:,0])),bacc=q3(cm_bacc(cg),cm_bacc(Bh[:,0])),kappa=float(M.cohen_kappa_score(yt,qg)),mcc=float(M.matthews_corrcoef(yt,qg)),recall={s:q3(cm_rec(cg)[i],cm_rec(Bh[:,0])[:,i]) for i,s in enumerate(sup)},
            gap_acc=q3(cm_acc(cr)-cm_acc(cg),cm_acc(Bh[:,1])-cm_acc(Bh[:,0])),gap_bacc=q3(cm_bacc(cr)-cm_bacc(cg),cm_bacc(Bh[:,1])-cm_bacc(Bh[:,0])),
            fine_to_group={c:float((qg[y==i]==yt[y==i]).mean()) for i,c in enumerate(classes)},fine_given_group=float((PR['grouped'][mn].argmax(1)[qg==yt]==y[qg==yt]).mean()))
        hier[f'random|{mn}|{sn}']=dict(acc=q3(cm_acc(cr),cm_acc(Bh[:,1])),bacc=q3(cm_bacc(cr),cm_bacc(Bh[:,1])),kappa=float(M.cohen_kappa_score(yt,qr)),mcc=float(M.matthews_corrcoef(yt,qr)))
# reference: every other way of merging the eight settings into groups of the same sizes
def partitions(sizes):
    seen=set()
    def rec(rem,sz):
        if not sz: yield (); return
        for c in itertools.combinations(rem,sz[0]):
            for r in rec([x for x in rem if x not in c],sz[1:]): yield (c,)+r
    for p in rec(list(range(8)),sizes):
        k=frozenset(frozenset(c) for c in p)
        if k not in seen: seen.add(k); yield p
P8=PR['grouped']['LightGBM']
for sn,sizes in [('3',(3,3,2)),('4',(3,2,2,1))]:
    kv=[];bv=[];gv=[]; P8r=PR['random']['LightGBM']
    for part in partitions(sizes):
        lab=np.zeros(8,int)
        for gi_,c in enumerate(part): lab[list(c)]=gi_
        yt=lab[y]; q=np.stack([P8[:,list(c)].sum(1) for c in part],1).argmax(1); kv.append(M.cohen_kappa_score(yt,q)); bv.append(M.balanced_accuracy_score(yt,q)); qr_=np.stack([P8r[:,list(c)].sum(1) for c in part],1).argmax(1); gv.append(M.balanced_accuracy_score(yt,qr_)-bv[-1])
    h=hier[f'grouped|LightGBM|{sn}']; hier[f'randmerge|{sn}']=dict(gap_median=float(np.median(gv)),gap_min=float(np.min(gv)),gap_rank=int((np.array(gv)<h['gap_bacc'][0]-1e-12).sum())+1,n=len(kv),kappa_median=float(np.median(kv)),kappa_max=float(np.max(kv)),kappa_rank=int((np.array(kv)>h['kappa']+1e-12).sum())+1,bacc_median=float(np.median(bv)),bacc_max=float(np.max(bv)),bacc_rank=int((np.array(bv)>h['bacc'][0]+1e-12).sum())+1)
    print('random merges',sn,hier[f'randmerge|{sn}'])
for p in SPL:
    for nm,k,mp in [('coarse3',3,{'IAB':0,'CAB':0,'IOAB':0,'MORB':1,'BABB':1,'OIB':2,'OPB':2,'CFB':2}),('coarse4',4,{'IAB':0,'CAB':0,'IOAB':0,'MORB':1,'BABB':1,'OIB':2,'OPB':2,'CFB':3})]:
        yc=np.array([mp[classes[i]] for i in y]); q=xo(p,f'{nm}_LightGBM',k).argmax(1); hier[f'{p}|direct|{k}']=dict(acc=float(np.mean(q==yc)),bacc=float(M.balanced_accuracy_score(yc,q)))
V['hier']=hier
V['lgb']={f'{p}|{c}':foldstats(p,xo(p,f'lgb_{c}')) for p in SPL for c in ['small','medium','large','slow']}
V['lgb_pooled']={f'{p}|{c}':dict(acc=float(cm_acc(CM(xo(p,f'lgb_{c}').argmax(1)))),bacc=float(cm_bacc(CM(xo(p,f'lgb_{c}').argmax(1))))) for p in SPL for c in ['small','medium','large','slow']}
for f_ in ['leak','revision']:
    if os.path.exists(f'res/{f_}.json'): V[f_]=json.load(open(f'res/{f_}.json'))
gs=pd.Series(groups).value_counts(); pet=pd.Series(refs[db=='PetDB']).value_counts().head(4)
V['data']=dict(n=len(y),counts={c:int((y==i).sum()) for i,c in enumerate(classes)},n_refs=int(len(gs)),largest_petdb={k:int(v) for k,v in pet.items()},mgo=[float(gi('MgO').min()),float(gi('MgO').max())])
json.dump(V,open('res/V.json','w'))
print('hier',{k:(round(100*v['acc'][0],1),round(100*v['bacc'][0],1),round(v['kappa'],3)) for k,v in hier.items() if k.startswith('grouped|LightGBM')})
print('dups',{k:v for k,v in DU.items() if k not in ('top_code_pairs','dedup')}); print(V['babb_noise'],V['opb_idx'],V['seedvar'],V['ci_comp'],V['data']['largest_petdb'])
