import sys; sys.path.insert(0, '.')
from ml_common import *
d,X,y,classes,names,groups=load(sys.argv[1])
out={}
# duplicated sample IDs
dup=d[d.duplicated('Sample ID',keep=False)]
out['n_dup_id_rows']=int(len(dup)); out['n_dup_ids']=int(dup['Sample ID'].nunique())
g=dup.groupby('Sample ID'); out['dup_ids_diff_class']=int((g['Tectonic Setting'].nunique()>1).sum()); out['dup_ids_diff_ref']=int((g['References'].nunique()>1).sum())
gs=pd.Series(groups).value_counts()
out['n_refs']=int(len(gs)); out['median_per_ref']=float(gs.median()); out['max_per_ref']=int(gs.max()); out['refs_ge50']=int((gs>=50).sum()); out['share_in_refs_ge50']=float(gs[gs>=50].sum()/len(y))
out['refs_multi_class']=int((pd.Series(y).groupby(groups).nunique()>1).sum())
for prot in ['random','grouped']:
    same=[];lab=[];has=[]
    for f,(tr,te) in enumerate(make_splits(y,groups,prot)):
        Z=np.load(f'cache/{prot}_{f}.npz')['Z44']
        nn=NearestNeighbors(n_neighbors=1).fit(Z[tr]); dist,ind=nn.kneighbors(Z[te]); j=tr[ind[:,0]]
        same.append((groups[j]==groups[te]).mean()); lab.append((y[j]==y[te]).mean()); has.append(np.isin(groups[te],groups[tr]).mean())
    out[prot]=dict(nn_same_ref=float(np.mean(same)),nn_same_label=float(np.mean(lab)),test_with_ref_in_train=float(np.mean(has)))
json.dump(out,open('res/leak.json','w'),indent=1); print(json.dumps(out,indent=1))
