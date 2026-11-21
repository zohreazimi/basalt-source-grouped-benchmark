"""Near-duplicate analyses: all 16 trace elements within 1% (relative) of another analysis."""
import sys,json; sys.path.insert(0, '.')
from ml_common import *
d,X,y,classes,names,groups=load(sys.argv[1]); T=np.log(X[:,8:]); db=d['Database'].values; refs=d['References'].astype(str).values; sid=d['Sample ID'].astype(str).values
nn=NearestNeighbors(radius=np.log(1.01)*4,metric='euclidean').fit(T)   # candidate radius; exact test below
_,ind=nn.radius_neighbors(T); pairs=[]
for i,nb in enumerate(ind):
    for j in nb:
        if j>i and np.all(np.abs(T[i]-T[j])<=np.log(1.01)): pairs.append((i,j))
pairs=np.array(pairs); print('pairs',len(pairs))
diffcode=groups[pairs[:,0]]!=groups[pairs[:,1]]; crossdb=db[pairs[:,0]]!=db[pairs[:,1]]; difflab=y[pairs[:,0]]!=y[pairs[:,1]]
maj=np.max(np.abs(X[pairs[:,0],:8]-X[pairs[:,1],:8]),1)
out=dict(n_pairs=int(len(pairs)),pairs_diff_code=int(diffcode.sum()),pairs_cross_db=int(crossdb.sum()),pairs_diff_label=int(difflab.sum()),samples_in_pairs=int(len(np.unique(pairs))),samples_in_diffcode_pairs=int(len(np.unique(pairs[diffcode]))),median_major_diff=float(np.median(maj[diffcode])) if diffcode.any() else None)
lab=pd.Series([f'{classes[y[i]]}|{classes[y[j]]}' for i,j in pairs[difflab]]).value_counts().to_dict(); out['label_pairs']=lab
cp=pd.Series([f'{refs[i]} <-> {refs[j]}' for i,j in pairs[diffcode]]).value_counts().head(12).to_dict(); out['top_code_pairs']=cp
print(json.dumps(out,indent=1)); 
for i,j in pairs[difflab][:4]: print(sid[i],refs[i],classes[y[i]],'|',sid[j],refs[j],classes[y[j]])
for i,j in pairs[diffcode&~difflab][:4]: print(sid[i],refs[i],classes[y[i]],'|',sid[j],refs[j],classes[y[j]])
# how many grouped-test samples have a duplicate in their training fold
dup_of={}; 
for i,j in pairs: dup_of.setdefault(i,[]).append(j); dup_of.setdefault(j,[]).append(i)
cnt=0
for tr,te in make_splits(y,groups,'grouped'):
    S=set(tr); cnt+=sum(any(k in S for k in dup_of.get(i,[])) for i in te)
out['grouped_test_with_dup_in_train']=int(cnt); print('grouped test samples with duplicate in train',cnt)
np.save('res/dup_pairs.npy',pairs); json.dump(out,open('res/dups.json','w'),indent=1)
