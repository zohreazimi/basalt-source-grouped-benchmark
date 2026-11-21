import sys,json; sys.path.insert(0, '.')
from ml_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'axes.edgecolor':'#555','xtick.color':'#444','ytick.color':'#444','axes.labelcolor':'#222','grid.color':'#e3e3e0','grid.linewidth':0.6,'savefig.dpi':400,'legend.frameon':False,'axes.axisbelow':True})
BLUE,ORANGE,GREY,AQUA,VIOLET='#2a78d6','#eb6834','#8a8a85','#1baf7a','#6250d6'; PAL=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#6250d6','#e34948']
d,X,y,classes,names,groups=load(sys.argv[1]); OUT='figs'; V=json.load(open('res/V.json'))
def lab(ax,t): ax.text(-0.02,1.06,t,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='right')
def fig1():
    leak=V['leak']; fig,ax=plt.subplots(1,3,figsize=(7.5,2.6),gridspec_kw={'width_ratios':[1.15,1,0.9]})
    cnt=np.bincount(y); o=np.argsort(-cnt); db=d['Database'].values; g=np.array([(db[y==c]=='GeoROC').sum() for c in range(8)]); p=cnt-g
    ax[0].barh(range(8),g[o],color=BLUE,height=0.62,label='GEOROC'); ax[0].barh(range(8),p[o],left=g[o],color=ORANGE,height=0.62,label='PetDB'); ax[0].set_yticks(range(8),[classes[i] for i in o]); ax[0].invert_yaxis(); ax[0].set_xlabel('Number of analyses'); ax[0].legend(loc='lower right',fontsize=7.5)
    for i,c in enumerate(o): ax[0].text(cnt[c]+40,i,f'{cnt[c]:,}',va='center',fontsize=7.5,color='#333')
    ax[0].set_xlim(0,3600); lab(ax[0],'a')
    gs=np.sort(pd.Series(groups).value_counts().values)[::-1]; cum=np.cumsum(gs)/gs.sum(); ax[1].plot(np.arange(1,len(gs)+1),100*cum,color=BLUE,lw=1.8); ax[1].set_xscale('log'); ax[1].set_xlabel('Reference codes, ranked by size'); ax[1].set_ylabel('Cumulative share of analyses (%)'); ax[1].grid(True,axis='y')
    k=int((gs>=50).sum()); ax[1].plot([k],[100*cum[k-1]],'o',ms=5,color=BLUE,mec='white',mew=1.2); ax[1].annotate(f'{k} codes\n({100*cum[k-1]:.0f}% of data)',(k,100*cum[k-1]),xytext=(k*1.6,100*cum[k-1]-22),fontsize=7.5,color='#333',arrowprops=dict(arrowstyle='-',color='#888',lw=0.6)); lab(ax[1],'b')
    v=[100*leak['random']['nn_same_ref'],100*leak['grouped']['nn_same_ref']]; w=[100*leak['random']['nn_same_label'],100*leak['grouped']['nn_same_label']]; xx=np.arange(2)
    ax[2].bar(xx-0.19,v,0.34,color=BLUE,label='same reference code'); ax[2].bar(xx+0.19,w,0.34,color=ORANGE,label='same tectonic setting')
    for i in range(2): ax[2].text(i-0.19,v[i]+1.5,f'{v[i]:.0f}',ha='center',fontsize=7.5); ax[2].text(i+0.19,w[i]+1.5,f'{w[i]:.0f}',ha='center',fontsize=7.5)
    ax[2].set_xticks(xx,['Random\nsplit','Source-grouped\nsplit']); ax[2].set_ylabel('Nearest training neighbour (%)'); ax[2].set_ylim(0,118); ax[2].legend(fontsize=7,loc='upper center',bbox_to_anchor=(0.5,1.22)); lab(ax[2],'c')
    fig.tight_layout(w_pad=1.6); fig.savefig(f'{OUT}/fig1.png',bbox_inches='tight'); plt.close(fig)
NICE={'KNN':'k-nearest neighbours','LDA':'Linear discriminant analysis','Extra Trees':'Extremely randomised trees','Random Forest':'Random forest','Ensemble':'Soft-voting ensemble','SVM (RBF)':'SVM (RBF kernel)','SVM (linear)':'SVM (linear kernel)','Naive Bayes':'Gaussian naive Bayes'}
def fig2():
    CIv=V['ci']; rows=[(NICE.get(m,m),v) for m,v in CIv.items() if '[24]' not in m and 'Dummy' not in m]; rows.sort(key=lambda t:t[1]['bacc'][0])
    fig,ax=plt.subplots(1,2,figsize=(7.5,3.9),sharey=True)
    for a,mt,xl in [(ax[0],'acc','Accuracy (%)'),(ax[1],'bacc','Balanced accuracy (%)')]:
        for i,(nm,v) in enumerate(rows):
            g_,r_=v[mt],v[mt+'_r']; a.plot([100*g_[0],100*r_[0]],[i,i],color='#c9c9c4',lw=2,zorder=1)
            for t,col in [(g_,ORANGE),(r_,BLUE)]: a.errorbar(100*t[0],i,xerr=[[100*(t[0]-t[1])],[100*(t[2]-t[0])]],fmt='o',ms=5,color=col,mec='white',mew=0.8,elinewidth=0.7,zorder=3)
        a.set_xlabel(xl); a.grid(True,axis='x'); a.set_yticks(range(len(rows)),[r[0] for r in rows],fontsize=7.6)
    ax[0].plot([],[],'o',color=BLUE,label='Random stratified 5-fold'); ax[0].plot([],[],'o',color=ORANGE,label='Source-grouped 5-fold'); lab(ax[0],'a'); lab(ax[1],'b'); fig.tight_layout(w_pad=1.2); fig.legend(*ax[0].get_legend_handles_labels(),fontsize=7.5,loc='lower center',bbox_to_anchor=(0.6,-0.04),ncol=2)
    fig.savefig(f'{OUT}/fig2.png',bbox_inches='tight'); plt.close(fig)
def cmplot(pairs,fname):
    fig,ax=plt.subplots(1,2,figsize=(7.5,3.6))
    for a,(key,t) in zip(ax,pairs):
        cm=np.array(V['cm'][key]); R=100*cm/cm.sum(1,keepdims=True); a.imshow(R,cmap='Blues',vmin=0,vmax=100)
        for i in range(8):
            for j in range(8):
                if R[i,j]>=0.5: a.text(j,i,f'{R[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if R[i,j]>55 else '#222')
        a.set_xticks(range(8),classes,rotation=45,ha='right',fontsize=7.5); a.set_yticks(range(8),classes,fontsize=7.5); a.set_xlabel('Predicted setting'); a.set_title(t,fontsize=8.5,loc='left'); a.tick_params(length=0)
        for s in a.spines.values(): s.set_visible(False)
    ax[0].set_ylabel('True setting'); lab(ax[0],'a'); lab(ax[1],'b'); fig.tight_layout(w_pad=1.5); fig.savefig(f'{OUT}/{fname}.png',bbox_inches='tight'); plt.close(fig)
def fig3(): cmplot([('random|LightGBM','Random split'),('grouped|LightGBM','Source-grouped split')],'fig3')
def figS2(): cmplot([('grouped|SVM (RBF)','SVM (RBF kernel), source-grouped'),('grouped|CatBoost','CatBoost, source-grouped')],'figS2')
def fig4():
    fig,ax=plt.subplots(1,2,figsize=(7.5,2.9)); bins=np.linspace(0,1,11); pr=V['pub']['random']['acc']; pg=V['pub']['grouped']['acc']; n=len(pg); hr,_=np.histogram(pr,bins); hg,_=np.histogram(pg,bins); xx=np.arange(10)
    ax[0].bar(xx-0.2,100*hr/n,0.38,color=BLUE,label='Random split'); ax[0].bar(xx+0.2,100*hg/n,0.38,color=ORANGE,label='Source-grouped split'); ax[0].set_xticks(xx,[f'{int(100*bins[i])}–{int(100*bins[i+1])}' for i in range(10)],rotation=45,ha='right',fontsize=7); ax[0].set_xlabel('Accuracy within reference code (%)'); ax[0].set_ylabel('Share of reference codes (%)'); ax[0].legend(fontsize=7.5,loc='upper left'); ax[0].grid(True,axis='y'); lab(ax[0],'a')
    for k,col,nm in [('SVM (RBF)',BLUE,'SVM (RBF kernel)'),('LightGBM',ORANGE,'LightGBM'),('CatBoost',AQUA,'CatBoost'),('Ensemble',VIOLET,'Soft-voting ensemble')]:
        rc=np.array(V['cal'][k]['risk_cov']); ax[1].plot(100*rc[:,0],100*rc[:,1],color=col,lw=1.6,label=nm)
    ax[1].set_xlabel('Samples retained, most confident first (%)'); ax[1].set_ylabel('Accuracy of retained samples (%)'); ax[1].set_ylim(80,100.5); ax[1].set_xlim(0,100); ax[1].grid(True,axis='y'); ax[1].legend(fontsize=7.5,loc='lower left'); lab(ax[1],'b')
    fig.tight_layout(w_pad=1.6); fig.savefig(f'{OUT}/fig4.png',bbox_inches='tight'); plt.close(fig)
def fig5():
    pc=pd.read_csv('res/perm_coherent.csv'); a=pc.groupby(['kind','item'])['drop'].agg(['mean','std']); el=a.loc['element'].sort_values('mean',ascending=False).head(12)
    sub=lambda s:s.replace('Fe2O3t','Fe$_2$O$_3^{t}$').replace('SiO2','SiO$_2$').replace('TiO2','TiO$_2$').replace('Al2O3','Al$_2$O$_3$').replace('K2O','K$_2$O').replace('Na2O','Na$_2$O')
    fig,ax=plt.subplots(1,2,figsize=(7.5,3.0),gridspec_kw={'width_ratios':[1,1.15]}); ek=dict(lw=0.7,capsize=1.5,ecolor='#444'); yy=np.arange(len(el))[::-1]
    ax[0].barh(yy,100*el['mean'],0.6,xerr=100*el['std'],color=BLUE,error_kw=ek); ax[0].set_yticks(yy,[sub(f) for f in el.index]); ax[0].set_xlabel('Loss of balanced accuracy (points)'); ax[0].grid(True,axis='x'); ax[0].set_xlim(left=0); lab(ax[0],'a')
    order=['all 24 concentrations','16 trace elements','8 major oxides','HFSE (TiO2, Zr, Hf, Nb, Ta)','LILE (K2O, Rb, Ba, Sr)','REE and Y']; nm=['All 24 concentrations','16 trace elements','8 major oxides','HFSE','LILE','REE and Y']; s=a.loc['block'].loc[order]; yy=np.arange(len(order))[::-1]
    ax[1].barh(yy,100*s['mean'],0.6,xerr=100*s['std'],color=BLUE,error_kw=ek); ax[1].set_yticks(yy,nm); ax[1].set_xlabel('Loss of balanced accuracy (points)'); ax[1].grid(True,axis='x'); lab(ax[1],'b')
    fig.tight_layout(w_pad=1.4); fig.savefig(f'{OUT}/fig5.png',bbox_inches='tight'); plt.close(fig)
def fig6():
    fig,ax=plt.subplots(1,2,figsize=(7.5,3.0),gridspec_kw={'width_ratios':[1,1.05]})
    sch=['8','6','4','3']; H=V['hier']
    for off,p,col,nm in [(-0.2,'random',BLUE,'Random split'),(0.2,'grouped',ORANGE,'Source-grouped split')]:
        v=[H[f'{p}|LightGBM|{s}']['bacc'] for s in sch]; ax[0].bar(np.arange(4)+off,[100*t[0] for t in v],0.38,color=col,label=nm,yerr=[[100*(t[0]-t[1]) for t in v],[100*(t[2]-t[0]) for t in v]],error_kw=dict(lw=0.7,capsize=1.5,ecolor='#444'))
    ax[0].set_xticks(range(4),['8\nsettings','6\ngroups','4\ngroups','3\ngroups']); ax[0].set_ylabel('Balanced accuracy (%)'); ax[0].set_ylim(0,105); ax[0].grid(True,axis='y'); ax[0].legend(fontsize=7.5,loc='lower center',bbox_to_anchor=(0.6,1.0),ncol=2); lab(ax[0],'a')
    sc=[('dup','Duplication (control)'),('smote','Nearest-neighbour interpolation'),('within','Within-source mixing'),('noise','Analytical noise'),('xsource','Cross-source mixing')]; AU=V['aug']
    for i,(k,nm) in enumerate(sc):
        t=AU[f'grouped|LightGBM|{k}']['d_bacc']; ax[1].errorbar(100*t[0],i,xerr=[[100*(t[0]-t[1])],[100*(t[2]-t[0])]],fmt='o',ms=5.5,color=ORANGE,mec='white',mew=0.8,elinewidth=0.9,capsize=2,ecolor='#444')
    ax[1].axvline(0,color='#555',lw=0.8); ax[1].set_yticks(range(len(sc)),[n for _,n in sc],fontsize=7.8); ax[1].set_ylim(-0.6,len(sc)-0.4); ax[1].set_xlabel('Change in grouped balanced accuracy (points)'); ax[1].grid(True,axis='x'); lab(ax[1],'b')
    fig.tight_layout(w_pad=1.5); fig.savefig(f'{OUT}/fig6.png',bbox_inches='tight'); plt.close(fig)
def figS1():
    from sklearn.decomposition import PCA; from scipy.stats import skew
    Z=PowerTransformer().fit_transform(X); pc=PCA(2).fit(Z); S=pc.transform(Z); fig,ax=plt.subplots(1,2,figsize=(7.5,3.3)); rng=np.random.default_rng(0)
    for i in rng.permutation(8): ax[0].scatter(S[y==i,0],S[y==i,1],s=3,color=PAL[i],alpha=0.45,lw=0,label=classes[i],rasterized=True)
    h,l=ax[0].get_legend_handles_labels(); o=np.argsort(l); ax[0].legend([h[i] for i in o],[l[i] for i in o],fontsize=7,markerscale=3,ncol=2,loc='upper left'); ax[0].set_xlabel(f'PC1 ({100*pc.explained_variance_ratio_[0]:.1f}%)'); ax[0].set_ylabel(f'PC2 ({100*pc.explained_variance_ratio_[1]:.1f}%)'); lab(ax[0],'a')
    sk=skew(X,axis=0); o=np.argsort(-sk); ax[1].bar(range(24),sk[o],color=BLUE,width=0.7); ax[1].set_xticks(range(24),[names[i] for i in o],rotation=90,fontsize=7); ax[1].axhline(1,color=GREY,lw=0.8,ls='--'); ax[1].set_ylabel('Skewness of raw concentrations'); lab(ax[1],'b')
    fig.tight_layout(w_pad=1.5); fig.savefig(f'{OUT}/figS1.png',bbox_inches='tight'); plt.close(fig)
if __name__=='__main__':
    for f in [fig1,fig2,fig3,fig4,fig5,fig6,figS1,figS2]: f(); print(f.__name__,'ok')
