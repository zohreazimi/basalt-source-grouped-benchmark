"""Fig. 7 and Fig. S3: the Nb-La-Yb diagram for oceanic basalts (reads res/ternary_oceanic.json)."""
import sys,json; sys.path.insert(0, '.')
from ml_common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d,X,y,classes,names,groups=load(sys.argv[1]); L=np.log(X); g=lambda e:L[:,names.index(e)]; out=json.load(open('res/ternary_oceanic.json'))
mp={'IAB':0,'IOAB':0,'MORB':1,'OIB':2}; lab=['Oceanic arc (IAB, IOAB)','MORB','OIB']; yt=np.array([mp.get(classes[i],0) for i in y]); idx=np.where(np.array([classes[i] in mp for i in y]))[0]
F=np.column_stack([g('Nb')-g('La'),g('La')-g('Yb')]); coef=np.array(out['coef']); inter=np.array(out['intercept']); MULT=out['mult']
# ---------- figures ----------
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'savefig.dpi':400,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6})
COL=['#eb6834','#2a78d6','#1baf7a']
def bary(a,b,c): s=a+b+c; a,b,c=a/s,b/s,c/s; return c+0.5*a, np.sqrt(3)/2*a          # a top, b left, c right
def comp(e): return X[:,names.index(e)]*MULT[e]
def frame(ax):
    ax.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],color='#333',lw=0.9,zorder=4)
    for t in np.arange(0.2,1,0.2):
        for (p,q) in [((t,0),(0.5*t,np.sqrt(3)/2*t)),((1-t,0),(1-0.5*t,np.sqrt(3)/2*t)),((0.5*t,np.sqrt(3)/2*t),(1-0.5*t,np.sqrt(3)/2*t))]: ax.plot([p[0],q[0]],[p[1],q[1]],color='#d0d0cc',lw=0.5,zorder=3)
    ax.text(0.5,np.sqrt(3)/2+0.03,'2 × Nb',ha='center',fontsize=10); ax.text(-0.03,-0.04,'La',ha='right',fontsize=10); ax.text(1.03,-0.04,'3 × Yb',ha='left',fontsize=10)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.12,1.12); ax.set_ylim(-0.1,0.95)
def regions(ax):
    n=400; pts=[(i/n,j/n,1-i/n-j/n) for i in range(1,n) for j in range(1,n-i)]; a,b,c=np.array(pts).T
    x1=np.log((a/MULT['Nb'])/(b/MULT['La'])); x2=np.log((b/MULT['La'])/(c/MULT['Yb'])); z=np.argmax(np.column_stack([x1,x2])@coef.T+inter,1); xx,yy=bary(a,b,c)
    for k in range(3): ax.scatter(xx[z==k],yy[z==k],s=1.2,color=COL[k],alpha=0.10,lw=0,zorder=1,rasterized=True)
def lnpanel(ax,sets,leg=True):
    x0,x1,y0,y1=-2.6,0.9,-1.2,4.4; gx,gy=np.meshgrid(np.linspace(x0,x1,500),np.linspace(y0,y1,500)); z=np.argmax(np.column_stack([gx.ravel(),gy.ravel()])@coef.T+inter,1).reshape(gx.shape)
    ax.contourf(gx,gy,z,levels=[-.5,.5,1.5,2.5],colors=COL,alpha=0.10,zorder=0); ax.contour(gx,gy,z,levels=[.5,1.5],colors='#444',linewidths=0.8,zorder=3)
    for m,col,mk_,nm in sets: ax.scatter(F[m,0],F[m,1],s=3,color=col,alpha=0.5,lw=0,marker=mk_,zorder=2,rasterized=True,label=nm)
    ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_xlabel('ln(Nb/La)'); ax.set_ylabel('ln(La/Yb)')
    if leg: ax.legend(fontsize=7.2,frameon=False,markerscale=2.5,loc='upper left',handletextpad=0.2)
fig,ax=plt.subplots(1,2,figsize=(7.5,3.6),gridspec_kw={'width_ratios':[1.1,1]})
frame(ax[0]); regions(ax[0]); rng=np.random.default_rng(0); order=rng.permutation(idx)
xx,yy=bary(comp('Nb')[order],comp('La')[order],comp('Yb')[order]); ax[0].scatter(xx,yy,s=3,c=[COL[k] for k in yt[order]],alpha=0.55,lw=0,zorder=2,rasterized=True)
for k in range(3): ax[0].scatter([],[],s=18,color=COL[k],label=lab[k])
ax[0].legend(fontsize=7.2,loc='upper center',bbox_to_anchor=(0.5,0.02),frameon=False,handletextpad=0.2,ncol=3,columnspacing=0.8); ax[0].text(-0.1,0.88,'a',fontsize=11,fontweight='bold')
x0,x1,y0,y1=-2.6,0.9,-1.2,4.4; gx,gy=np.meshgrid(np.linspace(x0,x1,500),np.linspace(y0,y1,500)); z=np.argmax(np.column_stack([gx.ravel(),gy.ravel()])@coef.T+inter,1).reshape(gx.shape)
ax[1].contourf(gx,gy,z,levels=[-.5,.5,1.5,2.5],colors=COL,alpha=0.10,zorder=0); ax[1].contour(gx,gy,z,levels=[.5,1.5],colors='#444',linewidths=0.8,zorder=3)
ax[1].scatter(F[order,0],F[order,1],s=3,c=[COL[k] for k in yt[order]],alpha=0.5,lw=0,zorder=2,rasterized=True); ax[1].set_xlim(x0,x1); ax[1].set_ylim(y0,y1); ax[1].set_xlabel('ln(Nb/La)'); ax[1].set_ylabel('ln(La/Yb)')
for (tx,ty,t) in [(-2.5,3.9,'Oceanic arc'),(0.2,-1.0,'MORB'),(-0.8,4.05,'OIB + enriched MORB')]: ax[1].text(tx,ty,t,fontsize=8,color='#222',fontweight='bold')
ax[1].text(-0.16,1.02,'b',transform=ax[1].transAxes,fontsize=11,fontweight='bold')
fig.tight_layout(w_pad=1.0); fig.savefig('figs/fig7.png',bbox_inches='tight'); plt.close(fig)
fig,ax=plt.subplots(2,2,figsize=(7.0,6.0),sharex=True,sharey=True)
for a,c in zip(ax.ravel(),['BABB','OPB','CFB','CAB']):
    m=y==classes.index(c); lnpanel(a,[(m,'#333333','o',None)],leg=False); sh=out['excluded'][c]; a.set_title(f"{c} (n = {m.sum():,}): arc {100*sh[0]:.0f}%, MORB {100*sh[1]:.0f}%, OIB {100*sh[2]:.0f}%",fontsize=8,loc='left')
fig.tight_layout(); fig.savefig('figs/figS3.png',bbox_inches='tight'); plt.close(fig)
