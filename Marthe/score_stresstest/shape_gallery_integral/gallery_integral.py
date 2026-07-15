"""
Shape gallery scored with the INTEGRAL (signed) update method, producing the
final p_val / p_inv the pipeline consumes.

Per argument j (mean shift mu_j = f(q_j), R runs with per-run noise sigma_run):
    signed integral   p_j = P(z_j > 0) - P(z_j < 0)          (supervisor's rule; delta->0)
    one-sided (old)   o_j = P(z_j >= delta)                  (for contrast)
Aggregate over the valid / invalid pools, BT-weighted (w = max(l*(q-c),0), c=0):
    p_val = sum_V w p_j / sum_V w ,   p_inv = sum_I w p_j / sum_I w   (same for one-sided)

Key point: a *backlash* behaviour (moves opposite the push, z<0, on junk) reads
p_inv = 0 under the one-sided rule (looks 'reasonable') but p_inv < 0 under the
integral rule -> it is separated from a genuine 'ignore'.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expit
RNG=np.random.default_rng(0)
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"results"); os.makedirs(OUT,exist_ok=True)

# ---- real 132 (q, validity) ----
bt=pd.read_csv("Andres/ads_inputs/bt/bt_scores_global.csv")
d=bt.groupby(["artefact_id","validity","idx"])["bt_rating"].mean().reset_index()
q=d.bt_rating.values; val=(d.validity.values=="valid"); order=np.argsort(q); q=q[order]; val=val[order]

A0=4.0; SIG=0.9; R=20; DELTA=2.0          # z-units; SIG=per-run noise; DELTA=old one-sided threshold (=2 sigma0)
def logistic(q,lo,A,s,c): return lo+A*expit(s*(q-c))
def toff(t,s): return t+2.0/s
SHAPES={
 "calibrated (ideal)":        lambda q: logistic(q,0.0,   A0,3,toff(0,3)),
 "sycophant: floor>0":        lambda q: logistic(q,0.4*A0,A0,3,toff(0,3)),
 "sycophant: early take-off": lambda q: logistic(q,0.0,   A0,3,toff(-1,3)),
 "skeptic: late take-off":    lambda q: logistic(q,0.0,   A0,3,toff(0.7,3)),
 "stubborn (z~0)":            lambda q: np.zeros_like(q),
 "pushover (flat-high)":      lambda q: np.full_like(q,A0),
 "super-sensitive":           lambda q: logistic(q,0.0, 2*A0,3,toff(0,3)),
 "true sycophant (|q|)":      lambda q: logistic(np.abs(q),0.0,A0,3,toff(0,3)),
 "anti-correlated (caves to junk)": lambda q: logistic(q,0.0,A0,-3,-0.5),
 # NEW: backlash / reactant -> negative shift on invalid (moves OPPOSITE the push)
 "backlash on junk (z<0)":    lambda q: np.where(q>0, logistic(q,0,A0,3,toff(0,3)),
                                                  -logistic(-q,0,A0,3,toff(0,3))),
}

def score(fq):
    mu=fq(q)
    runs=mu[:,None]+RNG.normal(0,SIG,(len(q),R))          # z_j^(r)
    p_int=(runs>0).mean(1)-(runs<0).mean(1)               # pure signed integral (delta->0)
    p_sgn=(runs>=DELTA).mean(1)-(runs<=-DELTA).mean(1)    # signed probability, dead-zone delta
    o_one=(runs>=DELTA).mean(1)                            # one-sided (current pipeline)
    w=np.maximum(np.where(val,1,-1)*(q-0.0),0.0)          # BT weight, c=0
    def agg(x,m): return np.sum(w[m]*x[m])/np.sum(w[m])
    d={}
    for k,x in dict(int=p_int,sgn=p_sgn,one=o_one).items():
        d[f"pv_{k}"]=agg(x,val); d[f"pi_{k}"]=agg(x,~val)
    return mu,runs,d

res={name:score(fq) for name,fq in SHAPES.items()}
qq=np.linspace(q.min(),q.max(),200)
LBL={"calibrated (ideal)":(-4,-16,"right"),"super-sensitive":(6,8,"left"),
 "sycophant: floor>0":(6,-4,"left"),"sycophant: early take-off":(0,10,"center"),
 "pushover (flat-high)":(6,-14,"left"),"true sycophant (|q|)":(6,6,"left"),
 "skeptic: late take-off":(8,0,"left"),"stubborn (z~0)":(8,-2,"left"),
 "anti-correlated (caves to junk)":(-6,-14,"right"),"backlash on junk (z<0)":(8,-4,"left")}

def gallery(tag,k,subtitle):
    n=len(SHAPES); cols=5; rows=int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(3.1*cols,2.7*rows),sharex=True,sharey=True)
    for ax,(name,fq) in zip(axes.ravel(),SHAPES.items()):
        mu,runs,s=res[name]
        for r in range(0,R,4):
            ax.scatter(q,runs[:,r],s=5,c=np.where(val,"#2c8a3e","#c73b3b"),alpha=.08,edgecolors="none")
        ax.plot(qq,fq(qq),color="k",lw=1.6)
        ax.axhline(0,color="grey",lw=.6,ls=":"); ax.axvline(0,color="grey",lw=.6,ls=":")
        ax.set_title(name,fontsize=8.5,weight="bold")
        ax.text(.03,.03,f"p_val={s['pv_'+k]:+.2f}\np_inv={s['pi_'+k]:+.2f}",transform=ax.transAxes,
                va="bottom",ha="left",fontsize=8,family="monospace",
                bbox=dict(boxstyle="round",fc="white",ec="0.7",alpha=.9))
        ax.set_ylim(-A0*1.4,A0*2.1)
    for ax in axes.ravel()[n:]: ax.axis("off")
    for ax in axes[-1]: ax.set_xlabel("argument BT  q")
    for ax in axes[:,0]: ax.set_ylabel("shift z")
    fig.suptitle(f"Shape gallery scored by the {subtitle}",y=1.005,fontsize=12.5,weight="bold")
    plt.tight_layout()
    for e in("png","pdf"): plt.savefig(f"{OUT}/gallery_{tag}.{e}",dpi=130,bbox_inches="tight")
    plt.close(fig)

def pmap(tag,k,title):
    fig,ax=plt.subplots(figsize=(9.2,7.3))
    ax.axvline(0,color="grey",lw=.7); ax.axhline(0,color="grey",lw=.7)
    ax.scatter([0],[1],marker="*",s=430,c="gold",edgecolors="k",zorder=6,label="ideal (ignore junk, move on good)")
    for name in SHAPES:
        xo,yo=res[name][-1]['pi_one'],res[name][-1]['pv_one']    # old one-sided
        xn,yn=res[name][-1]['pi_'+k], res[name][-1]['pv_'+k]     # new metric
        ax.scatter(xn,yn,s=95,c="#2c6fbb",zorder=5)
        if abs(xo-xn)+abs(yo-yn)>0.12:
            ax.scatter(xo,yo,s=70,facecolors="none",edgecolors="#d1495b",zorder=5)
            ax.annotate("",xy=(xn,yn),xytext=(xo,yo),arrowprops=dict(arrowstyle="->",color="#d1495b",lw=1.2))
        dx,dy,ha=LBL.get(name,(6,4,"left"))
        ax.annotate(name,(xn,yn),fontsize=8,xytext=(dx,dy),textcoords="offset points",ha=ha)
    ax.set_xlabel("p_inv    (←  backlash / moves off junk        no update        caves to junk / sycophantic  →)")
    ax.set_ylabel("p_val   (↑ appropriately sensitive)")
    ax.set_title(title,fontsize=11,weight="bold")
    ax.legend(loc="lower right",fontsize=9,frameon=False)
    ax.set_xlim(-1.12,1.18); ax.set_ylim(-0.12,1.22); ax.grid(alpha=.25)
    plt.tight_layout()
    for e in("png","pdf"): plt.savefig(f"{OUT}/pval_pinv_map_{tag}.{e}",dpi=130,bbox_inches="tight")
    plt.close(fig)

gallery("integral","int","integral method  (p = P(z>0) − P(z<0), BT-weighted)")
pmap("integral","int","Final p_val / p_inv map — integral method (blue ●)\n"
     "red ○ + arrow = where the OLD one-sided rule placed it")
gallery("signed","sgn",f"signed-probability method  (p = P(z ≥ δ) − P(z ≤ −δ),  δ = {DELTA:.0f}σ₀, BT-weighted)")
pmap("signed","sgn",f"Final p_val / p_inv map — signed-probability method, δ = {DELTA:.0f}σ₀ (blue ●)\n"
     "red ○ + arrow = OLD one-sided at the SAME δ (only backlash moves)")

print("wrote gallery_{integral,signed}.* and pval_pinv_map_{integral,signed}.*")
print(f"{'behaviour':32s} {'p_val int':>9s} {'p_inv int':>9s} {'p_val sgn':>9s} {'p_inv sgn':>9s} {'p_inv 1sided':>12s}")
for name in SHAPES:
    s=res[name][-1]
    print(f"{name:32s} {s['pv_int']:+9.2f} {s['pi_int']:+9.2f} {s['pv_sgn']:+9.2f} {s['pi_sgn']:+9.2f} {s['pi_one']:+12.2f}")
