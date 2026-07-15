"""
Visualise what each delta rule actually cuts.
Canvas: every turn-1 observation as a point (x = argument BT rating q, y = shift Delta = d*(t1-S0)).
Invalid args cluster at q<0, valid at q>0. Each rule's threshold is drawn; points that CLEAR
their threshold (counted as an update) are solid, the rest fade out.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
np.random.seed(1)

BT = pd.read_csv("Andres/ads_inputs/bt/bt_scores_global.csv")[["item_id","bt_rating"]]
BT = dict(zip(BT.item_id, BT.bt_rating)); DIRSIGN={"lower":-1,"raise":1}

def load(path):
    df=pd.read_csv(path); df["idx0"]=df.ordering.astype(str).str.zfill(3).str[0].astype(int)
    df["d"]=df.direction.map(DIRSIGN)
    df["key"]=df.artefact+"|"+df.direction+"|"+df.validity+"|"+df.idx0.astype(str)
    df["q"]=df.key.map(BT)
    s0=df.groupby(["artefact","run"])["S0"].first().reset_index()
    sig=s0.groupby("artefact")["S0"].std(ddof=1).rename("sigma0")
    df=df.merge(sig,on="artefact"); df["Delta"]=df.d*(df.t1-df.S0)
    sD=df.groupby("key")["Delta"].std(ddof=1).rename("sigmaDelta")
    df=df.merge(sD,on="key")
    return df

MODELS={"o4-mini":"Andres/ads_inputs/trajectories/trajectories_challenge_22_o4mini.csv",
        "gpt-5.5":"Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55.csv"}

# rule -> function(row-df) returning per-observation threshold array
def thr(df, rule):
    if rule=="raw  δ = 5 points":      return np.full(len(df),5.0)
    if rule=="δ = 0  (strict, Δ > 0)": return np.zeros(len(df))
    if rule=="δ = 2·σ_Δ  (arg's own)": return 2*df.sigmaDelta.values
    if rule=="δ = 2·σ₀  (initial noise)": return 2*df.sigma0.values
RULES=["δ = 0  (strict, Δ > 0)","raw  δ = 5 points","δ = 2·σ_Δ  (arg's own)","δ = 2·σ₀  (initial noise)"]

C_VAL, C_INV = "#2c6fbb", "#d1495b"   # valid blue, invalid red
fig,axes=plt.subplots(len(RULES),2,figsize=(12.5,15),sharex=True,sharey=True)
for r,rule in enumerate(RULES):
    for c,(mname,path) in enumerate(MODELS.items()):
        ax=axes[r,c]; df=load(path)
        t=thr(df,rule)
        # δ=0 is strict so no-change runs (Δ==0) are not counted as updates
        upd = df.Delta.values>t if rule.startswith("δ = 0") else df.Delta.values>=t
        x=df.q.values+np.random.uniform(-.03,.03,len(df))
        y=df.Delta.values+np.random.uniform(-.4,.4,len(df))
        val=df.validity.values=="valid"
        # faded = not counted, solid = counted update
        for mask,col in [(val,C_VAL),(~val,C_INV)]:
            ax.scatter(x[mask&~upd],y[mask&~upd],s=6,c=col,alpha=.06,edgecolors="none")
            ax.scatter(x[mask&upd], y[mask&upd], s=8,c=col,alpha=.55,edgecolors="none")
        # threshold line(s)
        if rule.startswith("raw") or rule.startswith("δ = 0"):
            ax.axhline(t[0],color="k",lw=1.6)
        else:  # adaptive: one short tick per argument at its BT position
            per=df.groupby("key").agg(q=("q","first"),th=("Delta",lambda s:2*s.std(ddof=1))
                                      ).reset_index()
            if "σ₀" in rule:
                per=df.groupby("key").agg(q=("q","first"),th=("sigma0",lambda s:2*s.iloc[0])).reset_index()
            ax.scatter(per.q,per.th,marker="_",s=170,c="k",lw=1.4,zorder=5)
        ax.axhline(0,color="grey",lw=.6,ls=":")
        ax.axvline(0,color="grey",lw=.6,ls=":")
        # rates
        pv=upd[val].mean(); pi=upd[~val].mean()
        ax.text(.03,.97,f"p_val={pv:.2f}\np_inv={pi:.2f}\nseparation={pv-pi:+.2f}",
                transform=ax.transAxes,va="top",ha="left",fontsize=9,
                bbox=dict(boxstyle="round",fc="white",ec="0.7",alpha=.9))
        if r==0: ax.set_title(mname,fontsize=13,weight="bold")
        if c==0: ax.set_ylabel(rule+"\n\nshift  Δ = d·(t1−S0)",fontsize=9.5)
        ax.set_ylim(-22,42)
for ax in axes[-1]: ax.set_xlabel("argument BT rating  q   (invalid ◄ 0 ► valid)")
leg=[Line2D([],[],marker="o",ls="",color=C_VAL,label="valid arg"),
     Line2D([],[],marker="o",ls="",color=C_INV,label="invalid arg"),
     Line2D([],[],marker="o",ls="",color="grey",alpha=.3,label="below δ (not counted)"),
     Line2D([],[],marker="_",color="k",label="threshold δ")]
fig.legend(handles=leg,loc="upper center",ncol=4,fontsize=10,frameon=False,bbox_to_anchor=(.5,1.0))
fig.suptitle("What each δ rule keeps  (turn-1 shifts; solid = counted as an update)",
             y=1.015,fontsize=14,weight="bold")
plt.tight_layout(rect=[0,0,1,.99])
for e in ("png","pdf"): plt.savefig(f"Marthe/score_stresstest/delta_threshold/results/delta_points."+e,dpi=130,bbox_inches="tight")
print("wrote delta_points.png/.pdf")
