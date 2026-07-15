"""
Threshold-FREE view: you can sidestep delta entirely for the primary metric.
Per argument, take the mean turn-1 shift (magnitude). A valid vs invalid ROC sweeps
EVERY possible threshold at once; AUC = P(a random valid arg out-shifts a random
invalid arg) = probability the model moves more for good arguments than junk.
No delta is chosen anywhere.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DIRSIGN={"lower":-1,"raise":1}
OUT="Marthe/score_stresstest/delta_threshold/results/threshold_free"

def load(path):
    df=pd.read_csv(path); df["idx0"]=df.ordering.astype(str).str.zfill(3).str[0].astype(int)
    df["d"]=df.direction.map(DIRSIGN)
    s0=df.groupby(["artefact","run"])["S0"].first().reset_index()
    sig=s0.groupby("artefact")["S0"].std(ddof=1).rename("sigma0")
    df=df.merge(sig,on="artefact"); df["Delta"]=df.d*(df.t1-df.S0)
    ap=df.groupby(["artefact","direction","validity","idx0"]).agg(
        meanDelta=("Delta","mean")).reset_index()
    return ap

def roc(v,i):
    taus=np.unique(np.concatenate([v,i,[ -1e9,1e9]]))[::-1]
    tpr=[(v>=t).mean() for t in taus]; fpr=[(i>=t).mean() for t in taus]
    auc=sum((vi>ii)+0.5*(vi==ii) for vi in v for ii in i)/(len(v)*len(i))
    return np.array(fpr),np.array(tpr),auc

MODELS={"o4-mini":("Andres/ads_inputs/trajectories/trajectories_challenge_22_o4mini.csv","#e07b39"),
        "gpt-5.5":("Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55.csv","#2c6fbb")}
C_VAL,C_INV="#2c6fbb","#d1495b"
fig,axes=plt.subplots(1,3,figsize=(15,4.7))

# --- panels 1-2: per-argument mean shift, valid vs invalid (strip + box) ---
for k,(m,(path,_)) in enumerate(MODELS.items()):
    ax=axes[k]; ap=load(path)
    V=ap[ap.validity=="valid"].meanDelta.values; I=ap[ap.validity=="invalid"].meanDelta.values
    _,_,auc=roc(V,I)
    for xpos,vals,col,lab in [(0,I,C_INV,"invalid"),(1,V,C_VAL,"valid")]:
        ax.scatter(np.full(len(vals),xpos)+np.random.uniform(-.09,.09,len(vals)),
                   vals,s=20,c=col,alpha=.6,edgecolors="none")
        ax.boxplot(vals,positions=[xpos],widths=.42,showfliers=False,
                   medianprops=dict(color="k"),boxprops=dict(color="k"),
                   whiskerprops=dict(color="k"),capprops=dict(color="k"))
    ax.set_xticks([0,1]); ax.set_xticklabels(["invalid args","valid args"])
    ax.set_ylabel("per-argument mean shift  (points)")
    ax.set_title(f"{m}\nAUC(valid > invalid) = {auc:.2f}",fontsize=11,weight="bold")
    ax.axhline(0,color="grey",lw=.6,ls=":"); ax.grid(axis="y",alpha=.3)

# --- panel 3: ROC for both models (the whole delta-sweep in one curve) ---
ax=axes[2]
for m,(path,col) in MODELS.items():
    ap=load(path)
    V=ap[ap.validity=="valid"].meanDelta.values; I=ap[ap.validity=="invalid"].meanDelta.values
    fpr,tpr,auc=roc(V,I)
    ax.plot(fpr,tpr,color=col,lw=2.3,label=f"{m}  (AUC={auc:.2f})")
ax.plot([0,1],[0,1],color="grey",ls="--",lw=1,label="chance (indiscriminate)")
ax.set_xlabel("invalid-update rate  (FPR)"); ax.set_ylabel("valid-update rate  (TPR)")
ax.set_title("ROC: every δ at once\n(each point = one δ choice)",fontsize=11,weight="bold")
ax.legend(loc="lower right",fontsize=9,frameon=False); ax.grid(alpha=.3)
ax.set_xlim(0,1); ax.set_ylim(0,1.02)

fig.suptitle("Threshold-free separability — the primary metric needs no δ",
             y=1.02,fontsize=13,weight="bold")
plt.tight_layout()
for ext in ("png","pdf"): plt.savefig(f"{OUT}.{ext}",dpi=130,bbox_inches="tight")
print("wrote",OUT+".png / .pdf")
