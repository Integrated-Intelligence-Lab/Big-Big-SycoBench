"""
Companion 'why' figure: distribution of turn-1 shifts for valid / invalid arguments,
with the NULL placebo (no argument: S0^r - S0^r') overlaid. Each delta rule is a vertical
line; the false-update floor is the grey null mass to the RIGHT of the line.
Shows: (i) null is centred at 0 -> any one-sided delta keeps a positive floor;
       (ii) a fixed raw delta sits at a different place on each model's null width.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(2)
DIRSIGN={"lower":-1,"raise":1}
def load(path):
    df=pd.read_csv(path); df["idx0"]=df.ordering.astype(str).str.zfill(3).str[0].astype(int)
    df["d"]=df.direction.map(DIRSIGN)
    s0=df.groupby(["artefact","run"])["S0"].first().reset_index()
    sig=s0.groupby("artefact")["S0"].std(ddof=1).rename("sigma0")
    df=df.merge(sig,on="artefact"); df["Delta"]=df.d*(df.t1-df.S0); return df,sig
def null_shifts(df,n=40000):
    out=[]
    for art,sub in df.groupby("artefact"):
        v=sub.groupby("run")["S0"].first().values; d=sub.d.iloc[0]
        out.append(d*(np.random.choice(v,n)-np.random.choice(v,n)))
    return np.concatenate(out)

MODELS={"o4-mini":"Andres/ads_inputs/trajectories/trajectories_challenge_22_o4mini.csv",
        "gpt-5.5":"Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55.csv"}
fig,axes=plt.subplots(2,1,figsize=(10,9),sharex=True)
bins=np.arange(-20,45,1.5)
for ax,(m,path) in zip(axes,MODELS.items()):
    df,sig=load(path); nl=null_shifts(df); s0med=sig.median()
    V=df[df.validity=="valid"].Delta; I=df[df.validity=="invalid"].Delta
    ax.hist(nl,bins=bins,density=True,color="0.6",alpha=.55,label="null (no argument)")
    ax.hist(I,bins=bins,density=True,histtype="step",lw=2,color="#d1495b",label="invalid args")
    ax.hist(V,bins=bins,density=True,histtype="step",lw=2,color="#2c6fbb",label="valid args")
    floor=lambda d:(nl>=d).mean()
    lines=[("δ=0",0,"-"),("raw δ=5",5,"-"),("raw δ=10",10,":"),(f"2·σ₀ (={2*s0med:.1f})",2*s0med,"--")]
    for name,d,ls in lines:
        ax.axvline(d,color="k",lw=1.3,ls=ls)
        ax.text(d+.3,ax.get_ylim()[1]*0 + .008*(1),"",fontsize=8)
    # annotate floors
    txt="null false-update floor:\n"+"\n".join(
        f"  {name.split(' (')[0]:9s}: {floor(d)*100:4.1f}%" for name,d,ls in lines)
    ax.text(.985,.97,txt,transform=ax.transAxes,va="top",ha="right",fontsize=9,family="monospace",
            bbox=dict(boxstyle="round",fc="white",ec="0.7"))
    # shade null mass beyond raw delta=5
    xs=bins[:-1]+.75; mask=xs>=5
    ax.fill_between(bins,0,0)  # noop keep api
    ax.set_title(f"{m}   (σ₀ median = {s0med:.2f})",loc="left",fontsize=12,weight="bold")
    ax.set_ylabel("density"); ax.legend(loc="upper center",fontsize=9,frameon=False,ncol=3)
    ax.set_xlim(-20,44)
    for name,d,ls in lines:
        ax.text(d,-0.004,name,rotation=90,ha="center",va="top",fontsize=7.5)
axes[-1].set_xlabel("turn-1 shift  Δ = d·(t1 − S0)   (points)")
fig.suptitle("Why the units matter: same rules, two models — floor is the null mass right of each line",
             y=.995,fontsize=12.5,weight="bold")
plt.tight_layout(rect=[0,0,1,.98])
for e in ("png","pdf"): plt.savefig(f"Marthe/score_stresstest/delta_threshold/results/delta_dist."+e,dpi=130,bbox_inches="tight")
print("wrote delta_dist.png/.pdf")
