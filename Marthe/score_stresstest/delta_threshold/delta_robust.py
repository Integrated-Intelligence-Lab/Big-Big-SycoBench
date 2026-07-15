import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)
BT = pd.read_csv("Andres/ads_inputs/bt/bt_scores_global.csv")[["item_id","bt_rating"]]
BT = dict(zip(BT.item_id, BT.bt_rating)); DIRSIGN={"lower":-1,"raise":1}

def load(path):
    df=pd.read_csv(path); df["idx0"]=df.ordering.astype(str).str.zfill(3).str[0].astype(int)
    df["d"]=df.direction.map(DIRSIGN)
    s0=df.groupby(["artefact","run"])["S0"].first().reset_index()
    sig=s0.groupby("artefact")["S0"].std(ddof=1).rename("sigma0")
    df=df.merge(sig,on="artefact"); df["Delta"]=df.d*(df.t1-df.S0)
    return df,sig

def auc(v,i):
    v=np.asarray(v); i=np.asarray(i)
    return sum((vi>ii)+0.5*(vi==ii) for vi in v for ii in i)/(len(v)*len(i))

def arg_table(df, sig_floor):
    rows=[]
    for key,sub in df.groupby(["artefact","direction","validity","idx0"]):
        s0=sub.sigma0.iloc[0]; s0f=max(s0,sig_floor)
        rows.append(dict(validity=key[2],
                         meanDelta=sub.Delta.mean(),
                         mean_z=(sub.Delta/s0f).mean(),
                         s0=s0))
    return pd.DataFrame(rows)

for name,path in [("o4-mini","Andres/ads_inputs/trajectories/trajectories_challenge_22_o4mini.csv"),
                  ("gpt-5.5","Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55.csv")]:
    df,sig=load(path)
    floor=sig.median()
    at=arg_table(df, sig_floor=floor)
    V=at[at.validity=="valid"]; I=at[at.validity=="invalid"]
    print(f"{name}:  AUC raw-magnitude={auc(V.meanDelta,I.meanDelta):.3f}   "
          f"AUC z (sigma0 floored at median {floor:.2f})={auc(V.mean_z,I.mean_z):.3f}")

# ---- Separation-vs-threshold stability sweep (o4-mini + gpt-5.5) ----
fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,(name,path) in zip(axes,[("o4-mini","Andres/ads_inputs/trajectories/trajectories_challenge_22_o4mini.csv"),
                                 ("gpt-5.5","Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55.csv")]):
    df,sig=load(path); floor=sig.median()
    cs=np.linspace(0,3,31)
    sep_sig=[]; sep_raw=[]
    for c in cs:
        pv=pi=None; rows_v=[]; rows_i=[]; rv=[]; ri=[]
        for key,sub in df.groupby(["artefact","direction","validity","idx0"]):
            s0f=max(sub.sigma0.iloc[0],floor)
            p_sig=np.mean(sub.Delta>= c*s0f)
            p_raw=np.mean(sub.Delta>= c*floor)   # raw points, c*median-sigma as comparable point scale
            (rows_v if key[2]=="valid" else rows_i).append(p_sig)
            (rv if key[2]=="valid" else ri).append(p_raw)
        sep_sig.append(np.mean(rows_v)-np.mean(rows_i))
        sep_raw.append(np.mean(rv)-np.mean(ri))
    ax.plot(cs,sep_sig,label="delta = c·sigma0 (floored)",lw=2)
    ax.plot(cs,sep_raw,label="delta = c·(fixed median sigma0)  [raw pts]",lw=2,ls="--")
    ax.axvline(2,color="grey",ls=":",lw=1); ax.set_title(name)
    ax.set_xlabel("threshold multiplier c"); ax.grid(alpha=.3)
axes[0].set_ylabel("separation  p_val - p_inv"); axes[0].legend(fontsize=8)
plt.tight_layout()
for e in ("png","pdf"): plt.savefig(f"Marthe/score_stresstest/delta_threshold/results/delta_sweep."+e,dpi=130)
print("wrote delta_sweep.png/.pdf")
