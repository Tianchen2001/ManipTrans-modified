"""Summarise eval_multiclip.sh: AUC and physical metrics of the multi-clip policies, per clip / run / setting.

    python sweep_multiclip.py                       # every eval/multiclip_ep*/  (all checkpoints, all sets)
    python sweep_multiclip.py --ckpts 3000 --sets test

Same definitions as sweep_scale.py (single hand): per env  s* = max_{t >= grace} max_k dist_k(t) / (base_k / 0.7)
(inf on a numerical blow-up), SR(s) = fraction of envs with s* <= s, AUC = area under SR over log10 s in [smin, smax],
normalised; x_* = per-env mean over the whole (non-terminated) rollout, then mean over the envs of a clip.
Env e of an evaluation tracks clip e % n_clips (order in <exp>__<set>_clips.txt).

Writes eval/summary/multiclip_clips.csv   (method, n, seed, hand, ckpt, set, clip -> metrics)
       eval/summary/multiclip_runs.csv    (mean over the clips of a set, per run)
       eval/summary/multiclip_summary.csv (mean / std over seeds per method, n, ckpt, set, + paired ours - dd)
       eval/summary/multiclip_<hand>_<set>_ep<ckpt>.png  (AUC, P_wrist vs training-set size)
       eval/summary/multiclip_<hand>_<set>_n<N>_ckpt.png  (AUC, P_wrist vs checkpoint, if several were evaluated)
"""
import argparse, glob, os, re
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="eval")
ap.add_argument("--ckpts", nargs="+", type=int, default=None)
ap.add_argument("--sets", nargs="+", default=None)
ap.add_argument("--smin", type=float, default=0.1)
ap.add_argument("--smax", type=float, default=10.0)
ap.add_argument("--n-grid", type=int, default=401)
ap.add_argument("--grace", type=int, default=20)
ap.add_argument("--no-plot", action="store_true")
a = ap.parse_args()

NAME = re.compile(r"g_(?P<method>[a-z]+)_n(?P<n>\d+)_s(?P<seed>\d+)_(?P<hand>[a-z]+)(?P<tag>\S*?)__(?P<set>[a-z]+)$")
METRICS = ["sr_auc", "x_power_wrist_W", "x_power_joint_W", "x_err_tip_mm", "x_err_kp_mm", "x_jerk_wrist",
           "frac_blowup"]
grid = np.logspace(np.log10(a.smin), np.log10(a.smax), a.n_grid)
lg = np.log10(grid)


def auc_of(s_star):
    sr = (s_star[None, :] <= grid[:, None]).mean(1)
    return float(((sr[1:] + sr[:-1]) / 2 * np.diff(lg)).sum() / (lg[-1] - lg[0]))


rows = []
for p in sorted(glob.glob(f"{a.root}/multiclip_ep*/*_frames.npz")):
    ck = int(re.search(r"multiclip_ep(\d+)", p).group(1))
    stem = p[: -len("_frames.npz")]
    m = NAME.match(os.path.basename(stem))
    if not m or (a.ckpts and ck not in a.ckpts) or (a.sets and m["set"] not in a.sets):
        continue
    clips = [c.strip() for c in open(stem + "_clips.txt") if c.strip()]
    D = np.load(p)
    alive = D["alive"].astype(bool)                                    # [T, n_env]
    valid = alive & (D["run"] >= a.grace)
    R = np.where(valid, D["ratio"].max(-1), 0.0)
    blowup = ~D["success"].astype(bool)
    s_star = np.where(blowup, np.inf, R.max(0))
    env_clip = np.arange(alive.shape[1]) % len(clips)
    per_env = {}
    for k in D.files:
        if k.startswith("m_"):
            v = D[k].astype(np.float64)
            ok = alive & ~np.isnan(v)
            per_env["x_" + k[2:]] = np.where(ok, np.nan_to_num(v), 0).sum(0) / np.maximum(ok.sum(0), 1)
    g = m.groupdict()
    for ci, c in enumerate(clips):
        e = env_clip == ci
        rows.append({"method": g["method"], "n": int(g["n"]), "seed": int(g["seed"]), "hand": g["hand"],
                     "tag": g["tag"], "ckpt": ck, "set": g["set"], "clip": c, "envs": int(e.sum()),
                     "sr_auc": auc_of(s_star[e]), "frac_blowup": float(blowup[e].mean()),
                     "s_star_median": float(np.median(s_star[e])),
                     **{k: float(v[e].mean()) for k, v in per_env.items()}})

if not rows:
    raise SystemExit(f"no {a.root}/multiclip_ep*/*_frames.npz - run eval_multiclip.sh first")
out = os.path.join(a.root, "summary")
os.makedirs(out, exist_ok=True)
C = pd.DataFrame(rows)
C = C[C["tag"] == ""] if (C["tag"] == "").any() else C                 # tuning / smoke runs only if nothing else
C.to_csv(f"{out}/multiclip_clips.csv", index=False)
mets = [k for k in METRICS if k in C]
key = ["hand", "set", "ckpt", "n", "method", "seed"]
Rn = C.groupby(key)[mets].mean().reset_index()
Rn["clips"] = C.groupby(key)["clip"].nunique().values
Rn.to_csv(f"{out}/multiclip_runs.csv", index=False)

# mean / std over seeds; paired difference ours - dd over the seeds both have
S = Rn.groupby(["hand", "set", "ckpt", "n", "method"])[mets].agg(["mean", "std", "count"])
S.columns = [f"{m}_{s}" for m, s in S.columns]
S = S.reset_index()
P = Rn.pivot_table(index=["hand", "set", "ckpt", "n", "seed"], columns="method", values=mets)
diff = []
if {"dd", "ours"} <= set(Rn["method"]):
    for (hand, st, ck, n), g in P.groupby(level=[0, 1, 2, 3]):
        d = {"hand": hand, "set": st, "ckpt": ck, "n": n, "method": "ours-dd"}
        for k in mets:
            x = (g[(k, "ours")] - g[(k, "dd")]).dropna()
            d[f"{k}_mean"], d[f"{k}_std"], d[f"{k}_count"] = x.mean(), x.std(), len(x)
        # per clip: in how many test clips is ours better (seed-averaged)
        cc = C[(C["hand"] == hand) & (C["set"] == st) & (C["ckpt"] == ck) & (C["n"] == n)]
        q = cc.groupby(["clip", "method"])[["sr_auc", "x_power_wrist_W"]].mean().unstack("method")
        if ("sr_auc", "dd") in q and ("sr_auc", "ours") in q:
            d["clips_auc_ours_better"] = f'{int((q[("sr_auc", "ours")] > q[("sr_auc", "dd")]).sum())}/{len(q)}'
            d["clips_pwrist_ours_lower"] = \
                f'{int((q[("x_power_wrist_W", "ours")] < q[("x_power_wrist_W", "dd")]).sum())}/{len(q)}'
        diff.append(d)
S = pd.concat([S, pd.DataFrame(diff)], ignore_index=True).sort_values(["hand", "set", "ckpt", "n", "method"])
S.to_csv(f"{out}/multiclip_summary.csv", index=False)

pd.set_option("display.width", 200)
for (hand, st), g in S.groupby(["hand", "set"]):
    print(f"\n=== {hand} / {st}: mean +- std over seeds (count = seeds) ===")
    t = g[["ckpt", "n", "method"]].copy()
    for k in ("sr_auc", "x_power_wrist_W", "x_err_tip_mm", "x_power_joint_W"):
        if f"{k}_mean" in g:
            t[k] = [f"{m:.3f} +- {s:.3f} ({c:.0f})" if pd.notna(s) else f"{m:.3f} ({c:.0f})"
                    for m, s, c in zip(g[f"{k}_mean"], g[f"{k}_std"], g[f"{k}_count"])]
    for k in ("clips_auc_ours_better", "clips_pwrist_ours_lower"):
        if k in g:
            t[k] = g[k].fillna("")
    print(t.to_string(index=False))
print(f"\nwrote {out}/multiclip_clips.csv, multiclip_runs.csv, multiclip_summary.csv")

if not a.no_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"dd": "#6FD1D7", "ours": "#FF4081"}
    LAB = {"dd": "Geo", "ours": "Ours"}
    PANELS = [("sr_auc", "AUC (higher is better)"), ("x_power_wrist_W", "P_wrist (W, lower is better)")]

    def plot(sub, x, xlabel, fname, logx=False):
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
        for ax, (k, yl) in zip(axs, PANELS):
            for meth, g in sub.groupby("method"):
                if meth not in COL:
                    continue
                g = g.sort_values(x)
                ax.errorbar(g[x], g[f"{k}_mean"], yerr=g[f"{k}_std"].fillna(0), color=COL[meth], marker="o",
                            lw=1.8, capsize=3, label=LAB[meth])
            ax.set_xlabel(xlabel); ax.set_ylabel(yl)
            ax.set_xticks(sorted(sub[x].unique()))
            if logx:
                ax.set_xscale("log"); ax.set_xticks(sorted(sub[x].unique())); ax.get_xaxis().set_major_formatter(
                    matplotlib.ticker.ScalarFormatter())
        axs[0].legend(frameon=False)
        fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
        print("plot", fname)

    M = S[S["method"].isin(COL)]
    for (hand, st, ck), sub in M.groupby(["hand", "set", "ckpt"]):
        if sub["n"].nunique() > 1:
            plot(sub, "n", "training clips", f"{out}/multiclip_{hand}_{st}_ep{ck}.png", logx=True)
    for (hand, st, n), sub in M.groupby(["hand", "set", "n"]):
        if sub["ckpt"].nunique() > 1:
            plot(sub, "ckpt", f"training iterations (n={n})", f"{out}/multiclip_{hand}_{st}_n{n}_ckpt.png")
