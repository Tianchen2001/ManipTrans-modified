"""
Offline termination-threshold sweep from the per-frame dumps of `SWEEP=1 bash eval_all.sh`.

    python sweep_scale.py                              # all eval/sweep_ep1000/*_frames.npz
    python sweep_scale.py --match _traj --hands artimano inspire
    python sweep_scale.py --clips traj3 traj4 seq3 seq4 --out-suffix _new   # -> sweep_summary_<ckpt>_new.csv
    python sweep_scale.py --metrics-scale 1.0          # (default) also recompute every x_* metric at this scale

ManipTrans terminates an env when, after 20 steps, any of 7 keypoint-group distances exceeds
base_k / 0.7 * scale  (thumb/index/middle/pinky/ring tip, level-1, level-2 joints), or on a numerical blow-up
(scale-independent). The policy never sees the scale, so ONE rollout without termination contains the rollout
for every scale up to its first threshold crossing:
  r(t)   = max_k dist_k(t) / (base_k / 0.7)        (two-hand: max over both hands)
  s*     = max_{t >= 20} r(t)  per env              (= inf if the env blew up numerically)
  SR(s)  = fraction of envs with s* <= s
  AUC    = area under SR(s) over log10(s) in [--smin, --smax], normalised to [0, 1]

Per run it writes eval/sweep_<ckpt>/<exp>_scale.json (the evaluator itself writes eval/sweep_<ckpt>/<exp>.json = the
no-termination rollout, whose success_rate only counts numerical blow-ups):
  sr_auc, s_star_median, s_star_p10/p90, frac_blowup, sr_at_<s> for a few s,
  fail_trigger_at_1 (which criterion ends the episode at scale 1, counts), fail_frame_at_1_median,
  x_* metrics recomputed at --metrics-scale (same definitions as the normal evaluator; check against
  eval/sc1.0_ep1000/<exp>.json when --metrics-scale 1.0)
and eval/summary/sweep_summary_<ckpt>.csv + sweep_curves_<ckpt>.npz (SR(s) per run) + a plot per hand/task
(eval/summary/sweep_<hand>_<single|bih>_<task>_<ckpt>.png).
"""
import argparse, glob, json, os, re
import numpy as np

KEYS = ["thumb_tip", "index_tip", "middle_tip", "pinky_tip", "ring_tip", "level_1", "level_2"]
NAME = re.compile(r"h_(?P<method>[a-z]+)_(?P<clip>(?P<task>[a-z]*)\d+)_(?P<lab>[a-z0-9]+)"
                  r"(?:_(?P<side>rh|lh|bih))?_(?P<hand>[a-z]+)(?P<pid>_pid)?(?P<tag>_w[0-9.]+)?$")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ep1000")
ap.add_argument("--smin", type=float, default=0.1)
ap.add_argument("--smax", type=float, default=10.0)
ap.add_argument("--n-grid", type=int, default=401)
ap.add_argument("--metrics-scale", type=float, default=1.0)
ap.add_argument("--match", nargs="+", default=None, help="keep runs whose name contains ANY of these substrings")
ap.add_argument("--clips", nargs="+", default=None, help="keep only these clips, e.g. --clips traj3 traj4 seq3 seq4")
ap.add_argument("--out-suffix", default=None,
                help="suffix of the summary/curves/plot files (default: none for all runs, _subset with --match/--clips)")
ap.add_argument("--hands", nargs="+", default=None)
ap.add_argument("--grace", type=int, default=20,
                help="steps after reset that are not checked (ManipTrans: 20); larger = ignore the start-up transient")
ap.add_argument("--no-plot", action="store_true")
ap.add_argument("--print", nargs="+", default=None, help="metrics to print as tables (default: AUC, s*, SR, errors, power, contact)")
a = ap.parse_args()
GRACE = a.grace
SUF = a.out_suffix if a.out_suffix is not None else ("_subset" if (a.match or a.clips) else "")

grid = np.logspace(np.log10(a.smin), np.log10(a.smax), a.n_grid)
lg = np.log10(grid)
rows, curves = [], {}

SD, OUT = f"eval/sweep_{a.ckpt}", "eval/summary"
os.makedirs(OUT, exist_ok=True)
for p in sorted(glob.glob(f"{SD}/*_frames.npz")):
    exp = os.path.basename(p)[: -len("_frames.npz")]
    m = NAME.match(exp)
    if not m or (a.match and not any(x in exp for x in a.match)) or (a.hands and m["hand"] not in a.hands) \
            or (a.clips and m["clip"] not in a.clips):
        continue
    g = m.groupdict(); g["side"] = g["side"] or "-"; g["pid"] = bool(g["pid"]); g["tag"] = g["tag"] or ""
    if g["tag"] and not a.match:
        continue                                          # tuning runs only with --match
    D = np.load(p)
    alive = D["alive"].astype(bool)                       # [T, n]
    valid = alive & (D["run"] >= GRACE)
    bih = "ratio_rh" in D
    if bih:
        ratio = np.stack([D["ratio_rh"], D["ratio_lh"]], 2)  # [T, n, 2, 7]
        R_side = ratio.max(-1)                                # [T, n, 2]
        R = R_side.max(-1)
    else:
        ratio = D["ratio"]                                    # [T, n, 7]
        R = ratio.max(-1)
    R = np.where(valid, R, 0.0)
    blowup = ~D["success"].astype(bool)                   # without termination only numerical blow-ups fail
    s_star = np.where(blowup, np.inf, R.max(0))
    sr = (s_star[None, :] <= grid[:, None]).mean(1)
    auc = float(((sr[1:] + sr[:-1]) / 2 * np.diff(lg)).sum() / (lg[-1] - lg[0]))   # trapezoid (numpy-version safe)
    curves[exp] = sr

    # failure attribution and metric recomputation at a given scale
    def fail_step(s):
        over = R > s
        any_ = over.any(0)
        t = np.where(any_, over.argmax(0), alive.sum(0) - 1)   # last alive step if it never crosses
        return t, any_

    t1, f1 = fail_step(1.0)
    trig = {}
    for e in np.nonzero(f1)[0]:
        r_e = ratio[t1[e], e]
        if bih:
            si, ki = np.unravel_index(np.argmax(r_e), r_e.shape)
            k = f"{['rh', 'lh'][si]}_{KEYS[ki]}"
        else:
            k = KEYS[int(np.argmax(r_e))]
        trig[k] = trig.get(k, 0) + 1
    if blowup.any():
        trig["blowup"] = int(blowup.sum())
    ts, _ = fail_step(a.metrics_scale)
    mask = alive & (np.arange(alive.shape[0])[:, None] <= ts[None, :])
    X = {}
    for k in D.files:
        if not k.startswith("m_"):
            continue
        v = D[k].astype(np.float64)
        ok = mask & ~np.isnan(v)
        per_env = np.where(ok, np.nan_to_num(v), 0).sum(0) / np.maximum(ok.sum(0), 1)
        X["x_" + k[2:]] = float(per_env.mean())
    if bih:
        for k in ("power_joint_W", "power_wrist_W", "err_tip_mm", "err_kp_mm", "jerk_tip", "jerk_wrist"):
            if f"x_rh_{k}" in X:
                X[f"x_{k}"] = (X[f"x_rh_{k}"] + X[f"x_lh_{k}"]) * (1.0 if "power" in k else 0.5)
        cf = D["m_contact_force_N"]
        X["x_contact_frac"] = float(((np.where(mask, cf > 0.1, 0)).sum(0) / np.maximum(mask.sum(0), 1)).mean())
        X["x_contact_force_max_N"] = float(np.where(mask, cf, 0).max(0).mean())
    T = alive.shape[0]
    res = {**g, "exp": exp, "num_envs": int(alive.shape[1]), "clip_steps": int(T),
           "sr_auc": auc, "s_star_median": float(np.median(s_star)),
           "s_star_p10": float(np.percentile(s_star, 10)), "s_star_p90": float(np.percentile(s_star, 90)),
           "frac_blowup": float(blowup.mean()),
           **{f"sr_at_{s:g}": float((s_star <= s).mean()) for s in (0.5, 0.7, 1.0, 2.0, 5.0)},
           "fail_trigger_at_1": trig,
           "fail_frame_at_1_median": float(np.median(t1[f1])) if f1.any() else None,
           "metrics_scale": a.metrics_scale, "grace": GRACE,
           "success_rate": float((s_star <= a.metrics_scale).mean()),   # same key as the normal evaluator
           **X}
    with open(f"{SD}/{exp}_scale.json", "w") as f:
        json.dump(res, f, indent=2)
    # consistency check against the normal evaluation at scale 1
    ref = f"eval/sc1.0_{a.ckpt}/{exp}.json"
    chk = ""
    if os.path.exists(ref) and a.metrics_scale == 1.0:
        r0 = json.load(open(ref))
        chk = f" | check SR {r0['success_rate']:.3f} vs {res['sr_at_1']:.3f}, " \
              f"E_kp {r0.get('x_err_kp_mm', float('nan')):.1f} vs {X.get('x_err_kp_mm', float('nan')):.1f}"
    top = max(trig, key=trig.get) if trig else "-"
    print(f"{exp:42s} AUC {auc:.3f}  s* med {res['s_star_median']:6.2f}  SR@1 {res['sr_at_1']:.3f}"
          f"  blowup {res['frac_blowup']:.2f}  trigger@1 {top}{chk}")
    rows.append(res)

if not rows:
    raise SystemExit(f"no {SD}/*_frames.npz - run `SWEEP=1 bash eval_all.sh` first")

import pandas as pd
df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()} for r in rows])
df.to_csv(f"{OUT}/sweep_summary_{a.ckpt}{SUF}.csv", index=False)
np.savez(f"{OUT}/sweep_curves_{a.ckpt}{SUF}.npz", grid=grid, **curves)
print(f"\nwrote {OUT}/sweep_summary_{a.ckpt}{SUF}.csv, {OUT}/sweep_curves_{a.ckpt}{SUF}.npz ({len(rows)} runs)")

order = {"raw": 0, "f10": 1, "f5": 2}
# per-metric tables: rows = hand/side/method/lab, columns = clips + per-task means + overall mean
PRINT = ["sr_auc", "s_star_median", "success_rate", "x_err_tip_mm", "x_err_kp_mm", "x_power_joint_W",
         "x_power_wrist_W", "x_contact_frac", "x_contact_force_N"]
morder, lorder = {"dd": 0, "ours": 1}, {"raw": 0, "f10": 1, "f5": 2}
for col in [c for c in (a.print or PRINT) if c in df]:
    t = df.pivot_table(index=["hand", "side", "method", "lab"], columns="clip", values=col)
    for task in sorted(df["task"].unique()):
        cs = [c for c in sorted(df.loc[df["task"] == task, "clip"].unique()) if c in t]
        if cs:
            t[f"{task}_mean"] = t[cs].mean(axis=1)
    clips = [c for c in sorted(df["clip"].unique()) if c in t]
    t["mean"] = t[clips].mean(axis=1)
    t = t.reset_index()
    t = t.iloc[sorted(range(len(t)), key=lambda i: (t["hand"][i], t["side"][i], lorder.get(t["lab"][i], 9),
                                                   morder.get(t["method"][i], 9)))]
    print(f"\n=== {col} ===\n" + t.round(3).to_string(index=False))

if not a.no_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {("dd", "raw"): ("#c44e52", "-"), ("dd", "f10"): ("#dd8452", "--"), ("dd", "f5"): ("#937860", ":"),
             ("ours", "raw"): ("#4c72b0", "-")}
    label = {("dd", "raw"): "Data-driven", ("dd", "f10"): "+ LPF 10 Hz", ("dd", "f5"): "+ LPF 5 Hz",
             ("ours", "raw"): "Ours"}
    for (hand, side, task), sub in df.groupby(["hand", "side", "task"]):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for ml, (c, ls) in style.items():
            ex = sub[(sub["method"] == ml[0]) & (sub["lab"] == ml[1])]["exp"].tolist()
            if not ex:
                continue
            y = np.mean([curves[e] for e in ex], 0)
            auc = np.mean(sub[sub["exp"].isin(ex)]["sr_auc"])
            ax.plot(grid, 100 * y, color=c, ls=ls, lw=1.8, label=f"{label[ml]} ({auc:.2f})")
        ax.axvline(1.0, color="0.6", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_xlim(a.smin, a.smax); ax.set_ylim(-2, 102)
        ax.set_xlabel("termination threshold scale"); ax.set_ylabel("success rate (%)")
        ax.set_title(f"{hand} {'' if side == '-' else side + ' '}{task} (n={sub['clip'].nunique()} clips)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout()
        out = f"{OUT}/sweep_{hand}_{'single' if side == '-' else side}_{task}_{a.ckpt}{SUF}.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print("plot", out)