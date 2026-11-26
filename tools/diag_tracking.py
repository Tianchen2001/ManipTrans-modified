"""
Where does a run fail to track? Compares DD and Ours on one clip, per hand, from the sweep dumps.
Run from the ManipTrans root (needs eval/sweep_ep1000/<run>_frames.npz and data/handonly/right/<clip npz>):

    python diag_tracking.py --clips traj3 traj4 --hands artimano inspire shadow

Per run it prints, for each keypoint group, the error ratio (distance / termination threshold at scale 1, median over
the 1000 rollouts) averaged over the clip and at its worst frame, and which group/frame sets s*.
Per clip and hand it writes vis/diag/<clip>_<hand>.png:
  top    7 panels (one per keypoint group): median ratio over time, DD solid, Ours dashed (dotted line = 1)
  bottom reference finger motion: total flexion angle per finger (sum of |rotvec| of its 3 MANO joints), DD vs Ours,
         and the worst frame of each method (vertical lines)
"""
import argparse, os
import numpy as np

K = ["thumb tip", "index tip", "middle tip", "pinky tip", "ring tip", "level-1 joints", "level-2 joints"]
FINGERS = {"index": [0, 1, 2], "middle": [3, 4, 5], "pinky": [6, 7, 8], "ring": [9, 10, 11], "thumb": [12, 13, 14]}

ap = argparse.ArgumentParser()
ap.add_argument("--clips", nargs="+", default=["traj3", "traj4"])
ap.add_argument("--hands", nargs="+", default=["artimano", "inspire", "shadow"])
ap.add_argument("--lab", default="raw")
ap.add_argument("--sweep-dir", default="eval/sweep_ep1000")
ap.add_argument("--data-dir", default="data/handonly/right")
ap.add_argument("--out", default="vis/diag")
ap.add_argument("--grace", type=int, default=20)
a = ap.parse_args()
GRACE = a.grace
os.makedirs(a.out, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ratio(exp):
    p = f"{a.sweep_dir}/{exp}_frames.npz"
    if not os.path.exists(p):
        return None
    D = np.load(p)
    r = D["ratio"].astype(np.float64)                       # [T, n, 7]
    valid = D["alive"].astype(bool) & (D["run"] >= GRACE)
    r = np.where(valid[..., None], r, np.nan)
    return np.nanmedian(r, axis=1)                          # [T, 7]


def finger_flex(clip, m):
    p = f"{a.data_dir}/h_{m}_{clip}_{a.lab if m == 'dd' else 'raw'}.npz"
    hp = np.load(p)["hand_pose"].reshape(-1, 15, 3)
    ang = np.degrees(np.linalg.norm(hp, axis=-1))           # [T, 15]
    return {f: ang[:, j].sum(-1) for f, j in FINGERS.items()}


for clip in a.clips:
    flex = {m: finger_flex(clip, m) for m in ("dd", "ours")}
    for hand in a.hands:
        R = {m: load_ratio(f"h_{m}_{clip}_{a.lab if m == 'dd' else 'raw'}_{hand}") for m in ("dd", "ours")}
        if any(v is None for v in R.values()):
            print(f"skip {clip} {hand}: missing sweep dump"); continue
        print(f"\n=== {clip} | {hand}")
        worst = {}
        for m, r in R.items():
            rm = np.nan_to_num(r, nan=0.0)
            t, k = np.unravel_index(rm.argmax(), rm.shape)
            worst[m] = t
            print(f"  {m:4s} s*(median) {rm.max():.2f} at frame {t:3d} ({K[k]})")
            print("       mean ratio per group: " + "  ".join(f"{K[i].split()[0]} {np.nanmean(r[:, i]):.2f}"
                                                          for i in range(7)))
            print("       max  ratio per group: " + "  ".join(f"{K[i].split()[0]} {rm[:, i].max():.2f}"
                                                          for i in range(7)))
        fig, ax = plt.subplots(2, 7, figsize=(22, 6), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        for i in range(7):
            for m, ls in (("dd", "-"), ("ours", "--")):
                ax[0, i].plot(R[m][:, i], ls=ls, color="#c44e52" if m == "dd" else "#4c72b0", label=m)
            ax[0, i].axhline(1, color="0.6", ls=":", lw=0.8)
            ax[0, i].set_title(K[i], fontsize=9)
        ax[0, 0].set_ylabel("error / threshold")
        ax[0, 0].legend(fontsize=8)
        for i, f in enumerate(FINGERS):
            for m, ls in (("dd", "-"), ("ours", "--")):
                ax[1, i].plot(np.arange(len(flex[m][f])), flex[m][f], ls=ls,
                              color="#c44e52" if m == "dd" else "#4c72b0")
                ax[1, i].axvline(worst[m], ls=ls, color="#c44e52" if m == "dd" else "#4c72b0", lw=0.8, alpha=0.6)
            ax[1, i].set_title(f"reference {f} flexion (deg)", fontsize=9)
        for i in range(5, 7):
            ax[1, i].axis("off")
        for x in ax[1]:
            x.set_xlabel("frame (60 fps)")
        fig.suptitle(f"{clip} | {hand}: top = tracking error per keypoint group (median over rollouts), "
                     f"bottom = reference finger motion; vertical lines = worst frame", fontsize=10)
        fig.tight_layout()
        out = f"{a.out}/{clip}_{hand}.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print("  ->", out)