"""
Where does a two-hand run lose tracking? Reads a sweep eval dump (no termination, per-frame data):

    python diag_bih.py eval/sweep/h_dd_btraj0_raw_bih_shadow [eval/sweep_ep1000/h_dd_btraj0_raw_bih_inspire ...]

(argument = path prefix without _frames.npz; eval/sweep/ = CKPT=best, i.e. a run that is still training)
Per run and hand it prints:
  * per keypoint group: median (over rollouts) error / termination threshold, mean over the clip and max
  * first frame (after the 20-step grace) where the median ratio of any group exceeds 1, and which group
    -> roughly where training episodes get terminated
  * wrist position / rotation error at that frame and its max over the clip
  * speed of the reference keypoint 0 (wrist) around that frame (from <prefix>_traj.npz, if present)
and writes <prefix>_diag.png: ratio per group over time, both hands, plus wrist errors.
"""
import sys, os
import numpy as np

K = ["thumb", "index", "middle", "pinky", "ring", "lvl1", "lvl2"]
GRACE = 20

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for pre in sys.argv[1:]:
    p = pre + "_frames.npz"
    if not os.path.exists(p):
        print("missing", p); continue
    D = np.load(p)
    name = os.path.basename(pre)
    run = D["run"]                                            # [T, n]
    print(f"\n=== {name}  ({run.shape[1]} rollouts, {run.shape[0]} frames)")
    tr = np.load(pre + "_traj.npz") if os.path.exists(pre + "_traj.npz") else None
    fig, ax = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for j, s in enumerate(("rh", "lh")):
        r = np.where((run >= GRACE)[..., None], D[f"ratio_{s}"], np.nan)
        med = np.nanmedian(r, axis=1)                         # [T, 7]
        over = np.where(np.nan_to_num(med, nan=0).max(-1) > 1)[0]
        t1 = int(over[0]) if len(over) else None
        wp = np.nanmedian(D[f"m_{s}_err_wrist_pos_mm"], axis=1)
        wr = np.nanmedian(D[f"m_{s}_err_wrist_rot_deg"], axis=1)
        print(f"  [{s}] mean ratio: " + "  ".join(f"{k} {np.nanmean(med[:, i]):.2f}" for i, k in enumerate(K)))
        print(f"       max  ratio: " + "  ".join(f"{k} {np.nanmax(med[:, i]):.2f}" for i, k in enumerate(K)))
        if t1 is None:
            print("       median rollout never exceeds the threshold")
        else:
            g = K[int(np.nan_to_num(med[t1], nan=0).argmax())]
            print(f"       first frame over threshold: {t1} ({g}); wrist err there {wp[t1]:.0f} mm / {wr[t1]:.0f} deg")
        print(f"       wrist err max over clip: {np.nanmax(wp):.0f} mm / {np.nanmax(wr):.0f} deg "
              f"(at frames {int(np.nanargmax(wp))} / {int(np.nanargmax(wr))})")
        if tr is not None and f"kp_ref_{s}" in tr:
            ref = tr[f"kp_ref_{s}"][:, 0, 0]                  # [T, 3] keypoint 0 of env 0
            v = np.linalg.norm(np.diff(ref, axis=0), axis=-1) * 60
            lo, hi = (max(0, t1 - 10), t1 + 10) if t1 is not None else (0, len(v))
            print(f"       reference wrist speed: max {v.max():.2f} m/s at frame {int(v.argmax())}; "
                  f"around frames {lo}-{hi}: max {v[lo:hi].max():.2f} m/s")
        for i, k in enumerate(K):
            ax[0, j].plot(med[:, i], label=k)
        ax[0, j].axhline(1, color="0.5", ls=":")
        if t1 is not None:
            for a in ax[:, j]:
                a.axvline(t1, color="r", lw=0.8)
        ax[0, j].set_title(f"{s}: error / threshold (median over rollouts)")
        ax[1, j].plot(wp); ax[1, j].set_title(f"{s}: wrist position error (mm)")
        ax[2, j].plot(wr); ax[2, j].set_title(f"{s}: wrist rotation error (deg)")
        ax[2, j].set_xlabel("frame (60 fps)")
    ax[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(pre + "_diag.png", dpi=100); plt.close(fig)
    print("  ->", pre + "_diag.png")
