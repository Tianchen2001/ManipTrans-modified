"""
Upload rl_games tfevents to wandb WITHOUT wandb's tensorboard converter.
x-axis = training iteration (`iter`); optionally only the first --max-iter iterations.

  python upload_tb_to_wandb.py runs/h_*_traj*__* --project mano-tracking --max-iter 1000
  python upload_tb_to_wandb.py runs/h_ours_012__09-30-11-20-01 --project mano-tracking --name ours/012

Tags logged per iteration (".../iter") are used directly; tags logged only per frame (losses/*, info/*,
performance/*) are mapped frame -> iteration with info/epochs. ".../step" and ".../time" duplicates are dropped.
Name = run dir without timestamp and leading "h_"; group = <method>_<hand>[_pid] (as in train_single.sh),
tags = [lab, clip, task, hand]. Run id = <id-prefix><dir name without timestamp>, so re-running resumes the
same wandb run (delete it on wandb first if you want a clean re-upload).
Each successful upload leaves runs/<run>/.wandb_uploaded_<project>; such runs are skipped next time
(--force re-uploads; delete the wandb run first, otherwise rows are appended twice).
Runs still training (event file written in the last --active-minutes) are skipped, and so are runs that
stopped before --max-iter unless --allow-partial.
"""
import argparse, bisect, glob, os, re, time
from collections import defaultdict
import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ap = argparse.ArgumentParser()
ap.add_argument("run_dirs", nargs="+")
ap.add_argument("--project", required=True)
ap.add_argument("--entity", default=None)
ap.add_argument("--name", default=None, help="display name (single run only)")
ap.add_argument("--group", default=None, help="override the inferred group")
ap.add_argument("--max-iter", type=int, default=None, help="upload only iterations <= this")
ap.add_argument("--id-prefix", default="it_")
ap.add_argument("--dry-run", action="store_true", help="parse and print, do not upload")
ap.add_argument("--active-minutes", type=float, default=10,
                help="skip runs whose event files were written in the last N minutes (still training)")
ap.add_argument("--force", action="store_true", help="upload even if this run was already uploaded to --project")
ap.add_argument("--allow-partial", action="store_true",
                help="also upload runs that stopped before --max-iter (crashed / unfinished)")
a = ap.parse_args()
assert a.name is None or len(a.run_dirs) == 1, "--name only makes sense for a single run"

NAME_RE = re.compile(r"h_(?P<method>[a-z]+)_(?P<clip>(?P<task>[a-z]*)\d+)_(?P<lab>[a-z0-9]+)(?:_(?P<side>rh|lh|bih))?_(?P<hand>[a-z]+)(?P<pid>_pid)?$")


def load_scalars(summaries_dir):
    """-> {tag: [(step, value)]}; add_scalars sub-dirs become '<main tag>/<key>'."""
    out = {}
    for d in [summaries_dir] + sorted(glob.glob(os.path.join(summaries_dir, "**/"), recursive=True)):
        if not glob.glob(os.path.join(d, "events.out.tfevents.*")):
            continue
        ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
        sub = os.path.basename(os.path.relpath(d, summaries_dir).rstrip("/"))
        for tag in ea.Tags()["scalars"]:
            if sub in ("", "."):
                full = tag
            else:                                   # e.g. dir reward_dict_iter_reward_power, tag reward_dict/iter
                pre = tag.replace("/", "_") + "_"
                full = f"{tag}/{sub[len(pre):] if sub.startswith(pre) else sub}"
            out[full] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def to_iter_rows(scalars):
    ep = sorted(scalars.get("info/epochs", []))
    frames, epochs = [f for f, _ in ep], [int(round(v)) for _, v in ep]

    def f2i(frame):
        if not frames:
            return None
        k = bisect.bisect_left(frames, frame)
        k = min(max(k, 0), len(frames) - 1)
        return epochs[k]

    rows = defaultdict(dict)
    for tag, pts in scalars.items():
        parts = tag.split("/")
        if len(parts) >= 2 and parts[1] == "iter":  # already per iteration
            name = "/".join([parts[0]] + parts[2:])
            for it, v in pts:
                rows[int(it)][name] = v
        elif len(parts) >= 2 and parts[1] in ("step", "time"):
            continue                                # duplicates of the /iter series
        else:                                       # per-frame only -> map to iteration
            for fr, v in pts:
                it = f2i(fr)
                if it is not None:
                    rows[it][tag] = v
    return rows


for run_dir in a.run_dirs:
    run_dir = run_dir.rstrip("/")
    base = os.path.basename(run_dir).split("__")[0]
    name = a.name or (base[2:] if base.startswith("h_") else base)
    m = NAME_RE.match(base)
    g = m.groupdict() if m else {}
    group = a.group or (f"{g['method']}_{g['hand']}{g['pid'] or ''}" if g else None)
    tags = [t for t in (g.get("lab"), g.get("clip"), g.get("task") or None, g.get("hand"), g.get("side")) if t] if g else []

    marker = os.path.join(run_dir, f".wandb_uploaded_{a.project}")
    if os.path.exists(marker) and not a.force:
        print(f"skip {name}: already uploaded to {a.project} ({open(marker).read().strip()}; --force to redo)"); continue

    ev = glob.glob(os.path.join(run_dir, "summaries", "**", "events.out.tfevents.*"), recursive=True)
    if not ev:
        print(f"skip {run_dir}: no event files"); continue
    idle_min = (time.time() - max(os.path.getmtime(f) for f in ev)) / 60
    if idle_min < a.active_minutes:
        print(f"skip {name}: still running (last write {idle_min:.1f} min ago)"); continue

    scalars = load_scalars(os.path.join(run_dir, "summaries"))
    if not scalars:
        print(f"skip {run_dir}: no scalars"); continue
    rows = to_iter_rows(scalars)
    its = sorted(i for i in rows if a.max_iter is None or i <= a.max_iter)
    if not its:
        print(f"skip {run_dir}: no iterations"); continue
    if a.max_iter is not None and its[-1] < a.max_iter and not a.allow_partial:
        print(f"skip {name}: only reached iter {its[-1]} < {a.max_iter} (unfinished/crashed; --allow-partial to upload)")
        continue
    if a.dry_run:
        print(f"{name} [{group}] tags={tags}: iters {its[0]}..{its[-1]} ({len(its)} rows), "
              f"{len({k for i in its for k in rows[i]})} metrics")
        continue

    run = wandb.init(project=a.project, entity=a.entity, group=group, name=name, tags=tags,
                     id=a.id_prefix + base, resume="allow", reinit=True)
    wandb.define_metric("iter")
    wandb.define_metric("*", step_metric="iter")
    for it in its:
        wandb.log({**rows[it], "iter": it})
    run.finish()
    with open(marker, "w") as f:
        f.write(f"id={a.id_prefix + base} iters={its[0]}..{its[-1]} at {time.strftime('%Y-%m-%d %H:%M')}\n")
    print(f"{name} [{group}]: iters {its[0]}..{its[-1]}, {len(its)} rows")
