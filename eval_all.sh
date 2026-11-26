#!/bin/bash
# First-episode success evaluation for trained runs.
#   CUDA_VISIBLE_DEVICES=2 CKPT=ep1000 HAND=artimano bash eval_all.sh     # recommended
#   CUDA_VISIBLE_DEVICES=2 CKPT=ep1000 MATCH=_mib bash eval_all.sh        # only names containing _mib
# Env:
#   CKPT  = ep1000 (nn/last_<exp>_ep_1000_*.pth; runs that have not reached iter 1000 are skipped)
#         | best   (nn/<exp>.pth = best training reward so far; also picks up runs still training!)
#   HAND  = artimano|inspire|shadow|allegro   MATCH = substring of the run name
#   SINGLE=1  only single-hand runs (skip *_rh_*, *_lh_*, *_bih_*)
#   N     = number of rollouts (default 1000)  SCALE = testScaleFactor (default 1.0)
#   SWEEP=1  evaluate without termination and dump per-frame data -> eval/sweep_<CKPT>/<exp>_frames.npz
#            (needs the BiH env in this repo), then run sweep_scale.py
# Results: eval/<TAG>/<exp>.json/.npz/_x.npz/_traj.npz[/_frames.npz], logs in eval/<TAG>/logs/<exp>.log,
#          TAG = sc<SCALE>_<CKPT> | sweep_<CKPT> ; existing results are skipped.
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
N=${N:-1000}
SCALE=${SCALE:-1.0}
CKPT=${CKPT:-ep1000}
HAND=${HAND:-}
MATCH=${MATCH:-}
TAG=sc${SCALE}; [ "$CKPT" != "best" ] && TAG=${TAG}_${CKPT}
if [ "${SWEEP:-0}" = 1 ]; then          # SWEEP=1: no termination + per-frame dump for sweep_scale.py
  SCALE=1000000.0; export EVAL_FRAMES=1; TAG=sweep; [ "$CKPT" != "best" ] && TAG=${TAG}_${CKPT}
fi
OUTDIR=eval/$TAG; LOGDIR=$OUTDIR/logs
mkdir -p $LOGDIR

for d in $(ls -d runs/h_*__* 2>/dev/null | sort); do
  exp=$(basename "$d" | sed 's/__.*//')                 # h_dd_traj0_raw_shadow_pid
  pid=0; base=$exp
  [[ $base =~ (_w[0-9.]+)$ ]] && base=${base%${BASH_REMATCH[1]}}    # tuning tag (TAG=_w3), see run_dd.sh
  [[ $base == *_pid ]] && { pid=1; base=${base%_pid}; }
  hand=${base##*_}
  idx=${base%_${hand}}
  side=RH                                               # two-hand runs: ..._raw_lh_<hand> / _rh_
  [[ $idx == *_lh ]] && { side=LH; idx=${idx%_lh}; }
  [[ $idx == *_rh ]] && { side=RH; idx=${idx%_rh}; }
  [[ $idx == *_bih ]] && { side=BiH; idx=${idx%_bih}; }                                  # h_dd_traj0_raw
  [[ "$hand" =~ ^(artimano|inspire|shadow|allegro)$ ]] || continue
  [[ -n "$HAND" && "$hand" != "$HAND" ]] && continue
  [[ -n "$MATCH" && "$exp" != *"$MATCH"* ]] && continue
  [[ "${SINGLE:-0}" == 1 && "$exp" =~ _(rh|lh|bih)_ ]] && continue   # SINGLE=1: single-hand runs only
  if [ "$CKPT" = "best" ]; then
    ckpt=$d/nn/$exp.pth
  else
    ep=${CKPT#ep}
    ckpt=$(ls $d/nn/last_${exp}_ep_${ep}_*.pth 2>/dev/null | head -1)
  fi
  [ -n "$ckpt" ] && [ -f "$ckpt" ] || { echo "no $CKPT ckpt yet: $exp"; continue; }
  out=$OUTDIR/${exp}
  [ -f "$out.json" ] && { echo "skip $exp"; continue; }
  extra=""; [ $pid = 1 ] && extra="usePIDControl=True"
  [ $side = BiH ] && extra="$extra bimanual_mode=united"
  ts=$(grep -m1 -E '^ +translationScale:' $d/config.yaml 2>/dev/null | awk '{print $2}')
  os_=$(grep -m1 -E '^ +orientationScale:' $d/config.yaml 2>/dev/null | awk '{print $2}')
  extra="$extra task.env.translationScale=${ts:-1.0} task.env.orientationScale=${os_:-0.1}"   # as in training
  echo ">>> $(date '+%H:%M:%S') $exp  ($(basename $ckpt))"
  EVAL_OUT=$out python -u main/rl/train.py task=DexHandImitator dexhand=$hand side=$side headless=true \
      test=true num_envs=$N randomStateInit=false actionsMovingAverage=0.4 $extra \
      dataIndices=[$idx] "checkpoint='$ckpt'" task.env.testScaleFactor=$SCALE \
      num_rollouts_to_run=999999999 wandb_activate=false experiment=eval_$exp \
      > $LOGDIR/${exp}.log 2>&1
  res=$(grep "^\[eval\] {" $LOGDIR/${exp}.log | tail -1)
  if [ -n "$res" ]; then echo "    ${res:0:160}"; else echo "    FAILED, see $LOGDIR/${exp}.log"; fi
done