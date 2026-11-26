#!/bin/bash
# Sequentially train tracking policies for the given cutoff labels over all clips, on one dexhand.
# One instance per tmux window; two instances share a GPU.
#   bash train_single.sh raw f12                          # dd, artimano
#   DEXHAND=inspire bash train_single.sh raw f8           # dd, inspire
#   DEXHAND=inspire METHOD=ours bash train_single.sh raw  # ours, inspire
# Env overrides: DEXHAND=artimano|inspire|shadow|allegro  METHOD=dd|ours  CLIPS="22 841 2100"
#                EPOCHS=3000  NUM_ENVS=4096  SEED=42  PID=0|1 (PID wrist control; run names get _pid)
#                WS=<x> wrist force/torque multiplier (default 2 for shadow, else ManipTrans 1)  TAG=<suffix>
# Pick the GPU with a prefix:  CUDA_VISIBLE_DEVICES=2 DEXHAND=inspire bash train_single.sh raw f8
set -e
cd "$(dirname "$0")"                                   # run from the ManipTrans root, wherever invoked
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}       # lib/ main/ maniptrans_envs/ importable
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # libpython3.8 for isaacgym
LABS=("$@")
[ ${#LABS[@]} -eq 0 ] && { echo "usage: bash train_single.sh <lab> [<lab> ...]   e.g. raw f12"; exit 1; }
CLIPS=${CLIPS:-"22 841 2100"}
METHOD=${METHOD:-dd}
DEXHAND=${DEXHAND:-artimano}
EPOCHS=${EPOCHS:-3000}
NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-42}
PID=${PID:-0}
EXTRA=""
SUFFIX=""
[ "$PID" = "1" ] && { EXTRA="usePIDControl=True"; SUFFIX="_pid"; }
WS=${WS:-}                              # wrist authority: translationScale = WS, orientationScale = 0.1*WS
[ -z "$WS" ] && [ "$DEXHAND" = shadow ] && WS=2         # Shadow is 2x heavier than Arti-MANO (1.27 vs 0.64 kg)
[ -n "$WS" ] && EXTRA="$EXTRA task.env.translationScale=$WS task.env.orientationScale=$(python3 -c "print(round(0.1*$WS, 4))")"
TAG=${TAG:-}                            # appended to the run name (tuning runs only, e.g. TAG=_w3)
mkdir -p logs

for lab in "${LABS[@]}"; do
  for clip in $CLIPS; do
    idx=h_${METHOD}_${clip}_${lab}
    exp=${idx}_${DEXHAND}
    exp=${exp}${SUFFIX}
    [ "$SEED" != "42" ] && exp=${exp}_s${SEED}
    exp=${exp}${TAG}
    if [ ! -f data/handonly/right/$idx.npz ]; then echo "missing $idx.npz, skip"; continue; fi
    if ls runs/${exp}__* >/dev/null 2>&1; then echo "exists runs/${exp}__*, skip"; continue; fi
    echo "=== $(date '+%H:%M:%S') start $exp"
    python -u main/rl/train.py task=DexHandImitator dexhand=$DEXHAND side=RH headless=true seed=$SEED \
        num_envs=$NUM_ENVS dataIndices=[$idx] randomStateInit=true actionsMovingAverage=0.4 \
        early_stop_epochs=$EPOCHS max_iterations=$EPOCHS experiment=$exp $EXTRA \
        wandb_activate=true wandb_project=mano-tracking \
        wandb_group=${METHOD}_${DEXHAND}${SUFFIX} wandb_tags=[$lab,clip$clip,$DEXHAND] \
        2>&1 | tee logs/$exp.log
    echo "=== $(date '+%H:%M:%S') done  $exp"
  done
done