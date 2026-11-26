#!/bin/bash
# Bimanual imitator with hand-hand collision (one policy for both hands).
#   CUDA_VISIBLE_DEVICES=3 CLIPS="bseq1" METHOD=ours bash run_bih.sh raw
# Env: CLIPS  METHOD=dd|ours  DEXHAND=artimano  EPOCHS=1000  NUM_ENVS=4096  SEED=42  PID=0|1
#      WS=<x> wrist force/torque multiplier (default 2 for shadow)  TAG=<run-name suffix>
#      CONTACT_PAIRS=<n> PhysX max_gpu_contact_pairs (default 8388608 for shadow, else the task config value)
#      WANDB=true|false  WANDB_PROJECT=handx-bi
# Data: data/handonly/right/<idx>.npz AND data/handonly/left/<idx>.npz  (prep_handx_bi.py)
# Run name: h_<method>_<clip>_<lab>_bih_<dexhand>[_pid]
set -e
cd "$(dirname "$0")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
LABS=("$@")
[ ${#LABS[@]} -eq 0 ] && { echo "usage: bash run_bih.sh <lab> [<lab> ...]"; exit 1; }
CLIPS=${CLIPS:?set CLIPS, e.g. CLIPS=\"bseq1\"}
METHOD=${METHOD:-dd}
DEXHAND=${DEXHAND:-artimano}
EPOCHS=${EPOCHS:-1000}
NUM_ENVS=${NUM_ENVS:-4096}
SEED=${SEED:-42}
PID=${PID:-0}
WANDB=${WANDB:-true}                    # live logging (x-axis = iteration, needs the after_init mirror)
WANDB_PROJECT=${WANDB_PROJECT:-handx-bi}
EXTRA=""; SUFFIX=""
[ "$PID" = "1" ] && { EXTRA="usePIDControl=True"; SUFFIX="_pid"; }
WS=${WS:-}                              # wrist authority: translationScale = WS, orientationScale = 0.1*WS
[ -z "$WS" ] && [ "$DEXHAND" = shadow ] && WS=2         # Shadow is 2x heavier than Arti-MANO (1.27 vs 0.64 kg)
[ -n "$WS" ] && EXTRA="$EXTRA task.env.translationScale=$WS task.env.orientationScale=$(python3 -c "print(round(0.1*$WS, 4))")"
TAG=${TAG:-}                            # appended to the run name (tuning runs only, e.g. TAG=_w3)
# PhysX GPU contact buffer: two Shadow hands x 4096 envs need ~2.9M contact pairs (> default) -> overflow + CUDA crash
CONTACT_PAIRS=${CONTACT_PAIRS:-}
[ -z "$CONTACT_PAIRS" ] && [ "$DEXHAND" = shadow ] && CONTACT_PAIRS=8388608
[ -n "$CONTACT_PAIRS" ] && EXTRA="$EXTRA task.sim.physx.max_gpu_contact_pairs=$CONTACT_PAIRS"
# training termination thresholds x THR (const; eval is unaffected: it uses testScaleFactor / the scale sweep)
THR=${THR:-}
[ -n "$THR" ] && EXTRA="$EXTRA task.env.tightenMethod=const task.env.tightenFactor=$THR"
mkdir -p logs

for lab in "${LABS[@]}"; do
  for clip in $CLIPS; do
    idx=h_${METHOD}_${clip}_${lab}
    exp=${idx}_bih_${DEXHAND}${SUFFIX}
    [ "$SEED" != "42" ] && exp=${exp}_s${SEED}
    exp=${exp}${TAG}
    for d in right left; do
      [ -f data/handonly/$d/$idx.npz ] || { echo "missing $d/$idx.npz, skip"; continue 2; }
    done
    if ls runs/${exp}__* >/dev/null 2>&1; then echo "exists runs/${exp}__*, skip"; continue; fi
    echo "=== $(date '+%H:%M:%S') start $exp"
    python -u main/rl/train.py task=DexHandImitator dexhand=$DEXHAND side=BiH bimanual_mode=united \
        headless=true seed=$SEED num_envs=$NUM_ENVS dataIndices=[$idx] randomStateInit=true \
        actionsMovingAverage=0.4 early_stop_epochs=$EPOCHS max_iterations=$EPOCHS experiment=$exp $EXTRA \
        wandb_activate=$WANDB wandb_project=$WANDB_PROJECT wandb_group=${METHOD}_${DEXHAND}${SUFFIX} \
        "wandb_tags=[$lab,$clip,$DEXHAND]" \
        2>&1 | tee logs/$exp.log
    echo "=== $(date '+%H:%M:%S') done  $exp"
  done
done