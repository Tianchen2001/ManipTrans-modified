#!/bin/bash
# Generalisation experiment (single hand): one multi-clip tracking policy per (method, training-set size, seed).
# Training clips come from data/handonly/splits_gen/train_<method>_n<N>.txt (written by HandMotion/prep_gen.py);
# env i tracks clip i % N (ManipTrans dataIndices list). Run names: g_<method>_n<N>_s<seed>_<dexhand>.
# Jobs run sequentially in the order of NS, then SEEDS, then METHODS (default: n40 s1 dd/ours, n40 s2, n40 s3,
# then n20, n10); finished runs are skipped.
#   bash train_multiclip.sh                                    # everything: dd+ours x 40,20,10 x seeds 1,2,3 on inspire, 1000 iterations
#   METHODS=ours NS="10 20" SEEDS=1 bash train_multiclip.sh    # a subset (e.g. split over two GPUs / tmux windows)
# Env overrides: METHODS="dd ours"  NS="40 20 10"  SEEDS="1 2 3"  DEXHAND=inspire  EPOCHS=1000  NUM_ENVS=4096
#                SPLITS=data/handonly/splits_gen  WS=<x> (as train_single.sh)  TAG=<suffix>
# Pick the GPU with a prefix:  CUDA_VISIBLE_DEVICES=2 bash train_multiclip.sh
set -u
cd "$(dirname "$0")"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
METHODS=${METHODS:-"dd ours"}
NS=${NS:-"40 20 10"}
SEEDS=${SEEDS:-"1 2 3"}
DEXHAND=${DEXHAND:-inspire}
EPOCHS=${EPOCHS:-1000}
NUM_ENVS=${NUM_ENVS:-4096}
SPLITS=${SPLITS:-data/handonly/splits_gen}
EXTRA=""
WS=${WS:-}
[ -z "$WS" ] && [ "$DEXHAND" = shadow ] && WS=2
[ -n "$WS" ] && EXTRA="task.env.translationScale=$WS task.env.orientationScale=$(python3 -c "print(round(0.1*$WS, 4))")"
TAG=${TAG:-}
WANDB_PROJECT=${WANDB_PROJECT:-mano-tracking-gen}
mkdir -p logs

for n in $NS; do
  for seed in $SEEDS; do
    for m in $METHODS; do
      list=$SPLITS/train_${m}_n${n}.txt
      exp=g_${m}_n${n}_s${seed}_${DEXHAND}${TAG}
      [ -f "$list" ] || { echo "missing $list, skip"; continue; }
      idx=$(grep -v '^\s*$' "$list" | tr '\n' ',' | sed 's/,$//')
      k=$(echo "$idx" | tr ',' '\n' | wc -l)
      miss=$(for i in ${idx//,/ }; do [ -f data/handonly/right/$i.npz ] || echo $i; done)
      [ -n "$miss" ] && { echo "missing clips for $exp: $miss, skip"; continue; }
      [ "$k" -ne "$n" ] && echo "warning: $list has $k clips, expected $n"
      if ls runs/${exp}__* >/dev/null 2>&1; then echo "exists runs/${exp}__*, skip"; continue; fi
      echo "=== $(date '+%m-%d %H:%M:%S') start $exp  ($k clips)"
      python -u main/rl/train.py task=DexHandImitator dexhand=$DEXHAND side=RH headless=true seed=$seed \
          num_envs=$NUM_ENVS "dataIndices=[$idx]" randomStateInit=true rolloutStateInit=false \
          actionsMovingAverage=0.4 early_stop_epochs=$EPOCHS max_iterations=$EPOCHS experiment=$exp $EXTRA \
          wandb_activate=true wandb_project=${WANDB_PROJECT} wandb_group=gen_${DEXHAND} \
          "wandb_tags=[gen,${m},n${n},s${seed},${DEXHAND}]" \
          2>&1 | tee logs/$exp.log || echo "!!! failed $exp"
      echo "=== $(date '+%m-%d %H:%M:%S') done  $exp"
    done
  done
done
echo "=== train_multiclip.sh finished $(date '+%m-%d %H:%M')"
