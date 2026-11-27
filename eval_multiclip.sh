#!/bin/bash
# Evaluate the multi-clip tracking policies (train_multiclip.sh, runs/g_*) on held-out observed clips.
# One evaluation per (run, checkpoint, clip set): all clips of the set in one process, env i -> clip i % n_clips,
# no termination (scale 1e6) + per-frame dump, exactly like `SWEEP=1 bash eval_all.sh`; sweep_multiclip.py then
# computes AUC / P_wrist / ... per clip.
#   CUDA_VISIBLE_DEVICES=1 bash eval_multiclip.sh                        # ep3000 on the 10 test clips
#   CUDA_VISIBLE_DEVICES=1 CKPTS="1000 2000 3000 4000 5000" MATCH=_n40_s1 bash eval_multiclip.sh   # how long to train
#   CUDA_VISIBLE_DEVICES=1 SETS="test seen" bash eval_multiclip.sh       # + observed motion of the training clips
# Env:
#   CKPTS = iterations to evaluate (default 3000; runs without that checkpoint yet are skipped)
#   SETS  = test (data/handonly/splits_gen/test.txt) | seen (h_obs_* of the run's own training clips)
#   PER   = rollouts per clip (default 200 -> 2000 envs for 10 clips)   MATCH = substring of the run name
#   SPLITS = data/handonly/splits_gen
# Output: eval/multiclip_ep<ckpt>/<exp>__<set>{.json,_frames.npz,_x.npz,...} + <exp>__<set>_clips.txt (env order)
#         logs in eval/multiclip_ep<ckpt>/logs/ ; existing results are skipped.
cd "$(dirname "$0")"
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export EVAL_FRAMES=1
CKPTS=${CKPTS:-3000}
SETS=${SETS:-test}
PER=${PER:-200}
MATCH=${MATCH:-}
SPLITS=${SPLITS:-data/handonly/splits_gen}

for d in $(ls -d runs/g_*__* 2>/dev/null | sort -r); do              # newest first if a run was restarted
  exp=$(basename "$d" | sed 's/__.*//')
  [[ $exp =~ ^g_([a-z]+)_n([0-9]+)_s([0-9]+)_([a-z]+) ]] || continue
  m=${BASH_REMATCH[1]}; n=${BASH_REMATCH[2]}; hand=${BASH_REMATCH[4]}
  [[ -n "$MATCH" && "$exp" != *"$MATCH"* ]] && continue
  ts=$(grep -m1 -E '^ +translationScale:' $d/config.yaml 2>/dev/null | awk '{print $2}')
  os_=$(grep -m1 -E '^ +orientationScale:' $d/config.yaml 2>/dev/null | awk '{print $2}')
  for ck in $CKPTS; do
    ckpt=$(ls $d/nn/last_${exp}_ep_${ck}_*.pth 2>/dev/null | head -1)
    [ -n "$ckpt" ] || { echo "no ep$ck yet: $exp"; continue; }
    OUTDIR=eval/multiclip_ep$ck; mkdir -p $OUTDIR/logs
    for set in $SETS; do
      out=$OUTDIR/${exp}__${set}
      [ -f "$out.json" ] && { echo "skip $exp ep$ck $set"; continue; }
      if [ "$set" = test ]; then
        clips=$(grep -v '^\s*$' $SPLITS/test.txt)
      elif [ "$set" = seen ]; then                                  # observed motion of the training clips
        clips=$(grep -v '^\s*$' $SPLITS/train_${m}_n${n}.txt | sed "s/^h_${m}_/h_obs_/")
      else
        echo "unknown set $set"; continue
      fi
      k=$(echo "$clips" | wc -l)
      echo "$clips" > ${out}_clips.txt
      idx=$(echo "$clips" | tr '\n' ',' | sed 's/,$//')
      echo ">>> $(date '+%H:%M:%S') $exp ep$ck $set ($k clips x $PER)"
      EVAL_OUT=$out python -u main/rl/train.py task=DexHandImitator dexhand=$hand side=RH headless=true \
          test=true num_envs=$((k * PER)) randomStateInit=false actionsMovingAverage=0.4 \
          task.env.translationScale=${ts:-1.0} task.env.orientationScale=${os_:-0.1} \
          "dataIndices=[$idx]" "checkpoint='$ckpt'" task.env.testScaleFactor=1000000.0 \
          num_rollouts_to_run=999999999 wandb_activate=false experiment=eval_$exp \
          > $OUTDIR/logs/${exp}__${set}.log 2>&1
      res=$(grep "^\[eval\] {" $OUTDIR/logs/${exp}__${set}.log | tail -1)
      if [ -n "$res" ]; then echo "    ${res:0:150}"; else echo "    FAILED, see $OUTDIR/logs/${exp}__${set}.log"; fi
    done
  done
done
