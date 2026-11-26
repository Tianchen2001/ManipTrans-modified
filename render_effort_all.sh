#!/bin/bash
# Filmstrip (6 frames) + effort curve for every clip, single and two hands. Run from the ManipTrans root:
#   bash render_effort_all.sh
# Settings (edit here or pass as env vars, e.g. COLOR_DD="#888888" HANDS="inspire artimano" bash render_effort_all.sh)
HANDS=${HANDS:-"inspire"}                 # artimano inspire shadow
EFFORT=${EFFORT:-total}                   # wrist | joint | total
BASE_BIH=${BASE_BIH:-dd}                  # two-hand baseline: dd | hx
COLOR_DD=${COLOR_DD:-"#6FD1D7"}
COLOR_HX=${COLOR_HX:-"#e09f3e"}
COLOR_OURS=${COLOR_OURS:-"#FF4081"}
AZIM=${AZIM:-225}
ELEV=${ELEV:-"-89"}                       # -89 = top-down (-90 is degenerate in MuJoCo)
SNAPSHOTS=${SNAPSHOTS:-6}
STRIP_WIDTH=${STRIP_WIDTH:-2000}          # final width in px (2000 px ~ 6.7 in at 300 dpi = double column)
STRIP_SIZE=${STRIP_SIZE:-"800 480"}       # render size of one panel (w h); wider than tall -> flatter strip
STRIP_FONT=${STRIP_FONT:-57}              # text height in px in the final figure (strip labels and curve); 57 px ~ 13.7 pt at 300 dpi
STRIP_EFFORT_HEIGHT=${STRIP_EFFORT_HEIGHT:-0.15}   # curve height as a fraction of STRIP_WIDTH
GPU=${GPU:-3}
EXTRA=${EXTRA-"--strip-only"}            # "" -> also overlay image + side-by-side video with the effort panel
OUT=${OUT:-vis/mj_effort}

SINGLE="traj0 traj1 traj2 traj3 traj4 seq0 seq1 seq2 seq3 seq4"
BIH="btraj0 btraj1 btraj2 btraj3 btraj4 bseq0 bseq1 bseq2 bseq5 bseq4"     # bseq5 = bseq3 in the tables
mkdir -p $OUT
run() {   # $1 = baseline run, $2 = ours run
  for r in "$1" "$2"; do
    [ -f eval/sweep_ep1000/${r}_traj.npz ] && [ -f eval/sweep_ep1000/${r}_frames.npz ] || { echo "skip $1: missing $r"; return; }
  done
  echo "=== $1 vs $2"
  conda run -n mjvis env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU python mj_replay.py \
    --pair "$1" "$2" --strip --snapshots $SNAPSHOTS --effort $EFFORT --azim $AZIM --elev $ELEV \
    --strip-width $STRIP_WIDTH --strip-size $STRIP_SIZE --strip-font $STRIP_FONT --strip-effort-height $STRIP_EFFORT_HEIGHT \
    --color-dd "$COLOR_DD" --color-hx "$COLOR_HX" --color-ours "$COLOR_OURS" --out $OUT $EXTRA \
    || echo "!!! failed: $1"
}
for h in $HANDS; do
  for c in $SINGLE; do run h_dd_${c}_raw_${h} h_ours_${c}_raw_${h}; done
  for c in $BIH;    do run h_${BASE_BIH}_${c}_raw_bih_${h} h_ours_${c}_raw_bih_${h}; done
done
ls $OUT/*_strip_effort.png | wc -l