#!/bin/bash
# Run a queue of train_single.sh jobs sequentially. One line per job:  KEY=value ... <lab> [<lab> ...]
#   DEXHAND=inspire METHOD=ours    raw
#   DEXHAND=shadow                 raw f8
# Blank lines and lines starting with # are skipped. The file is read line by line, so you can
# append jobs while it is running (edits to lines already passed have no effect).
#
#   CUDA_VISIBLE_DEVICES=2 bash queue_single.sh queue_A.txt
set -u
cd "$(dirname "$0")"
[ $# -eq 1 ] && [ -f "$1" ] || { echo "usage: bash queue_single.sh <queue file>"; exit 1; }

while read -r line || [ -n "$line" ]; do
  [[ -z "$line" || "$line" == \#* ]] && continue
  vars=$(echo "$line" | grep -oE '[A-Z_]+=[^ ]+' | tr '\n' ' ')
  labs=$(echo "$line" | sed -E 's/([A-Z_]+=[^ ]+[[:space:]]*)+//')
  [ -z "$labs" ] && { echo "skip (no labs): $line"; continue; }
  echo "################  $(date '+%m-%d %H:%M')  $vars train_single.sh $labs"
  env $vars bash train_single.sh $labs || echo "!!! job failed: $line"
done < "$1"
echo "################  queue $1 finished $(date '+%m-%d %H:%M')"
