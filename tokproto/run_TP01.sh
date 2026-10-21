#!/usr/bin/env bash
# TP01 queue runner (2 workers). Usage: bash tokproto/run_TP01.sh <queue_file> <log_tag>
# queue file: one line per run = "dataset neg tok lam m seed". Run from repo root.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
mkdir -p logs results/tp
Q=$1; TAG=$2
run_one() {
  read -r ds neg tok lam m seed <<< "$1"
  name="${ds}_${neg}_${tok}_l${lam}_m${m}_s${seed}"
  echo "=== start $name $(date +%T)"
  python -m tokproto.train_tp --dataset $ds --neg $neg --tok $tok --lam $lam --m $m --seed $seed > logs/TP01_run_${name}.log 2>&1
  echo "=== end $name rc=$? $(date +%T)"
}
export -f run_one
nohup bash -c "grep -v '^#' $Q | xargs -P 2 -I{} bash -c 'run_one \"{}\"'; echo ALL_DONE \$(date +%T)" > logs/TP01_${TAG}.log 2>&1 &
