#!/usr/bin/env bash
# T04: seeds 1,2 (A, 4000 ep) then one 8000-ep run (B) per process. Run from repo root.
# B is skipped if logs/T04_skipB exists when a process reaches it.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
mkdir -p logs results/expB
run_proc() {  # $1=dataset  $2="arms for A"  $3=B arm (Amazon_eletronics, 8000 ep)  $4=log name
  nohup bash -c "
    for seed in 1 2; do for arm in $2; do
      echo \"=== A $1 \$arm s\$seed start \$(date +%T)\"
      python -m cape.train_proto --dataset $1 --arm \$arm --episodes 4000 --eval_every 250 --seed \$seed || exit 1
    done; done
    if [ -e logs/T04_skipB ]; then echo 'B SKIPPED'; else
      echo \"=== B Amazon_eletronics $3 e8000 start \$(date +%T)\"
      python -m cape.train_proto --dataset Amazon_eletronics --arm $3 --episodes 8000 --eval_every 250 --seed 0 || exit 1
    fi
    echo ALL_DONE" > logs/T04_$4.log 2>&1 &
}
run_proc Amazon_eletronics "naq_feat lr_only cape_lite" naq_feat p1
run_proc dblp "naq_diff lr_only cape_lite" cape_lite p2
