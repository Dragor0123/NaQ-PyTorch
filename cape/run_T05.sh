#!/usr/bin/env bash
# T05 B: one process per dataset, runs in priority order. Run from repo root.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
RUNS=${RUNS:-"naq_unl:0 naq_unl:1 lr_soft:0 lr_soft:1 naq_hard:0 naq_easy:0"}
mkdir -p logs results/expB
for ds in Amazon_eletronics dblp; do
  nohup bash -c "
    for r in $RUNS; do arm=\${r%%:*}; seed=\${r##*:}
      echo \"=== $ds \$arm s\$seed start \$(date +%T)\"
      python -m cape.train_proto --dataset $ds --arm \$arm --episodes 4000 --eval_every 250 --seed \$seed \
        || { echo \"RETRY \$arm s\$seed\"; python -m cape.train_proto --dataset $ds --arm \$arm --episodes 4000 --eval_every 250 --seed \$seed || echo \"FAILED \$arm s\$seed\"; }
    done; echo ALL_DONE" > logs/T05_$ds.log 2>&1 &
done
