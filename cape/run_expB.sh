#!/usr/bin/env bash
# Exp B: one background process per dataset, arms run sequentially. Run from repo root.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
EPISODES=${EPISODES:-4000}
ARMS="naq_feat naq_diff lr_only cape_lite oracle oracle_base"
mkdir -p logs results/expB
for ds in Amazon_eletronics dblp; do
  nohup bash -c "for arm in $ARMS; do echo \"=== \$arm start \$(date +%T)\"; python -m cape.train_proto --dataset $ds --arm \$arm --episodes $EPISODES --eval_every 250 --seed 0 || exit 1; done; echo ALL_DONE" > logs/expB_$ds.log 2>&1 &
done
