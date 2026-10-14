#!/usr/bin/env bash
# T08 B: 2 workers pull "ds arm seed [tag]" lines from logs/T08_queue.txt in priority order (flock).
# Deleting lines drops runs; no new run starts after DEADLINE (epoch s). All runs save embeddings.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
Q=logs/T08_queue.txt
DEADLINE=${DEADLINE:-0}
mkdir -p logs results/expB
E=Amazon_eletronics; D=dblp; C=Amazon_clothing
{
for s in 0 1; do echo "$E naq_mix_r0.5 $s"; echo "$D naq_mix_r0.5 $s"; done                               # P1
for s in 0 1 2; do echo "$E nov2c_t1_r0.5 $s"; echo "$D nov2c_t1_r0.5 $s"; done                          # P2
for s in 0 1; do echo "$E sup_only_r0.5 $s"; echo "$D sup_only_r0.5 $s"; done                             # P3
for a in mix_alt_p10_r0.5 mix_alt_p50_r0.5; do echo "$E $a 0"; echo "$D $a 0"; done                      # P4
} > $Q
for w in 1 2; do
  nohup bash -c "
    while true; do
      line=\$(flock $Q.lock -c 'head -n1 $Q; sed -i 1d $Q')
      [ -z \"\$line\" ] && break
      [ $DEADLINE -gt 0 ] && [ \$(date +%s) -ge $DEADLINE ] && { echo \"DEADLINE skip: \$line\"; continue; }
      set -- \$line
      tag=\${4:+--tag \$4}
      echo \"=== \$line start \$(date +%T)\"
      python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 --save_emb \$tag \
        || { echo \"RETRY \$line\"; python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 --save_emb \$tag || echo \"FAILED \$line\"; }
    done; echo ALL_DONE" > logs/T08_w$w.log 2>&1 &
done
