#!/usr/bin/env bash
# T07 B: 2 workers pull "ds arm seed [tag]" lines from logs/T07_queue.txt in priority order (flock).
# Deleting lines drops runs; no new run starts after DEADLINE (epoch s). All runs save embeddings.
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
Q=logs/T07_queue.txt
DEADLINE=${DEADLINE:-0}
mkdir -p logs results/expB
E=Amazon_eletronics; D=dblp; C=Amazon_clothing
{
for s in 0 1 2; do for a in nov2_t1_r0.5 nov2_t2_r0.5; do echo "$E $a $s"; echo "$D $a $s"; done; done   # P1
for s in 0 1; do echo "$E mix_alt_r0.5 $s"; echo "$D mix_alt_r0.5 $s"; done                            # P2
echo "$E naq_feat 0 _emb"; echo "$D naq_diff 0 _emb"; echo "$E naq_unl 0 _emb"                          # P3
for s in 0 1; do echo "$E nov2_t1_r0.2 $s"; echo "$D nov2_t1_r0.2 $s"; done                            # P4
for s in 0 1; do echo "$C naq_feat $s"; echo "$C nov2_t1_r0.5 $s"; done                                # P5
echo "$E nov2_t1_r1.0 0"; echo "$D nov2_t1_r1.0 0"                                                      # P6
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
    done; echo ALL_DONE" > logs/T07_w$w.log 2>&1 &
done
