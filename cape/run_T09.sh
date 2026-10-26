#!/usr/bin/env bash
# T09: 2 workers pull "ds arm seed" lines from logs/T09_queue.txt in priority order (flock).
# Existing results/expB/{ds}_{arm}_s{seed}.json are not queued. Deleting lines drops runs; no new run starts after DEADLINE (epoch s).
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
Q=logs/T09_queue.txt
DEADLINE=${DEADLINE:-0}
SAVE_EMB=${SAVE_EMB---save_emb}
mkdir -p logs results/expB
E=Amazon_eletronics; D=dblp; C=Amazon_clothing
{
for s in 0 1 2; do for a in naq_feat nov2_t1_r0.5 mix_alt_r0.5 naq_mix_r0.5; do echo "$C $a $s"; done; done        # P1
for s in 2 3 4; do for ds in $E $D; do for a in mix_alt_r0.5 naq_mix_r0.5; do echo "$ds $a $s"; done; done; done   # P2
for s in 3 4; do echo "$E naq_feat $s"; echo "$E nov2_t1_r0.5 $s"; echo "$D naq_diff $s"; echo "$D nov2_t1_r0.5 $s"; done  # P3
for s in 0 1 2; do for a in naq_unl sup_only_r0.5 naq_diff; do echo "$C $a $s"; done; done                          # P4
for s in 2 3 4; do echo "$E sup_only_r0.5 $s"; echo "$D sup_only_r0.5 $s"; done                                    # P5
} | while read ds a s; do [ -f results/expB/${ds}_${a}_s${s}.json ] && echo "exists, not queued: $ds $a $s" >&2 || echo "$ds $a $s"; done > $Q
wc -l < $Q
for w in 1 2; do
  nohup bash -c "
    while true; do
      line=\$(flock $Q.lock -c 'head -n1 $Q; sed -i 1d $Q')
      [ -z \"\$line\" ] && break
      [ $DEADLINE -gt 0 ] && [ \$(date +%s) -ge $DEADLINE ] && { echo \"DEADLINE skip: \$line\"; continue; }
      set -- \$line
      echo \"=== \$line start \$(date +%T)\"
      python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 $SAVE_EMB \
        || { echo \"RETRY \$line\"; python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 $SAVE_EMB || echo \"FAILED \$line\"; }
      echo \"=== \$line end \$(date +%T)\"
    done; echo ALL_DONE" > logs/T09_w$w.log 2>&1 &
done
