#!/usr/bin/env bash
# T06 B: 2 workers pull "ds arm seed" lines from logs/T06_queue.txt in priority order (flock).
# Deleting lines from the queue drops runs; no new run starts after DEADLINE (epoch seconds).
source ~/anaconda3/etc/profile.d/conda.sh && conda activate taskns
Q=logs/T06_queue.txt
DEADLINE=${DEADLINE:-0}
mkdir -p logs results/expB
cat > $Q <<'QEOF'
Amazon_eletronics unl_uni_r0.5 0
dblp unl_uni_r0.5 0
Amazon_eletronics nov_r0.5 0
dblp nov_r0.5 0
Amazon_eletronics unl_uni_r0.5 1
dblp unl_uni_r0.5 1
Amazon_eletronics nov_r0.5 1
dblp nov_r0.5 1
Amazon_eletronics unl_uni_r0.5 2
dblp unl_uni_r0.5 2
Amazon_eletronics nov_r0.5 2
dblp nov_r0.5 2
Amazon_eletronics naq_unl 2
dblp naq_unl 2
Amazon_eletronics unl_uni_r0.2 0
Amazon_eletronics nov_r0.2 0
dblp unl_uni_r0.2 0
dblp nov_r0.2 0
QEOF
for w in 1 2; do
  nohup bash -c "
    while true; do
      line=\$(flock $Q.lock -c 'head -n1 $Q; sed -i 1d $Q')
      [ -z \"\$line\" ] && break
      [ $DEADLINE -gt 0 ] && [ \$(date +%s) -ge $DEADLINE ] && { echo \"DEADLINE skip: \$line\"; continue; }
      set -- \$line
      echo \"=== \$1 \$2 s\$3 start \$(date +%T)\"
      python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 \
        || { echo \"RETRY \$line\"; python -m cape.train_proto --dataset \$1 --arm \$2 --episodes 4000 --eval_every 250 --seed \$3 || echo \"FAILED \$line\"; }
    done; echo ALL_DONE" > logs/T06_w$w.log 2>&1 &
done
