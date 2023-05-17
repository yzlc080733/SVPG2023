for REP in {0..2}; do
for LR in 0.001 0.0001; do
for ENTROPY in 0.02 0.2 2; do
for PPO_E in 10; do
for PPO_CLIP in 0.2; do
for DEL_CONN in none; do
for MODEL in snnbptt; do
tsp python run_PPO.py --model $MODEL --lr $LR --entropy $ENTROPY --cuda 0 --thread 10 --PPO_epochs $PPO_E --eps_clip $PPO_CLIP --rwta_del_connection $DEL_CONN --snn_num_steps 15 --seed $REP;
# qsub ~/bin/submit_4.sh "run_PPO.py --model $MODEL --lr $LR --entropy $ENTROPY --cuda -1 --PPO_epochs $PPO_E --eps_clip $PPO_CLIP --rwta_del_connection $DEL_CONN --snn_num_steps 15 --seed $REP --ignore_checkpoint";
# qsub ~/bin/submit_gpu_ARC4.sh "run_PPO.py --model $MODEL --lr $LR --entropy $ENTROPY --cuda 0 --PPO_epochs $PPO_E --eps_clip $PPO_CLIP --rwta_del_connection $DEL_CONN --snn_num_steps 15 --seed $REP --ignore_checkpoint";
# sbatch ~/bin/slurm6.sh "run_PPO.py --model $MODEL --lr $LR --entropy $ENTROPY --cuda -1 --PPO_epochs $PPO_E --eps_clip $PPO_CLIP --rwta_del_connection $DEL_CONN --snn_num_steps 15 --seed $REP --ignore_checkpoint";
done;
done;
done;
done;
done;
done;
done;
