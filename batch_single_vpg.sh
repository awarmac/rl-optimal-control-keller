#!/bin/bash
#SBATCH --job-name=rl-optimal-control
#SBATCH --account=Project_2005209
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# script to read exps.txt file and run the experiments in it.

module load pytorch/1.10 gcc/9.3.0 openmpi/4.0.3
#source fenv/bin/activate

if [ $# -ne 8 ]; then
  echo "batch_single_vpg.sh <experiment-number> <experiment-info> <hidden-sizes> <lr> <epochs> <gaussian-log-std> <track-length> <batch-size>"
  echo " e.g batch_single_vpg.sh 9 smallcapacity-longtrack 32,32 0.001 500 -0.5 1000 5000"
  exit 1;
fi

base_logdir="/scratch/project_2005209/velcontrol-rl-keller/runvpg/"
mkdir -p $base_logdir

exp_num=$1
exp_info=$2
hidden_sizes=$3
lr=$4
epochs=$5
gaussian_log_std=$6
track_length=$7
batch_size=$8

log_every=50 #episodes

exp_name=exp-$exp_num-$exp_info
mkdir -p $base_logdir/$exp_name
echo -e "hidden_sizes:$hidden_sizes \nlr: $lr \nepochs $epochs \
 	\ntrack_length: $track_length \nbatch_size: $batch_size \
 	\ngaussian_log_std: $gaussian_log_std" > $base_logdir/$exp_name/exp_parameters.txt

srun python vpg_train.py --epochs $epochs --batch-size $batch_size --lr $lr --hidden-sizes $hidden_sizes \
				--gaussian-log-std "$gaussian_log_std" --track-length $track_length \
				--log-dir $base_logdir --exp-name $exp_name --log-raw-csv --log-raw-csv-every $log_every >> $base_logdir/log-${exp_name}.txt 2>&1
python vpg_plot_train_history.py $base_logdir/$exp_name/${exp_name}_s0
python vpg_run_model.py $base_logdir/$exp_name/${exp_name}_s0/pyt_save/model.pt --num-episodes 10


