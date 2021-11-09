#!/bin/bash
#SBATCH --job-name=rl-optimal-control
#SBATCH --account=Project_2004564
#SBATCH --partition=small
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

# script to read exps.txt file and run the experiments in it.

module load pytorch/nvidia-19.11-py3 gcc/9.1.0 intel-mpi/18.0.5
#source fenv/bin/activate

if [ $# -ne 6 ]; then
  echo "batch_single_vpg.sh <experiment-number> <experiment-info> <hidden-sizes> <lr> <epochs> <gaussian-log-std>"
  echo " e.g batch_single_vpg.sh 9 smallcapacity-longtrack 32,32 0.001 500 -0.5"
  exit 1;
fi

base_logdir="/scratch/project_2004564/velcontrol-rl-keller/runvpg/"
mkdir -p $base_logdir
exp_num=$1
exp_info=$2
hidden_sizes=$3
lr=$4
epochs=$5
track_length=400
batch_size=5000
gaussian_log_std=$6
exp_name=exp-$exp_num-$exp_info
mkdir -p $base_logdir/$exp_name
echo -e "hidden_sizes:$hidden_sizes \nlr: $lr \nepochs $epochs \
 	\ntrack_length: $track_length \nbatch_size: $batch_size \
 	\ngaussian_log_std: $gaussian_log_std" > $base_logdir/$exp_name/exp_parameters.txt

srun python vpg_train.py --epochs $epochs --batch-size $batch_size --lr $lr --hidden-sizes $hidden_sizes \
				--gaussian-log-std "$gaussian_log_std" --track-length $track_length \
				--log-dir $base_logdir --exp-name $exp_name --log-raw-csv --log-raw-csv-every 200 >> $base_logdir/log-${exp_name}.txt 2>&1
python vpg_plot_train_history.py $base_logdir/$exp_name/${exp_name}_s0
python vpg_run_model.py $base_logdir/$exp_name/${exp_name}_s0/pyt_save/model.pt --num-episodes 10


