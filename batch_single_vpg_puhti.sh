#!/bin/bash
#SBATCH --job-name=rl-optimal-control
#SBATCH --account=Project_2004564
#SBATCH --partition=longrun
#SBATCH --time=14-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

module load pytorch/nvidia-19.11-py3 gcc/9.1.0 intel-mpi/18.0.5

if [ $# -ne 9 ]; then
  echo "batch_single_vpg.sh <experiment-number> <experiment-info> <hidden-sizes> \
  							<lr> <epochs> <gaussian-log-std> <track-length> \
							<batch-size> <delta-time>"
  echo " e.g batch_single_vpg.sh 9 smallcapacity-longtrack 32,32 0.001 500 -0.5 1000 5000 0.1"
  exit 1;
fi

base_logdir="/scratch/project_2004564/velcontrol-rl-keller/runvpg/"


exp_num=$1
exp_info=$2
hidden_sizes=$3
lr=$4
epochs=$5
gaussian_log_std=$6
track_length=$7
batch_size=$8
delta_time=$9

./batch_single_vpg_general.sh $exp_num $exp_info $hidden_sizes \
								$lr $epochs $gaussian_log_std $track_length \
								$batch_size $delta_time \
								$base_logdir

