#!/bin/bash
#SBATCH --job-name=rl-optimal-control
#SBATCH --account=Project_2005209
#SBATCH --partition=medium
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

module load pytorch/1.10 gcc/9.3.0 openmpi/4.0.3

# SAMPLE COMMAND sbatch batch_single_vpg.sh 71.0puh paper-L1500-muOffset1-rwd-log-Xbarrier-derivative-dt0.1-batchsize10k-1x32-lr0.001-60k-1mu-minus0.5logstd "32" 0.001 60000 "-0.5" 1500 10000

if [ $# -ne 9 ]; then
  echo "batch_single_vpg.sh <experiment-number> <experiment-info> \
  							<hidden-sizes> <lr> <epochs> <gaussian-log-std> \
							<track-length> <batch-size> <delta-time>"
  echo " e.g batch_single_vpg.sh 9 smallcapacity-longtrack 32,32 0.001 500 -0.5 1000 5000 0.1"
  exit 1;
fi

base_logdir="/scratch/project_2005209/velcontrol-rl-keller/runvpg/"

exp_num=$1
exp_info=$2
hidden_sizes=$3
lr=$4
epochs=$5
gaussian_log_std=$6
track_length=$7
batch_size=$8
delta_time=$9

bash -x ./batch_single_vpg_general.sh $exp_num $exp_info $hidden_sizes \
								$lr $epochs $gaussian_log_std $track_length \
								$batch_size $delta_time \
								$base_logdir

