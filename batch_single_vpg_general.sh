
#!/bin/bash

echo "General running"

if [ $# -ne 10 ]; then
  echo "batch_single_vpg_general.sh <experiment-number> <experiment-info> \
  							<hidden-sizes> <lr> <epochs> <gaussian-log-std> \
							<track-length> <batch-size> <delta-time> <base-dir>"
  echo " e.g batch_single_vpg.sh 9 smallcapacity-longtrack 32,32 0.001 500 -0.5 1000 5000 0.1 /scratch/project_2005209/"
  exit 1;
fi


exp_num=$1
exp_info=$2
hidden_sizes=$3
lr=$4
epochs=$5
gaussian_log_std=$6
track_length=$7
batch_size=$8
delta_time=$9
base_logdir=${10}

log_every=50 #episodes

mkdir -p $base_logdir

exp_name=exp-$exp_num-$exp_info
mkdir -p $base_logdir/$exp_name

echo -e "hidden_sizes:$hidden_sizes \nlr: $lr \nepochs $epochs \
 	\ntrack_length: $track_length \nbatch_size: $batch_size \
 	\ngaussian_log_std: $gaussian_log_std" > $base_logdir/$exp_name/exp_parameters.txt

srun python vpg_train.py --epochs $epochs --batch-size $batch_size --lr $lr --hidden-sizes $hidden_sizes \
				--gaussian-log-std "$gaussian_log_std" --track-length $track_length  --delta-time $delta_time\
				--log-dir $base_logdir --exp-name $exp_name --log-raw-csv --log-raw-csv-every $log_every >> $base_logdir/log-${exp_name}.txt 2>&1



