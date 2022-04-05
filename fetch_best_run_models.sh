#!/bin/bash


if [ $# -ne 3 ]; then
  echo "fetch_results.sh <puhti|mahti> <num-files> <experiment-dir>"
  echo " e.g fetch_results.sh puhti 20 exp-20-rwd-z.invtime-2x32-lr0.001-30k-10mu-minus0.5logstd"
  exit 1;
fi

base_logdir="/scratch/project_*/velcontrol-rl-keller/runvpg/"

server=$1
num_files=$2
exp_dir=$3
out_dir=$exp_dir/run_all_models
mkdir -p $out_dir

echo $exp_dir



scp $server:/${base_logdir}/$exp_dir/${exp_dir}_s0/pyt_save/all_run_models_figs_sorted_times.txt $out_dir/
scp $server:/${base_logdir}/$exp_dir/${exp_dir}_s0/pyt_save/all_run_models_figs_sorted_times_only_success.txt $out_dir/

head -n $num_files $out_dir/all_run_models_figs_sorted_times_only_success.txt | cut -d ',' -f 1 | sed "s|/csv/log_1.csv||g" | 
	while read line; do
		scp -r $server:/$line $out_dir;
	done; 
# cat $out_dir/all_run_models_figs_sorted_times_only_success.txt | while read line; do  done;

# python vpg_plot_train_history.py $exp_dir/${exp_dir}_s0
# # python vpg_run_model.py $exp_dir/${exp_dir}_s0/pyt_save/model.pt --num-episodes 5 --log-raw-csv

# ssh $server "ls /${base_logdir}/$exp_dir/csv | sort -t "_" -k 2 -n -r | head -n $num_last_csv_logs" | \
# 	while read line; do
# 		if [ ! -f $exp_dir/csv/$line ]; then
# 			scp $server:/${base_logdir}/$exp_dir/csv/$line $exp_dir/csv/;
# 		fi 
# 	done;
# python turn_csv_to_fig.py --plot-kellers --plot-kellers-track-length $track_length --num-last-files $num_last_csv_logs $exp_dir
