#!/bin/bash


if [ $# -ne 2 ]; then
  echo "fetch_results.sh <puhti|mahti> <experiment-dir>"
  echo " e.g fetch_results.sh puhti exp-20-rwd-z.invtime-2x32-lr0.001-30k-10mu-minus0.5logstd"
  exit 1;
fi

base_logdir="/scratch/project_*/velcontrol-rl-keller/runvpg/"

exp_dir=$2
server=$1

mkdir -p $exp_dir $exp_dir/csv $exp_dir/${exp_dir}_s0

echo $exp_dir
rsync -r --exclude '*.pkl' --exclude '*.pt' $server:/${base_logdir}/$exp_dir/${exp_dir}_s0/ $exp_dir/${exp_dir}_s0/
python vpg_plot_train_history.py $exp_dir/${exp_dir}_s0
# python vpg_run_model.py $exp_dir/${exp_dir}_s0/pyt_save/model.pt --num-episodes 5 --log-raw-csv

ssh $server "ls /${base_logdir}/$exp_dir/csv | sort -t "_" -k 2 -n -r | head -n 20" | \
	while read line; do
		if [ ! -f $exp_dir/csv/$line ]; then
			scp $server:/${base_logdir}/$exp_dir/csv/$line $exp_dir/csv/;
		fi 
	done;
python turn_csv_to_fig.py --num-last-files 20 $exp_dir
