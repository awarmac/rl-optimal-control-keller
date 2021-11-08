#!/bin/bash


if [ $# -ne 1 ]; then
  echo "fetch_results.sh <experiment-dir>"
  echo " e.g fetch_results.sh exp-20-rwd-z.invtime-2x32-lr0.001-30k-10mu-minus0.5logstd"
  exit 1;
fi

base_logdir="/scratch/project_2004564/velcontrol-rl-keller/runvpg/"
exp_dir=$1
mkdir -p $exp_dir $exp_dir/csv $exp_dir/${exp_dir}_s0

echo $exp_dir
rsync -r puhti:/${base_logdir}/$exp_dir/${exp_dir}_s0/ $exp_dir/${exp_dir}_s0/
python vpg_plot_train_history.py $exp_dir/${exp_dir}_s0
# python vpg_run_model.py $exp_dir/${exp_dir}_s0/pyt_save/model.pt --num-episodes 5 --log-raw-csv

ssh puhti ls -t /${base_logdir}/$exp_dir/csv | head -n 20 | while read line; do scp puhti:/${base_logdir}/$exp_dir/csv/$line $exp_dir/csv/; done;
python turn_csv_to_fig.py --num-last-files 20 $exp_dir
