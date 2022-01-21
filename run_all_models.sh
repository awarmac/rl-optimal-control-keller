#!/bin/bash


if [ $# -ne 2 ]; then
  echo "fetch_results.sh <models-dir> <track-length>"
  echo " e.g fetch_results.sh exp-20/pyt_save 400"
  exit 1;
fi

models_dir=$1
track_length=$2

all_figs_dir=$models_dir/all_run_models_figs/
mkdir -p $all_figs_dir

python vpg_run_model.py --num-episodes 1 --track-length $track_length \
							--log-dir $models_dir/run_models_dirs --all-figs-dir $all_figs_dir \
							$models_dir

echo $(ls $all_figs_dir | wc -l) "figures have been created."

find $models_dir/run_models_dirs -name log_1.csv | while read f; do ll=$(tail -n 1 $f); echo $f","$ll; done | tr -d "\"" | sort -t "," -k 2 -g > $models_dir/all_run_models_figs_sorted_times.txt
echo "top 5 records in $models_dir/all_run_models_figs_sorted_times.txt"
head -n 5 $models_dir/all_run_models_figs_sorted_times.txt