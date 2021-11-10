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