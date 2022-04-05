#!/bin/bash
#SBATCH --job-name=rl-optimal-control
#SBATCH --account=Project_2004564
#SBATCH --partition=small
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

module load pytorch/nvidia-19.11-py3 gcc/9.1.0 intel-mpi/18.0.5


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

echo "top 10 successful records in $models_dir/all_run_models_figs_sorted_times.txt"
grep -Fw "Success" $models_dir/all_run_models_figs_sorted_times.txt > $models_dir/all_run_models_figs_sorted_times_only_success.txt
head $models_dir/all_run_models_figs_sorted_times_only_success.txt 
 
