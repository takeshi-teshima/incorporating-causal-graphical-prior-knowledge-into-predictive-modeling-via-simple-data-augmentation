#!/usr/bin/env bash
dataset=sachs

repeats_from=1
repeats_to=10

for train_size in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85; do
    for id in `seq $repeats_from $repeats_to`; do
        d="$dataset"_$id
        method=base_xgb
        python run_experiment.py -m parallelization.data_run_id=$d method=$method data.train_size=$train_size recording.table_name=main_$dataset data=$dataset

        method=proposed
        python run_experiment.py -m parallelization.data_run_id=$d method=$method data.train_size=$train_size recording.table_name=main_$dataset data=$dataset
    done
done

repeats_from=11
repeats_to=20

for train_size in 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85; do
    for id in `seq $repeats_from $repeats_to`; do
        d="$dataset"_$id
        method=base_xgb
        python run_experiment.py -m parallelization.data_run_id=$d method=$method data.train_size=$train_size recording.table_name=main_$dataset data=$dataset

        method=proposed
        python run_experiment.py -m parallelization.data_run_id=$d method=$method data.train_size=$train_size recording.table_name=main_$dataset data=$dataset
    done
done
