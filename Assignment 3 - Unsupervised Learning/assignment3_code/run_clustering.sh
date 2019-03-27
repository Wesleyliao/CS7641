#!/bin/sh

# Replace 'X' below with the optimal values found
# If you want to first generate data and updated datasets, remove the "--skiprerun" flags below


python run_experiment.py --pca --fashion --dim 6  --skiprerun --verbose --threads 2 > pca-fashion-clustering.log 2>&1
python run_experiment.py --pca --nba   --dim 4  --skiprerun --verbose --threads 2 > pca-nba-clustering.log   2>&1
python run_experiment.py --ica --fashion --dim 8  --skiprerun --verbose --threads 2 > ica-fashion-clustering.log 2>&1
python run_experiment.py --ica --nba   --dim 9  --skiprerun --verbose --threads 2 > ica-nba-clustering.log   2>&1
python run_experiment.py --rp  --fashion --dim 10  --skiprerun --verbose --threads 2 > rp-fashion-clustering.log  2>&1
python run_experiment.py --rp  --nba   --dim 10  --skiprerun --verbose --threads 2 > rp-nba-clustering.log    2>&1
python run_experiment.py --rf  --fashion --dim 15  --skiprerun --verbose --threads 2 > rf-fashion-clustering.log  2>&1
python run_experiment.py --rf  --nba   --dim 7  --skiprerun --verbose --threads 2 > rf-nba-clustering.log    2>&1
