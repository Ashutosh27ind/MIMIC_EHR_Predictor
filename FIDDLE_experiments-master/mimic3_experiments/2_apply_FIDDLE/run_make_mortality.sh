#!/bin/bash
set -euxo pipefail

# Ashutosh giving full qualified path : 
#export PYTHONPATH="../../"
export PYTHONPATH="/home/azureuser/cloudfiles/code/Users/ashutoshind2017/FIDDLE-master"

DATAPATH=$(python -c "import yaml;print(yaml.full_load(open('../config.yaml'))['data_path']);")
mkdir -p log

# Ashutosh added missing output_dir and updated param name data_fname with data_path :
OUTCOME=mortality
T=48.0
dt=1.0

python -m FIDDLE.run \
    --data_fname="$DATAPATH/features/outcome=$OUTCOME,T=$T,dt=$dt/input_data.p" \
    --population="$DATAPATH/population/pop.mortality_benchmark.csv" \
    --output_dir="$DATAPATH/output/" \
    --T=48.0 \
    --dt=1.0 \
    --theta_1=0.001 \
    --theta_2=0.001 \
    --theta_freq=1 \
    --stats_functions 'min' 'max' 'mean' \
    > >(tee 'log/benchmark,outcome=mortality,T=48.0,dt=1.0.out') \
    2> >(tee 'log/benchmark,outcome=mortality,T=48.0,dt=1.0.err' >&2)
