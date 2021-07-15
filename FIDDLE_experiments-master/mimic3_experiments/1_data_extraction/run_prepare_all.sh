#!/bin/bash
set -euxo pipefail

# Ashutosh commented as we need only for mortality :#

#python prepare_input.py --outcome=ARF   --T=4  --dt=1
#python prepare_input.py --outcome=ARF   --T=12 --dt=1
#python prepare_input.py --outcome=Shock --T=4  --dt=1
#python prepare_input.py --outcome=Shock --T=12 --dt=1

# Ashutosh changed to correct T and dt :
python prepare_input.py --outcome=mortality --T=48.0 --dt=1.0
#mv ../data/processed/features/outcome=mortality,T=48.0,dt=1.0 ../data/processed/features/benchmark,outcome=mortality,T=48.0,dt=1.0

# ver2.0 Author suggestion:
cp -r ../data/processed/features/outcome=mortality,T=48.0,dt=1.0 ../data/processed/features/benchmark,outcome=mortality,T=48.0,dt=1.0

