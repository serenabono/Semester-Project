#!/bin/bash
source /itet-stor/sebono/net_scratch/conda/etc/profile.d/conda.sh
conda activate py36
python -m main.py --training True --epochs 60
python -m main.py --testing True