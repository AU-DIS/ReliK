#!/usr/bin/env bash

python approach/experiment_controller.py -e TransE -d $1 -st &
sleep 10
python approach/experiment_controller.py -e DistMult -d $1 -st &
python approach/experiment_controller.py -e PairRE -d $1 -st &
python approach/experiment_controller.py -e RotatE -d $1 -st &