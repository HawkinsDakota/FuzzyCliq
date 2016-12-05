#!/bin/bash

python SNN.py -e tmp/input_data.csv -i tmp/edge.txt
python Cliq.py -i tmp/edge.txt -o tmp/clusters.txt
