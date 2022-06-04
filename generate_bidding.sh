#!bin/bash

set -e

python ./bid_generator.py \
    --seed 1123 \
    --data_size 10000 \
    --output_file ./data/train.txt

python ./bid_generator.py \
    --seed 2022 \
    --data_size 10000 \
    --output_file ./data/valid.txt

python ./bid_generator.py \
    --seed 530 \
    --data_size 10000 \
    --output_file ./data/test.txt
