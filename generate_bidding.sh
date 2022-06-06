#!bin/bash

set -e

python ./bid_generator.py \
    --seed 1123 \
    --data_size 100000 \
    --output_file ./data/train_100k.txt

python ./bid_generator.py \
    --seed 2022 \
    --data_size 100000 \
    --output_file ./data/100k/valid.txt

python ./bid_generator.py \
    --seed 530 \
    --data_size 100000 \
    --output_file ./data/100k/test.txt
