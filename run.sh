#!bin/bash

DEVICE=6

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py --agent_a MCTS --agent_b DDS --max_threads 1
