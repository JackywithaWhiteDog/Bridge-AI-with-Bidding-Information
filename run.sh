#!bin/bash

DEVICE=6

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --agent_a_name RNN \
        --agent_a_generation_mode random \
        --agent_a_model_path ./ckpt/hands/20220605_2200/hands-epoch=12-valid_card_acc=0.47-valid_loss=0.26.ckpt \
    --agent_b_name DDS \
    --num_games 1000 \
    --output_file ./logs/RNN_10_DDS_1k.json

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --agent_a_name MCTS \
    --agent_b_name DDS \
    --num_games 1000 \
    --output_file ./logs/MCTS_10_DDS_1k.json
