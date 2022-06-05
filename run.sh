#!bin/bash

DEVICE=6

CUDA_VISIBLE_DEVICES=${DEVICE} \
python main.py \
    --agent_a_name RNN \
        --agent_a_n 10 \
        --agent_a_model_path ./ckpt/hands/20220602_0740/hands-epoch=40-valid_card_acc=0.46-valid_loss=0.28.ckpt \
    --agent_b_name MCTS \
        --agent_b_n 1 \
