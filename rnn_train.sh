#!bin/bash

DEVICE=4

# NULL result
# Null Result: (0.6254173517227173, 0.3752250075340271, 0.0)
# Null Result (out of hints): (0.4004518389701843, 0.0, 0.0)

# Version 7
# # encode -1 for card not hold
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 11
# remove dropout in fc from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 12
# add layer to GRU from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 2 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 15
# encode pass as last action of partner from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 16
# encode pass as last action from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 17
# Modify batch size from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 128 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 18
# Modify batch size from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 512 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 19
# Modify lr from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional


# Version 21
# Modify lr from version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 1e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 22
# Encode the known card in unknown hands to -1 from version 7 and fix generating & accuracy function
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 23
# Rerun version 7
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 24
# modify dropout in fc from version 22
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 25
# add layer to GRU from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 2 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 26
# encode pass as last action of partner from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 27
# encode pass as last action from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 28
# add 1 layer to fc from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 29
# enlarge GRU hidden size from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --gru_hidden_size 64 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 30
# add hand fc from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 31
# single direction from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208

# Version 32
# modify lr from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 1e-2 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208
#     --bidirectional

# Version 33
# modify lr from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 1e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208
#     --bidirectional

# Version 34
# only focus on masked part from version 24
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 35
# modify loss weight from version 34
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 1 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 36
# add layers to gru from version 22
# CUDA_VISIBLE_DEVICES=${DEVICE} \
# python ./bridge/rnn/train.py \
#     --lr 2e-3 \
#     --batch_size 256 \
#     --num_layers 3 \
#     --dropout 0 \
#     --hand_hidden_size 208 \
#     --bidirectional

# Version 37
# Use 100k data from version 24 (10k data)
CUDA_VISIBLE_DEVICES=${DEVICE} \
python ./bridge/rnn/train.py \
    --lr 2e-3 \
    --batch_size 256 \
    --num_layers 1 \
    --dropout 0 \
    --hand_hidden_size 208 \
    --bidirectional \
    --data_dir ./data/100k
