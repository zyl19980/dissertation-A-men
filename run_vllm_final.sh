#!/bin/bash

# 加载 Singularity
module load singularity

# 1. 基础配置
MODEL_PATH="/scratch/users/ntu/yzheng05/model_weights/Qwen2.5-7B-Instruct"
SIF_IMAGE="/home/users/ntu/yzheng05/scratch/vllm_image/agent_vllm_v2.sif"
PORT=10203

# 2. 预创建并重定向所有缓存路径 (解决 500 报错的关键)
CACHE_ROOT="/scratch/users/ntu/yzheng05/.cache"
mkdir -p $CACHE_ROOT/numba $CACHE_ROOT/outlines $CACHE_ROOT/huggingface

export NUMBA_CACHE_DIR=$CACHE_ROOT/numba
export OUTLINES_CACHE_DIR=$CACHE_ROOT/outlines
export HF_HOME=$CACHE_ROOT/huggingface

# 3. HPC 环境兼容性补丁
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export LC_ALL=C.UTF-8

# 4. 启动 vLLM 服务
nohup singularity run --nv \
  -B /dev/shm:/dev/shm \
  -B /scratch/users/ntu/yzheng05:/scratch \
  -B $MODEL_PATH:/model \
  $SIF_IMAGE \
  --model /model \
  --served-model-name qwen \
  --port $PORT \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --enforce-eager \
  --trust-remote-code \
  > server.log 2>&1 &