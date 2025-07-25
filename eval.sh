#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MODEL_PATH="/home/ubuntu/jhna-east1/cv_grpo/Qwen2.5-MATH-7B-MATH345-GRPO-EP20-LR2e06/checkpoint-522"

OUTPUT_DIR="/ext_hdd/jhna/cv_grpo/evals/Qwen2.5-MATH-7B-MATH345-GRPO-EP20-LR2e06/checkpoint-522"

DTYPE="bfloat16"  
MAX_LEN=32768
MAX_NEW_TOKENS=2048
TEMP=0.7
TOP_P=0.9
GPU_UTIL=0.8

MODEL_ARGS="model_name=$MODEL_PATH,dtype=$DTYPE,max_model_length=$MAX_LEN,gpu_memory_utilization=$GPU_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

TASKS=(
  "gsm8k"
  "math_500"
  "agieval:sat-math"
  "aime24"
)

for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"
  lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
done
