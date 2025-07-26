#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="/ext_hdd/jhna/cv_grpo/DeepSeek-R1-Distill-Qwen-1.5B-MATH345-VLOO-EP5-LR1e06/checkpoint-522"

OUTPUT_DIR="/ext_hdd/jhna/cv_grpo/evals/DeepSeek-R1-Distill-Qwen-1.5B-MATH345-VLOO-EP5-LR1e06/checkpoint-522"

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
  "aime25"
  "amc23"
  "minervamath"
)


for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"

  if [ "$TASK" = "amc23" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /home/ubuntu/open-r1-cv-grpo/amc23_evals.py"
  elif [ "$TASK" = "minervamath" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /home/ubuntu/open-r1-cv-grpo/minervamath_evals.py"
  else
    PREFIX="lighteval"
    CUSTOM_TASKS=""
  fi

  lighteval vllm $MODEL_ARGS "${PREFIX}|${TASK}|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    $CUSTOM_TASKS
done
