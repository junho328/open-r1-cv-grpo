#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="Qwen/Qwen2.5-Math-7B-Instruct"

OUTPUT_DIR="/workspace/output/Qwen2.5-Math-7B-Instruct"

DTYPE="bfloat16"  
MAX_LEN=32768
MAX_NEW_TOKENS=2048
TEMP=0.7
TOP_P=0.9
GPU_UTIL=0.9

MODEL_ARGS="model_name=$MODEL_PATH,dtype=$DTYPE,max_model_length=$MAX_LEN,gpu_memory_utilization=$GPU_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

TASKS=(
  # "gsm8k"
  # "math_500"
  # "collegemath"
  # "aime24"
  # "aime25"
  # "amc23"
  # "minervamath"
  "mmlu_stem"
  # "gpqa"
)


for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"

  if [ "$TASK" = "amc23" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /{PATH}/open-r1-cv-grpo/amc23_evals.py"
  elif [ "$TASK" = "minervamath" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /{PATH}/open-r1-cv-grpo/minervamath_evals.py"
  elif [ "$TASK" = "collegemath" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /{PATH}/open-r1-cv-grpo/collegemath_evals.py"
  elif [ "$TASK" = "mmlu_stem" ]; then
    PREFIX="community"
    CUSTOM_TASKS="--custom-tasks /workspace/open-r1-cv-grpo/mmlu_stem_evals.py"
  else
    PREFIX="lighteval"
    CUSTOM_TASKS=""
  fi

  if [ "$TASK" = "mmlu_stem" ]; then
    lighteval vllm $MODEL_ARGS "${PREFIX}|${TASK}|5|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR" \
    $CUSTOM_TASKS
  else
    lighteval vllm $MODEL_ARGS "${PREFIX}|${TASK}|0|0" \
      --use-chat-template \
      --output-dir "$OUTPUT_DIR" \
      $CUSTOM_TASKS
  fi

done
