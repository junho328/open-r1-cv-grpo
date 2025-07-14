#!/bin/bash

export VLLM_WORKER_MULTIPROC_METHOD=spawn


MODEL_PATH="/ext_hdd/jhna/cv_grpo/Qwen2.5-1.5B-OPENR1-RLOO"

OUTPUT_DIR="/ext_hdd/jhna/cv_grpo/evals/Qwen2.5-1.5B-OPENR1-RLOO"

DTYPE="bfloat16"  
MAX_LEN=32768
MAX_NEW_TOKENS=2048
TEMP=0.7
TOP_P=0.9
GPU_UTIL=0.8

MODEL_ARGS="model_name=$MODEL_PATH,dtype=$DTYPE,max_model_length=$MAX_LEN,gpu_memory_utilization=$GPU_UTIL,generation_parameters={max_new_tokens:$MAX_NEW_TOKENS,temperature:$TEMP,top_p:$TOP_P}"

TASKS=(
#   "aime24"
  "math_500"
#   "gpqa:diamond"
)

for TASK in "${TASKS[@]}"; do
  echo "Running task: $TASK"
  lighteval vllm $MODEL_ARGS "lighteval|$TASK|0|0" \
    --use-chat-template \
    --output-dir "$OUTPUT_DIR"
done

# echo "Running task: LiveCodeBench"
# lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
#   --use-chat-template \
#   --output-dir "$OUTPUT_DIR"