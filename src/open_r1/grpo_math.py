# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys

import datasets
from datasets import load_dataset, DatasetDict
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config
from cv_grpo.cv_grpo_trainer import GRPOTrainer


logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    
    os.environ["WANDB_PROJECT"] = script_args.project_name

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    # dataset = get_dataset(script_args)
    
    logger.info(f"Loading dataset: {script_args.dataset_name}")
    dataset = load_dataset(script_args.dataset_name, split="test")
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=100, seed=42)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)
    
    def build_qwen_fewshot_messages(question:str):

        system_message = f"""
    You are a helpful assistant that solves math problems step-by-step with clear reasoning.
    Your task:
    1. Break down the solution into clear, logical steps.
    2. Label each step explicitly as "Step 1)", "Step 2)", "Step 3)", etc.
    3. Always include a "Final answer:" section.
    4. The **final answer** must be written strictly on the line **immediately following** "Final answer:" and must start with "####".
    5. After "####", write **only the final result** without any extra words, symbols, or punctuation.
    6. Absolutely do not include any additional text or commentary outside these steps or in the final answer."""

        messages = [
            {"role": "system", "content": system_message},

            {"role": "user", "content": """# Example 1
    **Question:**
    Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did she sell in May?"""},
            {"role": "assistant", "content": """Step 1) Natalia sold 48 clips in April.
    Step 2) In May, she sold half as many as in April.
    Step 3) To find the number of clips sold in May, we divide:
            48 ÷ 2 = 24
    Step 4) Therefore, Natalia sold 24 clips in May.

    **Final answer:
    ####24**"""},

            {"role": "user", "content": """# Example 2
    **Question:**
    Weng earns $12 an hour for babysitting. Yesterday, she worked for 50 minutes. How much did she earn?"""},
            {"role": "assistant", "content": """Step 1) Weng's hourly rate is $12 per 60 minutes.
    Step 2) To find the per-minute rate, divide:
            12 ÷ 60 = $0.20 per minute.
    Step 3) Weng worked for 50 minutes, so her earnings are:
            50 × 0.20 = $10.
    Step 4) Therefore, Weng earned $10.

    **Final answer:
    ####10**"""},

            {"role": "user", "content": f"""# Now, solve the following problem:

    **Question:**
    {question}"""},

            {"role": "assistant", "content": ""}
        ]

        return messages
    
    def format_example(example):
        messages = build_qwen_fewshot_messages(example["problem"])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "solution": example["answer"]
        }

    dataset = dataset.map(format_example)
    
    dataset = DatasetDict({
        "train": split_dataset["train"].map(format_example),
        "test": split_dataset["test"].map(format_example),
    })

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    trainer = GRPOTrainer(
        method=script_args.method,
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=(dataset["test"] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)


# ACCELERATE_LOG_LEVEL=info \
#     accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
#     src/open_r1/grpo_math.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/qwen_math_demo.yaml \
#     --vllm_mode colocate