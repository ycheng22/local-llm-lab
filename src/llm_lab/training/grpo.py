import argparse
import sys
import os
import json
import torch
import time
import re
from pathlib import Path
from tqdm import tqdm
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from llm_lab.constants import HF_HOME, resolve_path
from llm_lab.verifier import DockerVerifier

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_code(completion: str) -> str:
    """Extract Python code from markdown code blocks if present."""
    match = re.search(r'```python\s*(.*?)\s*```', completion, re.DOTALL)
    if match:
        return match.group(1)
    # Fallback to general code block
    match = re.search(r'```\s*(.*?)\s*```', completion, re.DOTALL)
    if match:
        return match.group(1)
    return completion.strip()

def format_dataset(example):
    """
    Format dataset for GRPOTrainer. TRL expects a 'prompt' column which can be 
    a list of messages (dict) for chat templates.
    """
    prompt_text = example["prompt"]
    starter = example.get("starter_code", "")
    full_prompt = prompt_text + "\n\n" + starter if starter else prompt_text
    
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant. Write Python code to solve the user's problem. Only return the python code inside markdown code blocks."},
        {"role": "user", "content": full_prompt}
    ]
    return {"prompt": messages}

class ProgressLoggingCallback(TrainerCallback):
    """
    Logs clear training progress for GRPO.
    """
    def __init__(self, log_interval: int = 10):
        super().__init__()
        self.log_interval = log_interval
        self.last_step_time = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step_time = time.time()
        vram_str = ""
        if torch.cuda.is_available():
            curr_vram = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_str = f" | Initial VRAM: {curr_vram:.2f}GB"
        total_steps = state.max_steps if state.max_steps > 0 else "unknown"
        print(f"[Progress: Step 0/{total_steps} (0.0%) | Training started{vram_str}]", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and (state.global_step == 1 or state.global_step % self.log_interval == 0):
            now = time.time()
            step_delta = now - self.last_step_time
            steps_completed = 1 if state.global_step == 1 else self.log_interval
            steps_per_sec = steps_completed / step_delta if step_delta > 0 else 0
            
            progress_str = f"Step {state.global_step}"
            if state.max_steps > 0:
                pct = (state.global_step / state.max_steps) * 100
                remaining_steps = max(0, state.max_steps - state.global_step)
                eta_secs = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                progress_str = f"Step {state.global_step}/{state.max_steps} ({pct:.1f}%) | ETA: {eta_secs / 60:.1f}m"

            metrics_str = ""
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry or "reward" in entry:
                        loss = entry.get("loss", 0.0)
                        reward = entry.get("reward", 0.0)
                        metrics_str = f" | Loss: {loss:.4f} | Reward: {reward:.4f}"
                        break

            vram_str = ""
            if torch.cuda.is_available():
                curr_vram = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                vram_str = f" | VRAM: {curr_vram:.2f}GB (Peak: {peak_vram:.2f}GB)"

            print(f"[Progress: {progress_str}{metrics_str} | Speed: {steps_per_sec:.2f} steps/s{vram_str}]", flush=True)
            self.last_step_time = now

def build_reward_function():
    # Instantiate verifier once
    verifier = DockerVerifier(timeout=3.0, memory_limit="512m")
    
    def reward_fn(prompts, completions, **kwargs):
        """
        Custom reward function for GRPO.
        prompts: list of strings (or lists of dicts)
        completions: list of strings (the generated text, missing the prompt part)
        kwargs: dataset columns flattened (each is a list parallel to completions)
        """
        rewards = []
        
        # TRL GRPO flattens the prompt/completions into a single list of size (batch_size * num_generations)
        # We need to extract the corresponding task schema for each generation to verify it.
        # kwargs contains lists of the same size.
        for i in range(len(completions)):
            # Reconstruct the task dictionary for the verifier
            task_dict = {
                "id": kwargs["id"][i],
                "version": kwargs["version"][i],
                "category": kwargs["category"][i],
                "difficulty": kwargs["difficulty"][i],
                "prompt": kwargs.get("prompt_text", [""] * len(completions))[i], # Original prompt before chat template formatting if we saved it
                "starter_code": kwargs.get("starter_code", [""] * len(completions))[i],
                "visible_tests": kwargs["visible_tests"][i],
                "hidden_tests": kwargs["hidden_tests"][i],
            }
            
            # The completion in TRL includes the generated text. 
            # In latest TRL, completions are returned as a list of lists if chat template is used, or raw strings.
            completion = completions[i]
            if isinstance(completion, list):
                # If it's a list of messages, extract the assistant's content
                completion_text = completion[-1]["content"] if len(completion) > 0 else ""
            else:
                completion_text = completion
                
            candidate_code = extract_code(completion_text)
            
            try:
                result = verifier.verify(task_dict, candidate_code)
                rewards.append(result["reward"]) # 1.0 if passed, 0.0 otherwise
            except Exception as e:
                print(f"Error during verification: {e}")
                rewards.append(0.0)
                
        return rewards
    
    return reward_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to GRPO YAML config")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test instead of full training")
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    
    output_dir = str(resolve_path(config["logging"]["output_dir"]))
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = str(resolve_path(config["dataset"]["train"]))
    eval_path = str(resolve_path(config["dataset"]["eval"]))
    
    raw_dataset = load_dataset(
        "json",
        data_files={"train": train_path, "eval": eval_path}
    )
    
    # Keep the original prompt text as a separate column so the reward function can access it
    def add_prompt_text(example):
        example["prompt_text"] = example["prompt"]
        return example
        
    raw_dataset = raw_dataset.map(add_prompt_text)
    
    # Format dataset to have "prompt" as chat messages for TRL
    dataset = raw_dataset.map(format_dataset)
    
    # --- Model Loading ---
    model_config = config["model"]
    model_id = str(resolve_path(model_config["base_checkpoint"])) # From SFT Phase
    
    quant_config_dict = config.get("quantization", {})
    bnb_config = None
    if quant_config_dict.get("enabled", False):
        compute_dtype = getattr(torch, quant_config_dict.get("compute_dtype", "float16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config_dict.get("bits") == 4,
            bnb_4bit_quant_type=quant_config_dict.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

    print(f"Loading tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=model_config.get("local_files_only", True))
    
    print(f"Loading model from {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=model_config.get("local_files_only", True)
    )
    
    # Ensure PAD token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for LoRA
    if quant_config_dict.get("enabled", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config["grpo"].get("gradient_checkpointing", True)
        )
        
    lora_config_dict = config.get("lora", {})
    peft_config = None
    if lora_config_dict.get("enabled", False):
        peft_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("alpha", 32),
            lora_dropout=lora_config_dict.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
    # --- GRPO Training Arguments ---
    grpo_config_dict = config["grpo"]
    
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(grpo_config_dict.get("learning_rate", 1e-5)),
        per_device_train_batch_size=grpo_config_dict.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=grpo_config_dict.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=grpo_config_dict.get("gradient_checkpointing", True),
        logging_steps=grpo_config_dict.get("logging_steps", 10),
        save_steps=grpo_config_dict.get("save_steps", 50),
        max_completion_length=grpo_config_dict.get("max_completion_length", 384),
        num_generations=grpo_config_dict.get("num_generations", 4),
        max_steps=2 if args.smoke_test else -1,
        # TRL GRPO defaults
        remove_unused_columns=False, # We need our dataset columns in kwargs!
        report_to="none"
    )
    
    # Define reward function
    reward_fn = build_reward_function()
    
    # Patch model.generate and model.forward to ignore mm_token_type_ids
    # TRL GRPOTrainer incorrectly adds this for Qwen models thinking it's Qwen-VL.
    original_generate = model.generate
    def patched_generate(*args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return original_generate(*args, **kwargs)
    model.generate = patched_generate
    
    original_forward = model.forward
    def patched_forward(*args, **kwargs):
        kwargs.pop("mm_token_type_ids", None)
        return original_forward(*args, **kwargs)
    model.forward = patched_forward

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        peft_config=peft_config,
        callbacks=[ProgressLoggingCallback(log_interval=grpo_config_dict.get("logging_steps", 10))]
    )
    
    print("Starting GRPO Training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    # Save final model
    trainer.save_model(os.path.join(output_dir, "checkpoint-final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
    
    print("Final checkpoint saved.")

if __name__ == "__main__":
    main()
