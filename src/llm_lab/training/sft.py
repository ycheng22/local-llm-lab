import argparse
import sys
import os
import json
import torch
import time
from pathlib import Path
from tqdm import tqdm
import yaml

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

from llm_lab.constants import HF_HOME, resolve_path

def load_yaml_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def format_prompt(example, tokenizer):
    """
    Format the dataset example into the chat template format for Qwen.
    """
    prompt = example["prompt"]
    starter = example.get("starter_code", "")
    full_prompt = prompt + "\n\n" + starter
    
    # We want the model to generate the reference solution.
    reference = example.get("reference_solution", "")
    
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant. Write Python code to solve the user's problem. Only return the python code inside markdown code blocks."},
        {"role": "user", "content": full_prompt},
        {"role": "assistant", "content": f"```python\n{reference}\n```"}
    ]
    
    # apply_chat_template will convert this into the exact string the model expects
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

class ProgressLoggingCallback(TrainerCallback):
    """
    Logs clear training progress starting at Step 0, Step 1, and every `log_interval` steps with loss, speed, ETA, and VRAM.
    """
    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_step_time = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
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

            loss_str = ""
            if state.log_history:
                for entry in reversed(state.log_history):
                    if "loss" in entry:
                        loss_str = f" | Loss: {entry['loss']:.4f}"
                        break

            vram_str = ""
            if torch.cuda.is_available():
                curr_vram = torch.cuda.memory_allocated() / (1024 ** 3)
                peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
                vram_str = f" | VRAM: {curr_vram:.2f}GB (Peak: {peak_vram:.2f}GB)"

            print(f"[Progress: {progress_str}{loss_str} | Speed: {steps_per_sec:.2f} steps/s{vram_str}]", flush=True)
            self.last_step_time = now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to SFT YAML config")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test instead of full training")
    args = parser.parse_args()
    
    config = load_yaml_config(args.config)
    
    # 1. Setup output dir & logging
    output_dir = resolve_path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting SFT Training. Output dir: {output_dir}", flush=True)
    
    # Save the config for reproducibility
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    # 2. Setup Tokenizer
    model_id = config["model"]["id"]
    cache_dir = HF_HOME
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir, 
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load Dataset
    print("Loading datasets...", flush=True)
    train_path = str(resolve_path(config["training"]["dataset"]))
    eval_path = str(resolve_path(config["training"]["eval_dataset"]))
    print(f"  Train data: {train_path}", flush=True)
    print(f"  Eval data: {eval_path}", flush=True)
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_path, split="train")
    
    # Apply formatting
    train_dataset = train_dataset.map(lambda x: format_prompt(x, tokenizer), remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(lambda x: format_prompt(x, tokenizer), remove_columns=eval_dataset.column_names)
    
    if args.smoke_test:
        print("SMOKE TEST: Truncating dataset to 10 examples", flush=True)
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(5, len(eval_dataset))))
        
    # 4. Setup Model with Quantization
    quant_config = None
    if config["quantization"]["enabled"]:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=config["quantization"]["bits"] == 4,
            bnb_4bit_quant_type=config["quantization"]["quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, config["quantization"]["compute_dtype"]),
            bnb_4bit_use_double_quant=True
        )
        
    print(f"Loading base model {model_id}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    
    # 5. Apply LoRA PEFT
    if config["quantization"]["enabled"]:
        model = prepare_model_for_kbit_training(model)
        
    if config["lora"]["enabled"]:
        print("Applying LoRA...", flush=True)
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    # 6. Training Arguments
    max_steps = 5 if args.smoke_test else -1
    num_train_epochs = 1 if args.smoke_test else config["training"]["num_train_epochs"]
    
    training_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        eval_strategy="steps",
        eval_steps=config["training"]["save_steps"],
        save_total_limit=2,
        bf16=True, # Recommended for newer GPUs if supported
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        optim="paged_adamw_32bit",
        max_length=config["training"]["max_seq_length"],
        dataset_text_field="text"
    )
    
    # 7. Initialize Trainer
    progress_callback = ProgressLoggingCallback(log_interval=1 if args.smoke_test else 50)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[progress_callback],
    )
    
    # 8. Train
    print("Starting training...", flush=True)
    start_time = time.time()
    
    try:
        trainer.train(resume_from_checkpoint=False) # Configurable later
    except Exception as e:
        print(f"Training failed: {e}", flush=True)
        raise
        
    end_time = time.time()
    wall_hours = (end_time - start_time) / 3600.0
    
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_vram = 0.0
        
    print(f"Training completed in {wall_hours:.2f} hours. Peak VRAM: {peak_vram:.2f} GB", flush=True)
    
    # 9. Save final model
    if not args.smoke_test:
        print("Saving final checkpoint...", flush=True)
        trainer.save_model(str(output_dir / "checkpoint-final"))
        
        # Save metrics
        metrics = {
            "wall_hours": wall_hours,
            "peak_vram_gb": peak_vram,
        }
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
