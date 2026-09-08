import yaml
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

config = yaml.safe_load(open("configs/grpo_v1_5k.yaml"))
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_dataset(example):
    prompt_text = example["prompt"]
    starter = example.get("starter_code", "")
    full_prompt = prompt_text + "\n\n" + starter if starter else prompt_text
    messages = [
        {"role": "system", "content": "You are a helpful programming assistant."},
        {"role": "user", "content": full_prompt}
    ]
    return {"prompt": messages, "prompt_text": prompt_text}

raw_dataset = load_dataset("json", data_files={"train": "data/train_5k.jsonl", "eval": "data/dev_500.jsonl"})
dataset = raw_dataset.map(format_dataset)

def reward_fn(completions, prompt_text, **kwargs):
    return [1.0] * len(completions)

training_args = GRPOConfig(
    output_dir="tmp",
    remove_unused_columns=False,
    max_steps=1,
)

# We just want to see the dataset columns after Trainer initialization
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-2B", device_map="cpu", torch_dtype=torch.float16)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    reward_funcs=reward_fn,
)

print("Trainer train dataset columns:", trainer.train_dataset.column_names)
print("First item keys:", trainer.train_dataset[0].keys())
