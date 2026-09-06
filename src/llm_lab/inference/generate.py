import llm_lab.constants  # sets environment variables before transformers loads
import argparse
import yaml
import sys
import torch
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_model_and_tokenizer(model_cfg: dict):
    model_id = model_cfg.get("id")
    cache_dir = model_cfg.get("cache_dir")
    local_files_only = model_cfg.get("local_files_only", False)
    
    quantization_config = None
    if model_cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    print(f"Loading tokenizer {model_id} (cache_dir={cache_dir})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        trust_remote_code=True
    )
    
    print(f"Loading model {model_id} (cache_dir={cache_dir})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device_map=model_cfg.get("device_map", "auto"),
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        from peft import PeftModel
        from llm_lab.constants import resolve_path
        adapter_resolved = str(resolve_path(adapter_path))
        print(f"Loading LoRA adapter from {adapter_resolved}...")
        model = PeftModel.from_pretrained(model, adapter_resolved)
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Run inference using config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--prompt", type=str, help="Text to generate from (overrides interactive mode)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    try:
        model, tokenizer = setup_model_and_tokenizer(config["model"])
    except Exception as e:
        print(f"Failed to load model: {e}", file=sys.stderr)
        sys.exit(1)
        
    gen_cfg = config.get("generation", {})
    
    # We can do a quick interactive loop if no prompt provided
    def generate_response(text):
        start_time = time.time()
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 384),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            do_sample=gen_cfg.get("do_sample", True),
            pad_token_id=tokenizer.eos_token_id
        )
        
        latency = time.time() - start_time
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\nResponse:\n{generated_text}")
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"\n[Latency: {latency:.2f}s | Peak VRAM: {peak_mem:.2f}GB]")

    if args.prompt:
        generate_response(args.prompt)
    else:
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            try:
                user_in = input("\nPrompt> ")
                if user_in.strip().lower() in ('quit', 'exit'):
                    break
                if not user_in.strip():
                    continue
                generate_response(user_in)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
