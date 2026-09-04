import argparse
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import llm_lab.constants

def main():
    parser = argparse.ArgumentParser(description="Download model weights to local cache")
    parser.add_argument("--model", type=str, default=llm_lab.constants.PRIMARY_MODEL_ID, help="HuggingFace model ID")
    parser.add_argument("--cache-dir", type=str, default=str(llm_lab.constants.HF_HOME), help="Cache directory")
    args = parser.parse_args()

    model_id = args.model
    cache_dir = args.cache_dir
    print(f"Downloading tokenizer for {model_id} (cache_dir={cache_dir})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
        print("Tokenizer downloaded successfully.")
        
        print(f"Downloading model {model_id} (cache_dir={cache_dir})...")
        # We don't need to load the full model into memory here, just let it cache
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True, device_map="cpu")
        print(f"Model {model_id} downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
