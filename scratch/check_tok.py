from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
print(tokenizer("hello").keys())
