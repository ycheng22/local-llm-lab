from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
msgs = [{"role": "user", "content": "hello"}]
out = tokenizer.apply_chat_template(msgs, return_dict=True, add_generation_prompt=True)
print("apply_chat_template keys:", out.keys())
