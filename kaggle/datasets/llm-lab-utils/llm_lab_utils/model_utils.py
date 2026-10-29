"""Model loading shared by the GPU notebooks (bf16 by default, optional 4-bit and LoRA adapters)."""
from typing import List, Optional, Union

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(model_id: str, load_in_4bit: bool = False,
                              adapter_path: Optional[Union[str, List[str]]] = None,
                              device_map="auto"):
    """
    Loads a causal LM + tokenizer from the Hugging Face Hub.

    load_in_4bit=False -> bf16 weights (the Kaggle default; 32GB across 2xT4 needs no quantization).
    adapter_path: a LoRA adapter dir, or a list of dirs that are all activated together
                  (e.g. [sft_adapter, grpo_adapter], matching the local GRPO-on-SFT setup).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        if isinstance(adapter_path, (list, tuple)):
            for i, p in enumerate(adapter_path):
                if i == 0:
                    model = PeftModel.from_pretrained(model, p, adapter_name=f"adapter_{i}")
                else:
                    model.load_adapter(p, adapter_name=f"adapter_{i}")
            model.set_adapter([f"adapter_{j}" for j in range(len(adapter_path))])
        else:
            model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer
