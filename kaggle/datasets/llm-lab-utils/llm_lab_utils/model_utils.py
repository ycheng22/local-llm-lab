"""Model loading shared by the GPU notebooks (bf16 by default, optional 4-bit and LoRA adapters)."""
from typing import List, Optional, Union

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_model_and_tokenizer(model_id: str, load_in_4bit: bool = False,
                              adapter_path: Optional[Union[str, List[str]]] = None,
                              device_map="auto", dtype: str = "bfloat16", merge: bool = False):
    """
    Loads a causal LM + tokenizer from the Hugging Face Hub.

    dtype: "bfloat16" | "float16" | "float32". NOTE T4 (Turing) has no native bf16; see docs plan section 0 #4.
    load_in_4bit=False -> full-precision weights in `dtype`; True -> NF4 with `dtype` compute.
    adapter_path: a LoRA adapter dir, or a list of dirs that are all activated together
                  (e.g. [sft_adapter, grpo_adapter], matching the local GRPO-on-SFT setup).
    """
    import torch
    torch_dtype = getattr(torch, dtype)
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    if adapter_path:
        from peft import PeftModel

        paths = list(adapter_path) if isinstance(adapter_path, (list, tuple)) else [adapter_path]
        if merge:
            # Fold each LoRA into the weights: no per-step adapter overhead (NB 05 measured ~2.4x slower
            # tokens/s with the adapter left unmerged).
            for p in paths:
                model = PeftModel.from_pretrained(model, p).merge_and_unload()
        elif isinstance(adapter_path, (list, tuple)):
            for i, p in enumerate(adapter_path):
                if i == 0:
                    model = PeftModel.from_pretrained(model, p, adapter_name=f"adapter_{i}")
                else:
                    model.load_adapter(p, adapter_name=f"adapter_{i}")
            model.set_adapter([f"adapter_{j}" for j in range(len(adapter_path))])
        else:
            model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer
