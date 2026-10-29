"""Generates kaggle/kernels/01b-dtype-throughput. Run from the repo root: python kaggle/tools/build_nb01b.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 01b - dtype Throughput: rollouts and LoRA training steps (Phase 0)

**Mode:** GPU (T4 x2), background commit. Budget ~10-15 min.

NB 01 showed bf16 and fp16 both *load and run* on a T4, but its tasks were trivial (21 tokens each), so its speed numbers were mostly latency noise. The decision that actually matters is the cost of the two things SFT/GRPO do all day:

- **A. Rollouts**: sample 4 completions per prompt (GRPO `num_generations=4`) with realistic length.
- **B. Training steps**: LoRA forward + backward with gradient checkpointing; in fp16 this needs a GradScaler and can overflow (NaN), in bf16 on a T4 it runs through slower emulated paths.

Each dtype is warmed up before timing, and every part is wrapped in try/except so one failure does not waste the session.
"""),
    setup_cell(install_wheels=False),
    code("""
import gc, json, math
import torch, psutil
from llm_lab_utils import PRIMARY_MODEL_ID, load_dataset
from llm_lab_utils.model_utils import setup_model_and_tokenizer
from llm_lab_utils.evaluator import build_messages

tasks = load_dataset(f"{DATA}/train_complex_500.jsonl")[:8]   # MBPP: realistic prompts
GEN_TOKENS, N_GEN, TRAIN_LEN, TRAIN_BS = 256, 4, 512, 2
RUN_ROLLOUT = False   # run 1 measured rollouts (bf16 28.7 vs fp16 73.1 tok/s, bf16 reproduced exactly); re-run only the training part
print(torch.__version__, torch.cuda.get_device_name(0))
"""),
    md("""
## A. Rollout throughput
`min_new_tokens == max_new_tokens == 256` forces equal work for both dtypes (we measure compute, not how soon the model emits EOS). Sampling matches GRPO (temperature 0.7, top-p 0.9, 4 sequences per prompt).
"""),
    code("""
def rollout_bench(model, tok):
    prompts = [tok.apply_chat_template(build_messages(t.model_dump()), tokenize=False, add_generation_prompt=True) for t in tasks]
    def one(p):
        inp = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            return model.generate(**inp, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=N_GEN,
                                  max_new_tokens=GEN_TOKENS, min_new_tokens=GEN_TOKENS, pad_token_id=tok.eos_token_id)
    for p in prompts[:2]: one(p)                      # warm-up (kernels, allocator)
    torch.cuda.synchronize(); t = time.time(); n = 0
    for p in prompts[2:6]:
        o = one(p); n += N_GEN * GEN_TOKENS
    torch.cuda.synchronize(); dt = time.time() - t
    return {"rollout_tokens_per_s": round(n / dt, 1), "seconds_per_prompt_x4": round(dt / 4, 2)}
"""),
    md("""
## B. LoRA training step
r=16 on all linear layers, batch 2 x 512 tokens, gradient checkpointing. fp16 uses `GradScaler`; bf16 does not need one. We record step time, peak VRAM and whether the loss stayed finite.
"""),
    code("""
from peft import LoraConfig, get_peft_model

def train_bench(model, tok, dtype):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    use_scaler = (dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    text = "\\n".join(t.prompt + "\\n" + (t.reference_solution or "") for t in tasks) * 20
    ids = tok(text, return_tensors="pt").input_ids[:, :TRAIN_LEN].repeat(TRAIN_BS, 1).to(model.device)
    model.train()
    torch.cuda.reset_peak_memory_stats(0)
    times, losses = [], []
    for step in range(13):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=dtype):
            loss = model(input_ids=ids, labels=ids).loss
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= 3: times.append(time.time() - t)       # first 3 = warm-up
        losses.append(float(loss))
    return {"train_step_s": round(sum(times) / len(times), 3),
            "train_peak_vram_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
            "loss_first": round(losses[0], 3), "loss_last": round(losses[-1], 3),
            "loss_finite": all(math.isfinite(x) for x in losses),
            "trainable_params_m": round(sum(p.numel() for p in params) / 1e6, 1)}
"""),
    md("## Run both parts for each dtype (order bf16, fp16, then bf16 again to expose ordering/warm-up bias)"),
    code("""
results = []
for label, dtype_name in [("bf16", "bfloat16"), ("fp16", "float16"), ("bf16_repeat", "bfloat16")]:
    out = {"label": label}
    model = tok = None
    try:
        model, tok = setup_model_and_tokenizer(PRIMARY_MODEL_ID, dtype=dtype_name, device_map={"": 0})
        if RUN_ROLLOUT:
            out.update(rollout_bench(model, tok))
    except Exception as e:
        out["rollout_error"] = f"{type(e).__name__}: {e}"[:400]
    try:
        out.update(train_bench(model, tok, getattr(torch, dtype_name)))
    except Exception as e:
        out["train_error"] = f"{type(e).__name__}: {e}"[:400]
    del model, tok; gc.collect(); torch.cuda.empty_cache()
    print(out); results.append(out)
"""),
    md("## Decision"),
    code("""
import pandas as pd
df = pd.DataFrame(results); print(df.to_string())
by = {r["label"]: r for r in results}
def ok(r): return "train_step_s" in r and r.get("loss_finite")
cands = [r for r in (by.get("bf16"), by.get("fp16")) if r and ok(r)]
rec = min(cands, key=lambda r: r["train_step_s"])["label"] if cands else None
print("\\nRECOMMENDED training dtype:", rec)
if by.get("fp16") and "loss_finite" in by["fp16"] and not by["fp16"]["loss_finite"]:
    print("WARNING: fp16 produced a non-finite loss (overflow) - do not use it for this model.")
"""),
    code("""
from llm_lab_utils import ExperimentTracker
tracker = ExperimentTracker("exp00b_dtype_throughput", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset="train_complex_500.jsonl[:8]",
                   hyperparameters={"gen_tokens": GEN_TOKENS, "num_generations": N_GEN, "train_len": TRAIN_LEN,
                                    "train_bs": TRAIN_BS, "lora_r": 16, "grad_ckpt": True})
tracker.update_metrics({"results": results, "recommended": rec})
tracker.update_hardware(max([r.get("train_peak_vram_gb", 0) for r in results]), psutil.Process().memory_info().rss / 1e9)
wall_h = (time.time() - T0) / 3600
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
tracker.save()
print(f"saved; wall {wall_h * 60:.1f} min")
"""),
]

write_kernel("kaggle/kernels/01b-dtype-throughput", "01b_dtype_throughput.ipynb",
             "ycheng22/01b-dtype-throughput", "01b Dtype Throughput", cells, gpu=True, internet=True)
