"""Generates kaggle/kernels/04-sft-training (MBPP SFT, main path). Run from the repo root: python kaggle/tools/build_nb04.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 04 - SFT on MBPP reference solutions (Phase 2, redesigned)

**Mode:** GPU (T4 x2 session, trains on GPU 0), background commit, **1 h hard cap**. Estimated ~0.2-0.3 GPU hours.

## Why this differs from the local Phase 2
NB 03 showed the synthetic benchmark is saturated (base model: 99.6%), so the local SFT on `train_5k` (4 prompt templates, 3 GPU hours) had nothing to improve. MBPP with the fixed v2 prompt has real headroom (base: Pass@1 40.6%, Pass@8 69.0%). So SFT now trains on **256 MBPP training tasks** and is evaluated (NB 05) on **100 held-out MBPP tasks**. The synthetic reproduction is kept as optional notebook `04z`.

## Learning goals
1. What SFT does: next-token loss on (prompt -> reference solution) chat transcripts.
2. Why LoRA, and why fp16 + gradient checkpointing on a T4.
3. Small-data fine-tuning: 256 examples x 3 epochs, a larger LoRA learning rate than the local run (1e-4 vs 1e-5) because there are 20x fewer examples.

## Data split (disjoint, deterministic - `kaggle/tools/build_mbpp_splits.py`)
| File | Tasks | Use |
|---|---|---|
| `mbpp_v2_train.jsonl` | 256 | SFT / GRPO training |
| `mbpp_v2_dev.jsonl` | 50 | eval-loss monitoring only |
| `mbpp_v2_eval.jsonl` | 100 | **final reporting only** (same tasks as the NB 03 baseline) |

Hyperparameters here are fixed **before** seeing eval results (no tuning on the eval set).
"""),
    setup_cell(install_wheels=True, single_gpu=True),
    code("""
import json, shutil, torch, psutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from llm_lab_utils import PRIMARY_MODEL_ID, ExperimentTracker
from llm_lab_utils.training_utils import format_sft_example, ProgressLoggingCallback, BudgetGuardCallback

EXP_ID = "exp01_sft_mbpp_v1"
OUT = f"/kaggle/working/{EXP_ID}"
CFG = {
    "model": PRIMARY_MODEL_ID, "dtype": "float16",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    "train": {"dataset": "mbpp_v2_train.jsonl", "eval_dataset": "mbpp_v2_dev.jsonl", "lr": 1e-4, "epochs": 3,
              "per_device_batch": 4, "grad_accum": 2, "max_length": 2048, "eval_steps": 16,
              "optim": "adamw_torch", "gradient_checkpointing": True, "seed": 42},
    "max_hours_budget": 1.0,
}
os.makedirs(OUT, exist_ok=True)
json.dump(CFG, open(f"{OUT}/config.json", "w"), indent=2)      # exact config saved with the experiment (rule 4)
print(json.dumps(CFG, indent=1))
"""),
    md("## 1. Data -> chat-template text\nEach example becomes `system + user(prompt incl. example test) + assistant(```python reference```)`. Loss covers the whole sequence (as in the local run)."),
    code("""
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def prep(name):
    ds = load_dataset("json", data_files=f"{DATA}/{name}", split="train")
    return ds.map(lambda x: format_sft_example(x, tokenizer), remove_columns=ds.column_names)

train_ds, eval_ds = prep(CFG["train"]["dataset"]), prep(CFG["train"]["eval_dataset"])
print(len(train_ds), len(eval_ds))
print(train_ds[0]["text"])
lens = [len(tokenizer(x["text"]).input_ids) for x in train_ds]
print("token length: mean", sum(lens) // len(lens), "max", max(lens))
"""),
    md("## 2. Model + LoRA"),
    code("""
model = AutoModelForCausalLM.from_pretrained(PRIMARY_MODEL_ID, dtype=torch.float16, device_map={"": 0}, trust_remote_code=True)
model.config.use_cache = False
model.enable_input_require_grads()
lcfg = LoraConfig(r=CFG["lora"]["r"], lora_alpha=CFG["lora"]["alpha"], lora_dropout=CFG["lora"]["dropout"],
                  target_modules=CFG["lora"]["target_modules"], bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
print("VRAM after load (GB):", round(torch.cuda.memory_allocated(0) / 1e9, 2))
"""),
    md("## 3. Trainer"),
    code("""
args = SFTConfig(
    output_dir=OUT, per_device_train_batch_size=CFG["train"]["per_device_batch"],
    per_device_eval_batch_size=CFG["train"]["per_device_batch"],
    gradient_accumulation_steps=CFG["train"]["grad_accum"], learning_rate=CFG["train"]["lr"],
    num_train_epochs=CFG["train"]["epochs"], logging_steps=4,
    eval_strategy="steps", eval_steps=CFG["train"]["eval_steps"], save_strategy="no",
    fp16=True, bf16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=CFG["train"]["optim"], max_length=CFG["train"]["max_length"],
    dataset_text_field="text", seed=CFG["train"]["seed"], report_to="none",
)
guard = BudgetGuardCallback(max_hours=CFG["max_hours_budget"], check_step=10)
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                     processing_class=tokenizer, callbacks=[ProgressLoggingCallback(8), guard])
"""),
    md("## 4. Train\nThe `[BudgetGuard]` line after 10 steps shows seconds/step and the projected total."),
    code("""
torch.cuda.reset_peak_memory_stats(0)
t_train = time.time()
result = trainer.train()
train_hours = (time.time() - t_train) / 3600
peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
print(f"train hours {train_hours:.3f}  peak VRAM {peak_vram:.2f} GB  final train loss {result.training_loss:.4f}")
assert not guard.stopped_by_budget, f"aborted: projected {guard.projected_hours:.2f} h exceeds budget"
"""),
    md("## 5. Save the adapter, loss curve and provenance"),
    code("""
import hashlib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

final = f"{OUT}/checkpoint-final"
trainer.save_model(final)
tokenizer.save_pretrained(final)

log = trainer.state.log_history
tr = [(e["step"], e["loss"]) for e in log if "loss" in e]
ev = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
plt.figure(figsize=(7, 4)); plt.plot(*zip(*tr), label="train loss")
if ev: plt.plot(*zip(*ev), "o-", label="dev loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title(EXP_ID); plt.savefig(f"{OUT}/loss_curve.png", dpi=120)
json.dump(log, open(f"{OUT}/trainer_log_history.json", "w"))

sha = lambda f: hashlib.sha256(open(f"{DATA}/{f}", "rb").read()).hexdigest()
meta = {"experiment_id": EXP_ID, "parent_checkpoint": PRIMARY_MODEL_ID, "method": "SFT (LoRA, fp16)",
        "train_data": {CFG["train"]["dataset"]: sha(CFG["train"]["dataset"])},
        "dev_data": {CFG["train"]["eval_dataset"]: sha(CFG["train"]["eval_dataset"])},
        "optimizer_steps": trainer.state.global_step, "final_train_loss": result.training_loss,
        "final_dev_loss": ev[-1][1] if ev else None, "gpu_hours": train_hours, "peak_vram_gb": peak_vram,
        "tokens_trained_approx": sum(lens) * CFG["train"]["epochs"]}
json.dump(meta, open(f"{final}/experiment_meta.json", "w"), indent=2)

tracker = ExperimentTracker(EXP_ID + "_summary", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset=CFG["train"]["dataset"], hyperparameters=CFG)
tracker.update_metrics({k: v for k, v in meta.items() if isinstance(v, (int, float))})
tracker.update_hardware(peak_vram, psutil.Process().memory_info().rss / 1e9)
wall_h = (time.time() - T0) / 3600
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
tracker.save()
print(os.listdir(final), "| total wall hours:", round(wall_h, 3))
print("dev loss by step:", ev)
"""),
    md("## 6. Quick sanity generation (one dev task, greedy)"),
    code("""
from llm_lab_utils import load_dataset as load_tasks
from llm_lab_utils.evaluator import generate_completions
model.eval(); model.config.use_cache = True
demo = load_tasks(f"{DATA}/mbpp_v2_dev.jsonl")[0].model_dump()
print(demo["prompt"], "\\n---")
print(generate_completions(model, tokenizer, demo, {"max_new_tokens": 256, "do_sample": False}, n=1)[0][0])
"""),
    md("""
## Takeaways
- Real GPU-hour cost and seconds/step on the T4 (update the plan's budget table).
- Did dev loss fall and then flatten, or rise (overfitting on 256 examples)?
- Next: `05_sft_evaluation` mounts this notebook's output via `kernel_sources` and evaluates on the 100 held-out tasks.
"""),
]

write_kernel("kaggle/kernels/04-sft-training", "04_sft_training.ipynb",
             "ycheng22/04-sft-training", "04 SFT Training", cells, gpu=True, internet=True)
