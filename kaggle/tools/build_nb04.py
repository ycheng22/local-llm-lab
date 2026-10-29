"""Generates kaggle/kernels/04-sft-training. Run from the repo root: python kaggle/tools/build_nb04.py"""
import sys
sys.path.insert(0, "kaggle/tools")
from nbbuild import md, code, setup_cell, write_kernel

cells = [
    md("""
# 04 - SFT Training (Phase 2)

**Mode:** GPU (T4 x2 session, trains on GPU 0), background commit, **4 h hard cap**. Estimated 1.5-3 GPU hours.

## Learning goals
1. What supervised fine-tuning (SFT) does: next-token loss on (prompt -> reference solution) chat transcripts.
2. Why LoRA: train ~0.5-1% of the parameters; the base model stays frozen.
3. Why fp16 + gradient checkpointing on a T4 (measured in NB 01b: 3.5x faster training steps than bf16).
4. How to protect a free quota: the `BudgetGuardCallback` projects total time after 20 steps and aborts if it would blow the budget.

## Faithful port of the local run (`configs/sft_v1_5k.yaml`)
| Setting | Local (RTX 2060S, 8 GB) | Kaggle (T4, 15 GB) | Why |
|---|---|---|---|
| Base model | Qwen3.5-2B | same | |
| Weights | 4-bit NF4 + bf16 compute | **fp16, no quantization** | NB 01b |
| LoRA | r=16, alpha=32, dropout 0.05, 7 proj modules | same | |
| Data | `train_5k.jsonl` (5000), eval `dev_500` | same | |
| Learning rate / epochs | 1e-5 / 1 | same | |
| Effective batch | 1 x 8 accumulation = 8 | **4 x 2 accumulation = 8** | same optimizer behaviour, faster |
| Optimizer | paged_adamw_32bit | adamw_torch | no 4-bit memory pressure; same math |
| Precision flag | bf16 | **fp16 (GradScaler)** | T4 has no native bf16 |

> **Known weakness (from NB 02):** `train_5k` has only 4 prompt templates, so this SFT mostly teaches the output format. We reproduce it faithfully so results stay comparable with the local run; it is not evidence that SFT improves general coding.
"""),
    setup_cell(install_wheels=True),
    code("""
import json, shutil, torch, psutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from llm_lab_utils import PRIMARY_MODEL_ID, ExperimentTracker
from llm_lab_utils.training_utils import format_sft_example, ProgressLoggingCallback, BudgetGuardCallback

EXP_ID = "exp01_sft_kaggle_v1"
OUT = f"/kaggle/working/{EXP_ID}"
CFG = {
    "model": PRIMARY_MODEL_ID, "dtype": "float16",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
    "train": {"dataset": "train_5k.jsonl", "eval_dataset": "dev_500.jsonl", "lr": 1e-5, "epochs": 1,
              "per_device_batch": 4, "grad_accum": 2, "max_length": 4096, "save_steps": 250,
              "optim": "adamw_torch", "gradient_checkpointing": True, "seed": 42},
    "max_hours_budget": 3.5,
}
os.makedirs(OUT, exist_ok=True)
json.dump(CFG, open(f"{OUT}/config.json", "w"), indent=2)      # exact config saved with the experiment (rule 4)
print(json.dumps(CFG, indent=1))
"""),
    md("## 1. Data -> chat-template text\nEach example becomes `system + user(prompt) + assistant(```python reference```)`. Loss is computed over the whole sequence (as in the local run)."),
    code("""
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def prep(name):
    ds = load_dataset("json", data_files=f"{DATA}/{name}", split="train")
    return ds.map(lambda x: format_sft_example(x, tokenizer), remove_columns=ds.column_names)

train_ds, eval_ds = prep(CFG["train"]["dataset"]), prep(CFG["train"]["eval_dataset"])
print(len(train_ds), len(eval_ds))
print(train_ds[0]["text"])
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
    num_train_epochs=CFG["train"]["epochs"], logging_steps=10, save_steps=CFG["train"]["save_steps"],
    eval_strategy="steps", eval_steps=CFG["train"]["save_steps"], save_total_limit=2,
    fp16=True, bf16=False, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim=CFG["train"]["optim"], max_length=CFG["train"]["max_length"],
    dataset_text_field="text", seed=CFG["train"]["seed"], report_to="none",
)
guard = BudgetGuardCallback(max_hours=CFG["max_hours_budget"], check_step=20)
trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                     processing_class=tokenizer, callbacks=[ProgressLoggingCallback(50), guard])
"""),
    md("## 4. Train\nWatch the first lines: `[BudgetGuard]` prints the measured seconds/step and the projected total hours after 20 steps."),
    code("""
torch.cuda.reset_peak_memory_stats(0)
t_train = time.time()
result = trainer.train()
train_hours = (time.time() - t_train) / 3600
peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
print(f"train hours {train_hours:.3f}  peak VRAM {peak_vram:.2f} GB  final train loss {result.training_loss:.4f}")
assert not guard.stopped_by_budget, f"aborted: projected {guard.projected_hours:.2f} h exceeds budget; adjust CFG before re-running"
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
if ev: plt.plot(*zip(*ev), "o-", label="eval loss")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.title(EXP_ID); plt.savefig(f"{OUT}/loss_curve.png", dpi=120)
json.dump(log, open(f"{OUT}/trainer_log_history.json", "w"))

sha = lambda f: hashlib.sha256(open(f"{DATA}/{f}", "rb").read()).hexdigest()
meta = {"experiment_id": EXP_ID, "parent_checkpoint": PRIMARY_MODEL_ID, "method": "SFT (LoRA, fp16)",
        "train_data": {CFG["train"]["dataset"]: sha(CFG["train"]["dataset"])},
        "eval_data": {CFG["train"]["eval_dataset"]: sha(CFG["train"]["eval_dataset"])},
        "optimizer_steps": trainer.state.global_step, "final_train_loss": result.training_loss,
        "final_eval_loss": ev[-1][1] if ev else None, "gpu_hours": train_hours, "peak_vram_gb": peak_vram}
json.dump(meta, open(f"{final}/experiment_meta.json", "w"), indent=2)

tracker = ExperimentTracker(EXP_ID + "_summary", "/kaggle/working")
tracker.set_config(model=PRIMARY_MODEL_ID, dataset=CFG["train"]["dataset"], hyperparameters=CFG)
tracker.update_metrics({k: v for k, v in meta.items() if isinstance(v, (int, float))})
tracker.update_hardware(peak_vram, psutil.Process().memory_info().rss / 1e9)
wall_h = (time.time() - T0) / 3600
tracker.data["training"].update(wall_hours=round(wall_h, 4), gpu_hours=round(wall_h, 4))
tracker.save()
for d in glob.glob(f"{OUT}/checkpoint-[0-9]*"):          # keep Output small; the final adapter is what later notebooks use
    shutil.rmtree(d, ignore_errors=True)
print(os.listdir(final), "| total wall hours:", round(wall_h, 3))
"""),
    md("## 6. Quick sanity generation with the trained adapter"),
    code("""
from llm_lab_utils.evaluator import generate_completions
model.eval(); model.config.use_cache = True
demo = {"prompt": "Write a Python function called `reverse_string(s)` that returns the reversed string.", "starter_code": "def reverse_string(s):\\n    "}
print(generate_completions(model, tokenizer, demo, {"max_new_tokens": 128, "do_sample": False}, n=1)[0][0])
"""),
    md("""
## Takeaways
- How many seconds/step did fp16 achieve vs the local ~4-bit run, and what was the real GPU-hour cost? (compare with the 3.04 h local figure and update the plan's budget table)
- Did eval loss track train loss (no overfitting on 4 templates)?
- The adapter in `checkpoint-final` is mounted by NB 05+ through `kernel_sources`. Next: `05_sft_evaluation`.
"""),
]

write_kernel("kaggle/kernels/04-sft-training", "04_sft_training.ipynb",
             "ycheng22/04-sft-training", "04 SFT Training", cells, gpu=True, internet=True)
