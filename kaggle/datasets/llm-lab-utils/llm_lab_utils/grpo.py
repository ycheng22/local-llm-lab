"""
GRPO training (TRL GRPOTrainer) with a verifier-based binary reward.

Run as a subprocess so one notebook session can train two arms on two GPUs at once:
    CUDA_VISIBLE_DEVICES=1 python -m llm_lab_utils.grpo /kaggle/working/<exp>/run_config.json

The reward is the verifier's binary pass/fail only (AGENTS.md rule 11: no extra reward terms).
Code is extracted with the SAME function the evaluator uses, so training reward == evaluation correctness.
"""
import json
import os
import sys
import time
import traceback

from .evaluator import SYSTEM_PROMPT, extract_code
from .subprocess_verifier import SubprocessVerifier


def format_grpo_example(example):
    """TRL GRPO wants a 'prompt' column holding chat messages; all other columns are forwarded to the reward fn."""
    full_prompt = example["prompt"] + ("\n\n" + example["starter_code"] if example.get("starter_code") else "")
    return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": full_prompt}]}


def make_reward_fn(verifier):
    def reward_fn(prompts, completions, **kw):
        rewards = []
        for i, comp in enumerate(completions):
            text = (comp[-1]["content"] if comp else "") if isinstance(comp, list) else comp
            task = {"id": kw["id"][i], "version": kw["version"][i], "category": kw["category"][i],
                    "difficulty": kw["difficulty"][i], "prompt": "", "starter_code": "",
                    "visible_tests": kw["visible_tests"][i], "hidden_tests": kw["hidden_tests"][i]}
            try:
                rewards.append(float(verifier.verify(task, extract_code(text))["reward"]))
            except Exception as e:  # a broken sample must not kill a multi-hour run
                print(f"[reward] verification error: {e}", flush=True)
                rewards.append(0.0)
        return rewards
    return reward_fn


def run_grpo(cfg: dict) -> dict:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from .training_utils import BudgetGuardCallback, ProgressLoggingCallback

    out = cfg["output_dir"]
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], dtype=torch.float16, device_map={"": 0},
                                                 trust_remote_code=True)
    if cfg.get("init_adapter"):
        # Start from the SFT model: merge its LoRA into the base, then train a fresh GRPO LoRA on top.
        model = PeftModel.from_pretrained(model, cfg["init_adapter"]).merge_and_unload()
    model.config.use_cache = False

    # TRL adds mm_token_type_ids for Qwen (thinks it is Qwen-VL); the text model rejects it.
    _gen, _fwd = model.generate, model.forward
    model.generate = lambda *a, **k: _gen(*a, **{x: y for x, y in k.items() if x != "mm_token_type_ids"})
    model.forward = lambda *a, **k: _fwd(*a, **{x: y for x, y in k.items() if x != "mm_token_type_ids"})

    ds = load_dataset("json", data_files=cfg["train_file"], split="train")
    ds = ds.remove_columns([c for c in ("metadata", "reference_solution") if c in ds.column_names])
    ds = ds.map(format_grpo_example)

    g = cfg["grpo"]
    args = GRPOConfig(
        output_dir=out, learning_rate=g["lr"], per_device_train_batch_size=g["per_device_batch"],
        gradient_accumulation_steps=g["grad_accum"], num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"], max_steps=g["max_steps"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, bf16=False, logging_steps=g.get("logging_steps", 5),
        save_steps=g.get("save_steps", 50), save_total_limit=2, seed=g.get("seed", 42),
        remove_unused_columns=False, report_to="none",
    )
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    guard = BudgetGuardCallback(max_hours=cfg["max_hours"], check_step=cfg.get("guard_step", 5))
    trainer = GRPOTrainer(model=model, reward_funcs=[make_reward_fn(SubprocessVerifier(timeout=3.0, max_memory_mb=512))],
                          args=args, train_dataset=ds, peft_config=lcfg, processing_class=tok,
                          callbacks=[ProgressLoggingCallback(5), guard])

    torch.cuda.reset_peak_memory_stats(0)
    t_train = time.time()
    trainer.train()
    train_hours = (time.time() - t_train) / 3600

    final = os.path.join(out, "checkpoint-final")
    trainer.save_model(final)
    tok.save_pretrained(final)
    log = trainer.state.log_history
    json.dump(log, open(os.path.join(out, "trainer_log_history.json"), "w"))
    meta = {"experiment_id": cfg["experiment_id"], "method": "GRPO (LoRA, fp16)",
            "parent_checkpoint": cfg.get("init_adapter") or cfg["model_id"], "base_model": cfg["model_id"],
            "train_file": cfg["train_file"], "optimizer_steps": trainer.state.global_step,
            "stopped_by_budget": guard.stopped_by_budget, "projected_hours": guard.projected_hours,
            "gpu_hours": train_hours, "wall_hours": (time.time() - t0) / 3600,
            "peak_vram_gb": torch.cuda.max_memory_allocated(0) / 1e9, "config": cfg}
    json.dump(meta, open(os.path.join(final, "experiment_meta.json"), "w"), indent=2)
    return meta


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    cfg = json.load(open(cfg_path))
    try:
        run_grpo(cfg)
    except Exception:
        traceback.print_exc()
        open(os.path.join(cfg["output_dir"], "FAILED.txt"), "w").write(traceback.format_exc())
        sys.exit(1)
