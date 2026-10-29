"""Helpers shared by the SFT / GRPO notebooks (formatting, progress logging, GPU-budget guard)."""
import time

from .evaluator import SYSTEM_PROMPT


def format_sft_example(example, tokenizer):
    """Chat-template text for one task: system + user(prompt + starter) + assistant(```python reference```)."""
    full_prompt = example["prompt"] + "\n\n" + (example.get("starter_code") or "")
    reference = example.get("reference_solution") or ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
        {"role": "assistant", "content": f"```python\n{reference}\n```"},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


try:
    from transformers import TrainerCallback as _Base
except ImportError:  # keeps the module importable (and testable) without transformers
    _Base = object


class ProgressLoggingCallback(_Base):
    """Prints step, % done, ETA, last loss and VRAM at step 1 and every `log_interval` steps."""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.last = time.time()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last = time.time()
        print(f"[Progress: Step 0/{state.max_steps} | training started]", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        s = state.global_step
        if s == 1 or (s > 0 and s % self.log_interval == 0):
            import torch
            now = time.time()
            done = 1 if s == 1 else self.log_interval
            sps = done / max(now - self.last, 1e-9)
            eta_min = max(0, state.max_steps - s) / sps / 60 if sps else 0
            last = next((e for e in reversed(state.log_history) if "loss" in e or "reward" in e), {})
            loss = last.get("loss", float("nan"))
            reward = f" | reward {last['reward']:.3f}" if "reward" in last else ""
            peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(f"[Progress: Step {s}/{state.max_steps} ({100 * s / max(state.max_steps, 1):.1f}%) | "
                  f"ETA {eta_min:.1f}m | loss {loss:.4f}{reward} | {sps:.3f} steps/s | peak VRAM {peak:.2f}GB]", flush=True)
            self.last = now


class BudgetGuardCallback(_Base):
    """
    Protects the free GPU quota: after `check_step` steps, projects total wall time from the measured speed
    and stops training if the projection exceeds `max_hours`. Sets `.stopped_by_budget` and `.projected_hours`.
    """

    def __init__(self, max_hours: float, check_step: int = 20):
        self.max_hours = max_hours
        self.check_step = check_step
        self.t0 = None
        self.stopped_by_budget = False
        self.projected_hours = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.check_step and state.max_steps > 0:
            per_step = (time.time() - self.t0) / self.check_step
            self.projected_hours = per_step * state.max_steps / 3600
            print(f"[BudgetGuard] {per_step:.2f}s/step -> projected {self.projected_hours:.2f} h "
                  f"(limit {self.max_hours} h)", flush=True)
            if self.projected_hours > self.max_hours:
                print("[BudgetGuard] projection exceeds budget: stopping.", flush=True)
                self.stopped_by_budget = True
                control.should_training_stop = True
