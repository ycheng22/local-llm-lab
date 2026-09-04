# Local LLM Lab Development Rules

## Goal

Build a reproducible small-scale LLM capability research framework
for an 8GB NVIDIA GPU and 48GB system RAM.

Primary model:
Qwen/Qwen3.5-2B

Secondary model:
Qwen/Qwen3.5-4B

## Principles

1. Reproducibility is more important than convenience.
2. Never modify the frozen test set.
3. Every experiment must have a unique experiment ID.
4. Every experiment must save its exact config.
5. Every model checkpoint must record its parent checkpoint.
6. Never silently change hyperparameters.
7. Never mix train/dev/test examples.
8. Never execute generated code directly on the Windows host.
9. Python candidate code must run inside a sandbox/container.
10. Every experiment must report:
   - Pass@1
   - Pass@4
   - Pass@8
   - OOD Pass@1
   - peak VRAM
   - peak RAM
   - wall time
   - GPU hours
11. Do not introduce additional reward terms unless explicitly requested.
12. Keep evaluation deterministic where possible.
13. Add tests before refactoring core infrastructure.
14. Avoid unnecessary dependencies.
15. Do not redesign the model architecture.
16. Prefer small experiments that can finish in hours.
17. Never overwrite previous experiment artifacts.
18. We must document all metrics, important analysis, and a summary of completed tasks (based on the implementation plan instructions) in the README.md after each phase.
