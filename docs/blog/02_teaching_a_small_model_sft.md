# 2. Supervised fine-tuning gave us nothing (and that is a result)

*Series part 2. Experiments: notebooks 04 (training, 0.16 GPU h) and 05 (evaluation).*

## The experiment
256 MBPP tasks with reference solutions. LoRA fine-tuning in fp16, 96 steps, 7.3 GB peak VRAM, about ten minutes. Evaluation on 100 held-out tasks with 8 samples each, and a paired bootstrap over tasks so we can say how large a difference must be to count.

| Metric | Base | SFT | Difference | 95% CI |
| :--- | ---: | ---: | ---: | :--- |
| Pass@1 | 40.6% | 39.0% | -1.6 pt | [-8.9, +6.0] |
| Pass@4 | 61.4% | 57.6% | -3.8 pt | [-12.2, +5.0] |
| Pass@8 | 69.0% | 65.0% | -4.0 pt | [-14.0, +6.0] |

Every interval contains zero: **no detectable gain**, and if anything the point estimates are negative.

## Why a null result is worth reporting
- With 100 tasks, the 95% interval on a single comparison is about +/-6 to 9 points. An improvement of 2 points would have been invisible, and a "+2.4 pt" headline would have been noise. Quote the interval, not just the mean.
- The likely reason is mundane: the base model already writes valid Python for these tasks, and 256 reference solutions teach style, not search. SFT narrows the output distribution; Pass@8 (the diversity ceiling) dropped the most.
- SFT also became a worse starting point for RL (post 3): GRPO started from the SFT model reached only 42.6%, against 58.5% from the base model.

## Takeaway
Fine-tuning on solutions is not automatically progress. Measure with confidence intervals and keep the null results in the write-up. Next: [using unit tests as the reward](03_grpo_unit_tests_as_reward.md).
