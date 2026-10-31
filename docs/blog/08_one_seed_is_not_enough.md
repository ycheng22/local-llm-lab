# 8. One seed is not enough: the variance of an RL result

*Series part 8. Experiment: x05 (GRPO seeds 43 and 44, 3.65 GPU h including evaluation).*

## Why this experiment
Our first GRPO run improved Pass@1 by 17.9 points. A single training run cannot separate "the recipe works" from "this run was lucky": sampling during RL training is random, so two runs with identical settings end up in different places. We retrained twice with seeds 43 and 44, changing nothing else, and evaluated on the same 100 held-out tasks.

## Results

| Seed | Pass@1 | Pass@4 | Pass@8 | Gain over base (95% CI) | Truncated |
| :--- | ---: | ---: | ---: | :--- | ---: |
| 42 | 58.5% | 74.4% | 80.0% | +17.9 [+12.4, +23.9] | 8.5% |
| 43 | 51.9% | 70.3% | 75.0% | +11.3 [+5.5, +17.2] | 16.6% |
| 44 | 55.8% | 70.8% | 73.0% | +15.1 [+9.4, +21.1] | 5.5% |
| mean +/- std | **55.4% +/- 3.3** | 71.8% | 76.0% | **+14.8** | |

## Four lessons
1. **The effect is real**: all three seeds are significantly above base at Pass@1 and Pass@4.
2. **The headline moves**: the honest statement is "about +15 points, seed std 3.3 points". The first run was the best of three.
3. **Seeds differ from each other**: seed 42 vs seed 43 is +6.6 pt, CI [+2.9, +10.5], from identical data and hyperparameters. The task-level confidence interval measures evaluation noise; seed variance is a separate, additional noise source.
4. **Pass@8 is fragile**: for seeds 43 and 44 the Pass@8 gain has a CI containing zero. RL mostly improves how often one sample is right.

## What it does to the rest of the project
Any comparison between two single training runs with a gap under roughly 3-6 points is not evidence. This is why the mining ablation (post 4) is reported as a tie, not as "random beats learnable". A variance estimate cost 3.65 GPU hours, about one eighth of a weekly quota, and changed the conclusions of three other experiments. It is the cheapest rigour in the project.
