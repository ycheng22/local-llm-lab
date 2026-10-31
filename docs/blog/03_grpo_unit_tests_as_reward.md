# 3. Unit tests as the reward: GRPO, and how much of the gain is real

*Series part 3. Experiments: notebooks 06/07 (training and evaluation), x02 (HumanEval), x05 (seeds), x07 (renaming).*

## What GRPO is, in three sentences
For each prompt, sample a group of 4 completions. Reward is 1 if the code passes all the tests and 0 otherwise; nothing else (no length penalty, no style terms). Push up the probability of completions that scored above their group's average. There is no value network and no preference data; the verifier is the teacher.

A group where all four samples pass, or all four fail, has zero reward variance and teaches nothing. In our runs about 60% of the groups were like this (post 4).

## First result: +17.9 points
200 steps on 256 MBPP tasks (2.34 GPU hours, two arms in parallel on the two GPUs):

| Model | Pass@1 | Pass@4 | Pass@8 |
| :--- | ---: | ---: | ---: |
| base | 40.6% | 61.4% | 69.0% |
| SFT | 39.0% | 57.6% | 65.0% |
| GRPO from base | **58.5%** | 74.4% | 80.0% |
| GRPO from SFT | 42.6% | 56.4% | 63.0% |

+17.9 pt over base, 95% CI [+12.4, +23.9]. This was the headline. Then we tried to break it.

## Stress test 1: more seeds
We retrained with seeds 43 and 44, changing nothing else: **51.9% and 55.8%**. Mean 55.4%, std 3.3 pt, gain **+14.8 pt** (range +11.3 to +17.9). All three are significantly above base, but **the first run was the luckiest**. If we had trained once and stopped, we would have published a number 3 points too optimistic. Gains at Pass@8 for seeds 43 and 44 were not significant: RL mostly makes single samples more reliable, it does not raise the ceiling much.

## Stress test 2: rename the function
Replace every function name (prompt, tests, reference) with a neutral name like `compute_result`. GRPO minus base is still **+13.0 pt** [+7.2, +19.0]. The gain is not just memorised names.

## Stress test 3: a different benchmark
On HumanEval (157 of 164 problems expressible as plain asserts; the converter and its audit are in the repo) the gain is **+2.4 pt, CI [-0.8, +5.5]**: indistinguishable from zero. Failure analysis shows what changed on MBPP: the base model defined a differently-named function in 9.2% of samples (GRPO: 0.6%), and when the function exists the pass rate rose from 47.7% to 62.9%.

## Honest conclusion
GRPO with unit-test rewards made the model clearly better at **MBPP-style tasks** (+15 pt, robust to seeds and renaming) and did not measurably improve HumanEval. We do not claim "better at coding". A second training source is the natural next experiment.

Next: [which training examples matter?](04_hard_mining_and_mining_ablation.md)
