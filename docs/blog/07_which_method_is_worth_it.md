# 7. Which method is worth its compute? The final scorecard

*Series part 7 (final). Produced by `python kaggle/tools/make_dashboard.py` from the pulled experiment outputs; figures in `reports/figures_final/`.*

## The scorecard (100 held-out MBPP tasks, 8 samples, 384 tokens)

| Model | Pass@1 | Pass@4 | Pass@8 | Train GPU h |
| :--- | ---: | ---: | ---: | ---: |
| 2B base | 40.6% | 61.4% | 69.0% | - |
| 2B SFT | 39.0% | 57.6% | 65.0% | 0.16 |
| 2B GRPO (3 seeds) | **55.4% +/- 3.3** | 71.8% | 76.0% | about 2.5 each |
| 2B GRPO on 97 learnable / 97 random tasks | 55.8% / 58.5% | 72.5% / 73.7% | 77.0% / 77.0% | 2.4 / 2.2 |
| 4B base (384-token cap, a lower bound) | 53.8% | 66.2% | 69.0% | - |

Return on training compute: SFT, negative; GRPO about **6 Pass@1 points per training GPU hour** (4.5 to 7.7 across seeds).

## The seven questions
1. **Does RL with unit-test rewards help?** Yes: about +15 pt, three seeds, no regression on the synthetic check.
2. **Does SFT help first?** No; GRPO from SFT is much worse than GRPO from base.
3. **Train or sample more?** One GRPO sample is worth about four base samples; trained model stays ahead at equal k.
4. **Bigger or trained-smaller?** 4B is better (about +27 over 2B base at an adequate token cap); GRPO closes more than half the gap.
5. **Does mining help?** It cleans the signal (61% -> 40% wasted groups) but not the accuracy; 97 random tasks match all 256.
6. **Does it generalise?** Survives function renaming; **does not transfer to HumanEval**.
7. **Do agents help?** At 4B (+23 pt on 39 tasks), not at 2B.

## What we would claim, and what we would not
We would claim: unit-test-reward RL on a few hundred tasks makes a 2B model reliably better at the task style it was trained on, cheaply. We would not claim general coding improvement; HumanEval says otherwise. We would not quote +17.9: it was the best seed.

## Limits and what to do next
100 evaluation tasks, one model family, one training source; 4B results are partial or lower bounds; the hypothesis that the model learned to match the example assert's format was deliberately left untested. The most valuable next experiment is a second training source (to test transfer to HumanEval), then 4B GRPO with a cheaper generation setup.

## Cost of the whole study
About 23.8 GPU hours over two weekly quotas of the free tier, including failures and calibration runs. Every number here is reproducible from the repository: frozen data with hashes, a notebook generator per experiment, saved configs and parent checkpoints.
