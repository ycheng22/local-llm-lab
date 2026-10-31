# 9. Train smarter or use a bigger model? 2B + RL vs 4B (and a 90% truncation trap)

*Series part 9. Experiments: x03 (4B baseline, 1.45 GPU h), x04 (token cap, about 2 GPU h), x08s (4B GRPO calibration, 0.32 GPU h).*

## The question
GRPO gave the 2B model about +15 points. What would simply using Qwen3.5-4B give us?

## The first answer was wrong
With the same settings as every other run (384 new tokens, 8 samples per task) the 4B model scored 53.8% Pass@1, a +13.1 pt gain over 2B base (CI [+6.4, +20.3]). At Pass@1 that looked like a draw with 2B+GRPO, and Pass@4/8 favoured the trained 2B model. A clean story: "RL on 256 examples beats doubling the parameters".

But the log said **90.4% of 4B samples hit the 384-token cap** (2B: 8.4%), and 27.9% were syntax errors. Qwen3.5-4B writes a long reasoning preamble ("The user wants me to...") before the code, so it was being cut off before it could write the solution.

## Rerun with 768 tokens (x04)
Only the cap changed. For the 2B models, nothing: Pass@1 40.6 -> 40.8% (base) and 58.5 -> 59.5% (GRPO), truncation fell from about 8% to about 3%, but the truncated samples were rambling, not nearly correct. For 4B, a large change: **53.4% -> 68.5% on the first 73 tasks (+15.1 pt, CI [+8.0, +22.3])**. (The run hit its time limit during the 4B arm, so 73 of 100 tasks, 4 samples each.)

Corrected comparison on those 73 tasks at 768 tokens: **4B base 68.5% vs 2B GRPO 58.2%: +10.3 pt, CI [+2.4, +18.5]**. The bigger model wins. Doubling parameters gave about +27 points over 2B base; GRPO recovered more than half of that gap for 2.3 GPU hours.

## Why we cannot just train the 4B model
The 6-step calibration (x08s) showed 4B GRPO fits on a 16 GB T4 (12.0-13.7 GB) but takes 160-167 s per step, **about 9 hours for 200 steps**, against our pre-registered limit of 5 hours. The cause is generation: about 660 tokens per completion and half the decoding speed. A shorter run would use different hyperparameters and not be comparable, so we recorded a no-go instead.

## Takeaways
- A fixed generation budget is part of the experiment. A hidden confound (the 384 cap) reversed a conclusion for one model and did nothing for another. Always log the truncation rate.
- At 2B the trained model is a cost-effective substitute for more parameters; it is not a full one.
