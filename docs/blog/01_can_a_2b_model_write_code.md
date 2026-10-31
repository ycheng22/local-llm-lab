# 1. Can a 2B model write code? Building a research lab on a free GPU

*Series: small-scale LLM capability research on Kaggle's free T4 x2 (30 GPU hours per week). Numbers are from this repository; every table cites a notebook or experiment file.*

## The setup
We want to know how much a small model (Qwen3.5-2B) can improve at writing Python when we train it with unit tests as the teacher. The constraints are deliberate: a free Kaggle session (two 15.6 GB T4 GPUs, no native bf16), a 100-task held-out benchmark that is never touched during training, and a rule that every experiment gets a unique ID, a saved config and a reported cost.

Two engineering decisions paid for themselves immediately:

- **fp16, not bf16.** A T4 has no native bf16. In notebook 01b the same model generated 72.5 tokens/s in fp16 against 30.0 in bf16, and a LoRA step took 1.9 s instead of 6.4 s. The 2B model needs only 3.8 GB in 16-bit, so there is no reason to quantise.
- **Generated code never runs on the developer's machine.** Candidate programs run in Kaggle's isolated, internet-off container through a subprocess verifier with CPU/memory limits and a 5 s timeout.

## The first result was a bug, and the lesson is bigger than the bug
The first MBPP baseline gave **2.9% Pass@1**. A 2B model that bad would make the whole project pointless. Looking at the failures: **84% of them were `NameError`**. The prompt described the task in English but never said what the function is called, while the tests call `prime_num(...)`. The model wrote a perfectly good `is_prime`.

Adding one example assert to the prompt ("Your function must satisfy this example: `assert prime_num(5) == True`") changed the picture:

| Prompt | Pass@1 | Pass@4 | Pass@8 |
| :--- | ---: | ---: | ---: |
| v1: no function name | 2.9% | 6.0% | 7.0% |
| v2: one visible example | **40.6%** | 61.4% | 69.0% |

Same model, same tasks, same sampling. **A benchmark measures the model plus the prompt contract.** Later experiments show this contract matters even more than it looks (post 3).

## What else the audit found
- Our synthetic benchmark is saturated: 99.6% Pass@1 for the untrained model, and train/dev/test items overlap. It cannot separate methods, so we keep it only as a regression check ("OOD Pass@1").
- 14 of 420 MBPP tasks fail their own reference solution and were excluded before any model saw them.

## Takeaway
Before any training, audit the data and the prompt. A 14x change in the baseline came from one sentence. Next: [does supervised fine-tuning help?](02_teaching_a_small_model_sft.md)
