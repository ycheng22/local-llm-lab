可以。根据你现在的硬件，我会把方案进一步收紧成一个**真正能在 8GB VRAM + 48GB RAM laptop 上长期迭代的 research plan**。

而且我会改变前面一个建议：**不要一开始同时研究 2B 和 4B**。先用一个模型把训练闭环做出来，再用另一个模型做 scaling 对照。这样你最后得到的是“实验结果”，而不是一堆跑过的脚本。

---

# 0. 先定义你的硬件边界

你现在是：

* **NVIDIA GPU：8GB VRAM**
* **System RAM：48GB**
* GPU 可以使用 shared/system memory，但它不能等价替换 VRAM；一旦大量 spill 到 RAM，吞吐通常会明显下降。
* Hugging Face 的 bitsandbytes 目前支持 NVIDIA CUDA 下的 4-bit NF4/QLoRA；Unsloth 目前也明确支持 Windows/Linux 和 NVIDIA RTX 50 系列。([Hugging Face][1])

所以你的优化目标不是：

> “让最大模型跑起来”

而是：

> **让一个小模型可以高频率地进行“训练 → 评估 → 改数据 → 再训练”。**

这是完全不同的优化目标。

---

# 1. 我建议你的主实验模型

## Primary model：**Qwen3-1.7B**

我建议先用：

**`Qwen/Qwen3-1.7B`**

它是 Apache-2.0，而且官方模型页直接提供 Transformers / vLLM / local usage。([Hugging Face][2])

### 为什么不是直接 4B？

因为你的核心瓶颈不是“能不能装进去”，而是：

> **能不能便宜地做很多次 RL experiment。**

1.7B 对 QLoRA 和小规模 GRPO 的压力会小很多。

而且 Unsloth 当前文档给出的经验是，GRPO 建议至少从约 1.5B 参数级别开始；它还指出 GRPO 的显存压力会随着 context 和每个 prompt 的 generation 数量增加。([Unsloth - Train and Run Models Locally][3])

所以：

### 第一阶段

**1.7B = research model**

### 第二阶段

**4B = scaling validation**

---

# 2. 不要一上来研究“通用智能”

你的第一个 target capability 应该非常具体：

# **Python Code Reasoning**

为什么？

因为它天然带 verifier：

```text
model
  ↓
generate code
  ↓
run tests
  ↓
pass/fail
  ↓
reward
```

这比：

```text
model
→ judge says "7.5/10"
```

可靠太多。

---

# 3. 总体实验路线

整个项目我建议做成 8 个阶段：

```text
Phase 0  Environment
    ↓
Phase 1  Dataset + Verifier
    ↓
Phase 2  Baseline Evaluation
    ↓
Phase 3  QLoRA SFT
    ↓
Phase 4  RLVR / GRPO
    ↓
Phase 5  Hard Example Mining
    ↓
Phase 6  Test-Time Compute
    ↓
Phase 7  Tool-Use Agent
    ↓
Phase 8  4B Scaling
```

最重要的是：

> **每一步都必须能单独证明“到底有没有提升”。**

---

# 4. Phase 0 — 建环境

我建议你的 AI IDE 直接生成一个项目：

```text
local-llm-lab/
│
├── configs/
├── data/
├── models/
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── reward/
│   ├── mining/
│   └── agent/
│
├── experiments/
│   ├── exp00_baseline/
│   ├── exp01_sft/
│   ├── exp02_rlvr/
│   ├── exp03_mining/
│   └── exp04_search/
│
├── logs/
├── checkpoints/
└── README.md
```

### 推荐 software stack

```text
Python
PyTorch
Transformers
PEFT
TRL
bitsandbytes
datasets
accelerate
Unsloth
pytest
```

TRL 当前直接支持 SFT、GRPO、DPO；GRPO 还已经支持 tools / environments，也就是后面的 agent experiment 可以直接延伸过去。([Hugging Face][4])

---

# 5. Phase 1 — 先做你的 Dataset + Verifier

这是整个项目最重要的一步之一。

不要先训练模型。

先把下面两个东西做出来：

## A. Problem generator

你的数据格式：

```json
{
  "id": "task_000001",
  "prompt": "Write a Python function ...",
  "starter_code": "...",
  "tests": "...",
  "difficulty": "medium",
  "category": "array"
}
```

---

## B. Verifier

核心函数：

```python
reward = verify(solution, tests)
```

第一版只允许：

```text
0.0 = fail
1.0 = pass
```

之后再扩展。

---

# 6. 你的训练数据应该准备什么？

不要一口气准备 100 万题。

我建议第一版：

### Train

**5,000**

### Validation

**500**

### Test

**1,000**

总共：

**6,500 examples**

已经足够做很多实验。

---

# 7. 数据组成

不要全是同一种题。

我建议：

| 类型                        |  比例 |
| ------------------------- | --: |
| Basic programming         | 20% |
| Data structures           | 20% |
| Algorithms                | 25% |
| Edge cases                | 15% |
| Debugging                 | 10% |
| Multi-step implementation | 10% |

---

# 8. 更重要的是：每道题都要有 Hidden Tests

例如：

```text
prompt:
Write merge_intervals()

visible example:
[[1,3],[2,5]]
→ [[1,5]]
```

但 verifier 还有：

```text
[]
[[1,1]]
[[5,7],[1,2]]
negative values
duplicates
large input
```

模型只看到：

```text
prompt
```

而不知道：

```text
hidden tests
```

这样才能防止 memorization。

---

# 9. Dataset 要记录 difficulty

每道题存：

```json
{
  "difficulty": 0.0,
  "pass_rate": 0.35
}
```

这个字段以后非常重要。

因为你会发现：

> **真正有价值的数据不是“最难的问题”，而是模型当前恰好解决不了的问题。**

---

# 10. 数据来源怎么解决？

我建议用 **两条线同时做**。

### Track A：公开 benchmark / public datasets

用来建立 baseline。

比如可以参考：

* HumanEval
* MBPP
* APPS
* CodeContests
* BigCodeBench

但是：

> **不要把最终 test 直接混入训练。**

---

### Track B：Synthetic data

用更强的模型生成：

```text
problem
solution
tests
```

然后：

```text
solution
   ↓
run tests
   ↓
pass?
```

只有真正通过 verifier 的数据才进入训练集。

---

# 11. Phase 2 — Base Model Baseline

现在才加载：

**Qwen3-1.7B**

先完全不训练。

你的 `evaluate.py` 要做：

```text
500-1000 test problems
```

并记录：

### 必须记录的指标

| Metric                | 含义         |
| --------------------- | ---------- |
| Pass@1                | 一次生成成功率    |
| Pass@4                | 4 次里至少成功一次 |
| Pass@8                | 8 次里至少成功一次 |
| Avg generation tokens | 平均输出长度     |
| Generation latency    | 每题生成耗时     |
| Peak VRAM             | 峰值显存       |
| RAM usage             | 系统内存       |
| Timeout rate          | 超时比例       |

---

# 12. 一个非常重要的实验纪律

以后所有实验都必须跑：

```text
SAME TEST SET
SAME PROMPT
SAME EVALUATOR
```

比如：

```text
test_v1.jsonl
```

以后永远不要改。

---

# 13. 再建立一个 Experiment Log

每个实验都记录：

```json
{
  "experiment_id": "exp01",
  "model": "Qwen3-1.7B",
  "method": "baseline",
  "dataset_version": "v1",
  "steps": 0,
  "pass1": 0.32,
  "pass4": 0.47,
  "pass8": 0.55,
  "avg_tokens": 183,
  "peak_vram_gb": 6.4,
  "train_hours": 0
}
```

---

# 14. Phase 3 — QLoRA SFT

现在才训练。

我建议：

## Model

```text
Qwen3-1.7B
```

## Quantization

```text
4-bit NF4
```

Hugging Face 当前文档明确把 NF4 作为 4-bit base model training 的推荐配置，并说明 QLoRA 是通过 4-bit quantized base + LoRA trainable weights 实现。([Hugging Face][1])

---

# 15. 第一版 SFT 参数

不要追求最优。

追求：

> **快速得到可靠 baseline。**

建议：

```yaml
max_seq_length: 2048
micro_batch_size: 1
gradient_accumulation: 8
learning_rate: 1e-5
epochs: 1
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
gradient_checkpointing: true
```

具体值要根据你的 GPU 和 Qwen3 chat template 实测，不要把它们当成“唯一正确答案”。

---

# 16. SFT 数据不要太多

第一轮：

### **5K examples**

足够。

然后训练：

```text
Base
   ↓
SFT
```

比较：

|            | Base | SFT |
| ---------- | ---: | --: |
| Pass@1     |    ? |   ? |
| Pass@4     |    ? |   ? |
| Pass@8     |    ? |   ? |
| Avg tokens |    ? |   ? |

---

# 17. 这里定义第一个 Go / No-Go

如果：

```text
SFT Pass@1 > Base Pass@1
```

并且：

```text
OOD test 也提升
```

继续。

如果：

```text
train ↑
IID ↑
OOD 不变
```

说明：

> **你的 SFT 数据正在过拟合。**

不要急着做 RL。

---

# 18. Phase 4 — RLVR

现在才进入项目真正的核心。

使用：

# **GRPO**

TRL 当前 `GRPOTrainer` 支持多个 reward functions，并且支持 environment/tool based agent training。([Hugging Face][5])

---

# 19. 第一版 reward 极其简单

```text
passed hidden tests
    ↓
1.0

failed
    ↓
0.0
```

先不要 reward shaping。

---

# 20. GRPO 参数

你的 GPU 只有 8GB，所以我会先：

```yaml
num_generations: 4
max_prompt_length: 1024
max_completion_length: 384

per_device_train_batch_size: 1
gradient_accumulation_steps: 4

gradient_checkpointing: true
```

TRL 当前文档指出 `num_generations` 默认可以是 8，而 generation 数和 completion/context length 都会直接影响训练成本；你的机器不适合一开始就使用很大的 generation group。([Hugging Face][5])

---

# 21. 这里你要记录的不只是 reward

这是非常重要的。

每个 training step：

```text
step
mean_reward
reward_std
pass_rate
completion_length
KL
entropy
grad_norm
```

尤其：

### Reward

```text
0.31
0.34
0.39
0.44
...
```

### Reward variance

如果：

```text
reward_std → 0
```

要警惕：

> **所有 samples 都开始失败，或者都开始成功。**

这会让 GRPO 学习信号变差。

---

# 22. 重点记录一个指标：Pass@1 vs Pass@4

这是你整个研究中非常关键的一对指标。

假设：

```text
Base

Pass@1 = 30%
Pass@4 = 48%
```

RL 后：

```text
Pass@1 = 43%
Pass@4 = 71%
```

说明模型不仅自己更强，而且：

> 更容易产生多个正确 reasoning path。

---

# 23. Phase 5 — Hard Example Mining

这个阶段我认为：

# **可能是你整个项目 ROI 最高的一步。**

每训练一个 checkpoint：

```text
model v1
    ↓
run 10K problems
    ↓
collect failures
```

然后把失败题分成：

```text
A. 永远不会
B. 偶尔成功
C. 大部分成功
D. 已经解决
```

---

# 24. 最有价值的是 B 类

比如：

```text
4 generations:

wrong
wrong
correct
wrong
```

也就是：

```text
Pass@1 = 25%
Pass@4 = 100%
```

这种问题说明：

> 模型已经“知道怎么做”，只是 execution / reasoning 不稳定。

这是非常好的 RL 数据。

---

# 25. 建立自动 difficulty score

可以简单定义：

$$
Difficulty(x)=1-Pass@4(x)
$$

例如：

```text
Pass@4 = 1.0
difficulty = 0

Pass@4 = 0.5
difficulty = 0.5

Pass@4 = 0
difficulty = 1
```

然后下一轮训练：

```text
70% hard/borderline
30% regular
```

而不是：

```text
100% random
```

---

# 26. 你要做一个非常重要的 Ablation

训练三组：

### Experiment A

Random data

### Experiment B

Hard-only

### Experiment C

Random + hard

最终：

```text
                Pass@1
                  │
          A       B       C
```

这样你才知道：

> hard-example mining 到底有没有贡献。

---

# 27. Phase 6 — Test-Time Compute

现在不要训练。

单纯增加 inference compute。

比如测试：

```text
K=1
K=2
K=4
K=8
K=16
```

每个 K：

```text
generate K solutions
 ↓
run verifier
 ↓
pick best
```

记录：

|  K | Pass@K | Avg tokens | Time/problem |
| -: | -----: | ---------: | -----------: |
|  1 |        |            |              |
|  2 |        |            |              |
|  4 |        |            |              |
|  8 |        |            |              |
| 16 |        |            |              |

---

# 28. 然后做一张最重要的图

横轴：

```text
Inference compute
```

纵轴：

```text
Pass@K
```

如果你得到：

```text
accuracy
  │
  │                ●
  │           ●
  │       ●
  │    ●
  │ ●
  └────────────────────
       compute
```

说明你的模型存在：

# **test-time scaling**

这本身就是很有价值的实验结果。

---

# 29. 接下来做一个更有意思的实验：Verifier-guided Search

不要简单：

```text
generate 8
choose one
```

做成：

```text
generate candidate
      ↓
run tests
      ↓
failure?
      ↓
give error to model
      ↓
revise
      ↓
run tests
      ↓
success
```

这已经开始变成：

# **Agentic coding**

---

# 30. Phase 7 — Tool-use Agent

你的 environment 提供三个 tool：

```text
read_file()
write_file()
run_tests()
```

模型可以：

```text
inspect
 ↓
edit
 ↓
run
 ↓
observe failure
 ↓
edit again
```

TRL 当前 GRPO 环境接口已经支持这种 stateful environment，环境可以自己维护状态并通过 `get_reward()` 计算 episode reward。([Hugging Face][5])

---

# 31. Agent reward 可以设计成

第一版：

```text
all tests pass
    +1

otherwise
    0
```

第二版：

```text
tests_pass
+0.1 no_syntax_error
+0.1 fewer_steps
```

第三版才考虑：

```text
solution quality
code quality
efficiency
```

不要一开始 reward engineering 过度。

---

# 32. 关键的新 metric：Task Completion Rate

从这里开始，不要只看 token-level 指标。

增加：

### Task Completion Rate

```text
完成整个任务的 %
```

例如：

```text
Base       18%
SFT        27%
RLVR       38%
Agent RL   49%
```

这个指标比 loss 有意义得多。

---

# 33. 再加两个 agent 指标

## Steps-to-success

成功任务平均需要多少 tool calls：

```text
Base Agent = 14.2
RL Agent   = 8.7
```

### Recovery rate

模型第一次失败后：

> **能不能根据错误继续恢复？**

例如：

```text
first attempt failed
        ↓
second attempt succeeded
```

这就是非常有价值的 agent capability。

---

# 34. Phase 8 — 4B Scaling Experiment

等 1.7B 完成以后，再切：

# **Qwen3-4B-Instruct-2507**

当前模型页确认它是 Apache-2.0，并直接支持 Transformers；所以它很适合做你第二阶段的规模验证。([Hugging Face][6])

---

# 35. 不要重新发明全部实验

只重复：

```text
4B Base
4B SFT
4B RLVR
```

然后比较：

|                  | 1.7B | 4B |
| ---------------- | ---: | -: |
| Base Pass@1      |      |    |
| SFT Pass@1       |      |    |
| RLVR Pass@1      |      |    |
| Agent completion |      |    |
| Train time       |      |    |
| Peak VRAM        |      |    |

你的问题变成：

> **能力提升方法是否随着模型规模增加而保持有效？**

这才是很有意思的 scaling experiment。

---

# 36. 你最终应该维护这张总表

这是整个项目最重要的 artifact。

| Exp | Model | Training  | Data | Pass@1 | Pass@4 | Pass@8 | OOD | Completion | Time | VRAM |
| --- | ----- | --------- | ---: | -----: | -----: | -----: | --: | ---------: | ---: | ---: |
| E0  | 1.7B  | Base      |    0 |        |        |        |     |            |      |      |
| E1  | 1.7B  | SFT       |   5K |        |        |        |     |            |      |      |
| E2  | 1.7B  | RLVR      |   5K |        |        |        |     |            |      |      |
| E3  | 1.7B  | RL+Mining |   5K |        |        |        |     |            |      |      |
| E4  | 1.7B  | +Search   |   5K |        |        |        |     |            |      |      |
| E5  | 1.7B  | Agent RL  |   5K |        |        |        |     |            |      |      |
| E6  | 4B    | Base      |    0 |        |        |        |     |            |      |      |
| E7  | 4B    | SFT       |   5K |        |        |        |     |            |      |      |
| E8  | 4B    | RLVR      |   5K |        |        |        |     |            |      |      |

---

# 37. 另外一定记录“成本”

你以后比较方法的时候，不应该只说：

> RL +10%.

应该说：

> **RL +10% capability，增加 8 GPU-hours。**

所以每个 experiment 记录：

```text
GPU hours
CPU hours
wall clock
electricity estimate
peak VRAM
peak RAM
tokens trained
tokens generated
```

甚至可以算：

$$
ROI = \frac{\Delta Pass@1}{GPU\ Hours}
$$

这样你就真正拥有：

# **Capability / Compute Efficiency**

指标。

---

# 38. 我甚至建议做一个“Capability Gain per Dollar”

假设：

```text
SFT:
+5% Pass@1
3 GPU-hours

RLVR:
+11%
15 GPU-hours

Hard Mining:
+7%
5 GPU-hours
```

那么：

```text
SFT       1.67 % / GPU-hour
RLVR      0.73
Mining    1.40
```

你就真的可以回答：

> **什么方法 ROI 最高？**

这比凭感觉说：

> “RL 很强。”

科学很多。

---

# 39. 你的实验顺序，我会严格按照这个

## Stage A — Infrastructure

```text
[ ] CUDA/PyTorch
[ ] HF / TRL / Unsloth
[ ] Qwen3-1.7B loading
[ ] 4-bit inference
```

---

## Stage B — Dataset

```text
[ ] 5K train
[ ] 500 dev
[ ] 1K test
[ ] hidden tests
[ ] difficulty label
```

---

## Stage C — Baseline

```text
[ ] Pass@1
[ ] Pass@4
[ ] Pass@8
[ ] latency
[ ] VRAM
```

---

## Stage D — SFT

```text
[ ] QLoRA
[ ] 1 epoch
[ ] compare Base vs SFT
```

---

## Stage E — RLVR

```text
[ ] GRPO
[ ] 4 generations
[ ] pytest reward
[ ] reward logging
[ ] compare SFT vs RLVR
```

---

## Stage F — Mining

```text
[ ] collect failures
[ ] estimate difficulty
[ ] borderline set
[ ] hard-set training
[ ] ablation
```

---

## Stage G — Test-Time Compute

```text
[ ] K=1
[ ] K=2
[ ] K=4
[ ] K=8
[ ] K=16
```

---

## Stage H — Agent

```text
[ ] read_file
[ ] write_file
[ ] run_tests
[ ] recovery
[ ] multi-step task
```

---

## Stage I — Scale

```text
[ ] 4B
[ ] SFT
[ ] RLVR
[ ] compare scaling
```

---

# 40. 对你的电脑，我特别建议这几个“省显存规则”

### Rule 1

**context 默认 1024–2048**

不要一开始 8K。

---

### Rule 2

**batch size = 1**

靠 gradient accumulation。

Unsloth 当前文档也特别提醒，OOM 最常见原因之一就是 batch size/context 太高。([Unsloth - Train and Run Models Locally][7])

---

### Rule 3

**GRPO generations 从 4 开始**

不要 16。

---

### Rule 4

**尽量 gradient checkpointing**

Unsloth 目前的 GRPO 文档明确指出，其 gradient checkpointing / memory-efficient kernels 会显著降低 GRPO 显存占用，并且部分中间激活可以异步 offload 到 system RAM。([Unsloth - Train and Run Models Locally][3])

---

### Rule 5

**尽量不要同时跑 IDE + training + browser-heavy workload**

因为你的 48GB RAM 会被快速吃掉。

---

# 41. 我最推荐你让 AI IDE 写代码的方式

不要对 Cursor/Claude Code 说：

> “帮我做一个 GRPO training project。”

一次生成太多东西，很容易变成一团。

应该把任务拆开：

### Prompt 1

> Build a reproducible Python benchmark harness for evaluating Qwen3-1.7B on Python coding tasks. Implement Pass@1/4/8, deterministic test splits, JSONL experiment logging, and peak GPU/RAM measurement.

### Prompt 2

> Implement a secure local Python verifier that runs generated solutions against hidden pytest tests with subprocess isolation, timeout, stdout/stderr capture, and deterministic results.

### Prompt 3

> Implement QLoRA SFT using TRL and bitsandbytes for Qwen3-1.7B. Use a YAML config and log all hyperparameters to an experiment manifest.

### Prompt 4

> Implement GRPO/RLVR using TRL. Reward generated coding solutions based on hidden pytest results. Start with num_generations=4 and max_completion_length=384. Log reward mean/std, completion length, KL, entropy, and Pass@1/4/8.

### Prompt 5

> Implement hard-example mining. Run the current checkpoint on 10K tasks, estimate per-task Pass@4, classify tasks into solved/borderline/hard/impossible, and generate a new training manifest without touching the frozen test set.

这种方式会好很多。

---

# 42. 最重要的“禁止事项”

你的 AI IDE 很可能会主动给你很多 fancy features。

第一阶段全部拒绝：

❌ DPO
❌ multi-agent
❌ RAG
❌ 100K context
❌ multimodal
❌ LoRA rank 128
❌ 16-generation GRPO
❌ giant dataset
❌ custom attention
❌ complicated reward model

你现在研究的是：

# **“什么训练机制最能提升一个小模型的可验证能力？”**

不是：

# “我要做一个 mini OpenAI。”

---

# 43. 如果一切顺利，你最终希望看到的不是某个绝对数字

而是这种趋势：

```text
                 Pass@1

Base 1.7B       ███████
SFT             █████████
RLVR            ████████████
Mining          ██████████████
Agent           █████████████████
```

同时：

```text
                 Pass@8

Base            ███████████
SFT             █████████████
RLVR            ███████████████
Mining          █████████████████
```

而最漂亮的结果是：

> **Pass@1 上升，同时 Pass@8 上升，而且 OOD 也上升。**

那才意味着你不是单纯 overfit。

---

# 44. 你的第一阶段“成功标准”

我会把目标设置为：

### Minimum success

```text
RLVR > SFT > Base
```

至少在：

```text
Pass@1
OOD Pass@1
```

两个指标都成立。

### Strong success

```text
RLVR improves Pass@1 by 10%+
```

并且：

```text
OOD also improves
```

### Excellent success

发现：

```text
Hard-example mining
+
RLVR
```

明显优于：

```text
RLVR alone
```

并且：

```text
test-time compute
```

还能进一步 scaling。

到了这里，你这个项目就已经不只是“自己 fine-tune 模型玩玩”了，而是一个相当正经的 **small-scale reasoning-model research pipeline**。

---

# 45. 最后给你一个我认为最重要的实验哲学

你现在只有一台 8GB laptop。

这其实反而逼着你采用一个非常好的研究习惯：

> **不要追求“大实验”，追求“信息密度最高的实验”。**

例如：

```text
Experiment A
5K SFT
6 hours
+3%

Experiment B
2K RLVR
8 hours
+8%

Experiment C
hard mining
3 hours
+6%
```

那么 C 才可能是最有价值的发现。

你的最终目标不是：

> **训练一个很大的模型。**

而是找到：

$$
\boxed{
\text{Which intervention gives the highest capability gain per unit compute?}
}
$$

这正好和你最开始问的：

> “现在 GPT / Claude / Opus 到底靠什么变强？”

对应起来。

---

## 我建议你的第一阶段最终固定成这个配置

```text
GPU                 8GB
RAM                 48GB

Model               Qwen3-1.7B
Quantization        4-bit NF4
Fine-tuning         QLoRA
RL                  GRPO
Reward              hidden pytest
Train set           5,000
Dev                 500
Test                1,000
Context             1K-2K
GRPO generations    4
Max completion      384
Primary metric      Pass@1
Secondary           Pass@4 / Pass@8
Generalization      OOD Pass@1
Agent metric        Task Completion Rate
Efficiency          GPU-hours / +1% capability
```

TRL 当前 GRPO 既支持自定义 reward，也支持 environment/tool-based training，所以你可以从最简单的代码 verifier 一直升级到真正的 coding agent，而不必推翻整个框架。([Hugging Face][5])

**下一步最值得做的不是再讨论架构，而是把 `Phase 0–2` 真正落地。** 我可以直接给你一份适合 **Windows + 8GB NVIDIA + 48GB RAM + Cursor** 的完整项目 blueprint，包括 **目录、`pyproject.toml`、环境安装、数据 JSON schema、verifier 设计、baseline evaluator、experiment YAML，以及给 Cursor 的逐个 coding prompts**。

[1]: https://huggingface.co/docs/transformers/quantization/bitsandbytes?utm_source=chatgpt.com "Bitsandbytes · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/config.json?utm_source=chatgpt.com "config.json · Qwen/Qwen3-1.7B at main"
[3]: https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide?utm_source=chatgpt.com "Reinforcement Learning (RL) Guide | Unsloth Documentation"
[4]: https://huggingface.co/docs/trl/quickstart?utm_source=chatgpt.com "Quickstart · Hugging Face"
[5]: https://huggingface.co/docs/trl/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer · Hugging Face"
[6]: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/main?utm_source=chatgpt.com "Qwen/Qwen3-4B-Instruct-2507 at main"
[7]: https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements?utm_source=chatgpt.com "Unsloth Requirements | Unsloth Documentation"



# ============================================
给你一份适合 Windows + 8GB NVIDIA + 48GB RAM + Cursor 的完整项目 blueprint，包括 目录、pyproject.toml、环境安装、数据 JSON schema、verifier 设计、baseline evaluator、experiment YAML，以及给 Cursor 的逐个 coding prompts。


可以。下面我直接把它设计成一个**可以交给 Cursor / Claude Code 逐步实现的本地 research repo**。目标不是“把所有东西一次写完”，而是让每个阶段都能独立运行、记录、回滚、比较。

我会基于你现在的：

* **Windows**
* **NVIDIA 8GB VRAM**
* **48GB system RAM**
* GPU 可以发生 shared-memory / CPU offload
* 目标模型：**Qwen3-1.7B**
* 后续再用 4B 做 scaling experiment

Qwen3-1.7B 官方模型页目前提供 Transformers 使用方式，许可证为 Apache-2.0。([Hugging Face][1])
TRL 当前原生支持 SFT、GRPO，以及带 tools/environment 的 agent training，所以后面不需要换训练框架。([Hugging Face][2])
bitsandbytes 当前官方文档支持 NVIDIA GPU 上的 4-bit quantization，并要求 Python >=3.10、PyTorch >=2.4。([Hugging Face][3])

---

# 1. 最终项目结构

我建议 Cursor 最终生成：

```text
local-llm-lab/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
│
├── configs/
│   ├── baseline.yaml
│   ├── sft_v1.yaml
│   ├── grpo_v1.yaml
│   └── eval.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
│
├── scripts/
│   ├── download_model.py
│   ├── validate_dataset.py
│   └── run_experiment.py
│
├── src/
│   └── llm_lab/
│       ├── __init__.py
│       │
│       ├── config.py
│       ├── logging_utils.py
│       ├── hardware.py
│       │
│       ├── inference/
│       │   ├── generate.py
│       │   └── sampling.py
│       │
│       ├── evaluation/
│       │   ├── evaluator.py
│       │   ├── metrics.py
│       │   └── splits.py
│       │
│       ├── verifier/
│       │   ├── interface.py
│       │   ├── python_verifier.py
│       │   └── docker_verifier.py
│       │
│       ├── data/
│       │   ├── schema.py
│       │   ├── loader.py
│       │   ├── generator.py
│       │   └── miner.py
│       │
│       ├── training/
│       │   ├── sft.py
│       │   └── grpo.py
│       │
│       └── reporting/
│           ├── experiment.py
│           └── plots.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_metrics.py
│   ├── test_verifier.py
│   └── test_smoke.py
│
├── experiments/
│   ├── exp00_baseline/
│   ├── exp01_sft/
│   ├── exp02_grpo/
│   ├── exp03_mining/
│   └── exp04_search/
│
└── reports/
    ├── experiment_log.csv
    └── figures/
```

**这里最重要的是 `experiments/`。**

每次实验都产生一个独立目录：

```text
experiments/exp02_grpo/
├── config.yaml
├── metrics.json
├── train.log
├── generations.jsonl
├── checkpoint/
└── summary.md
```

这样以后你不会陷入：

> “这个 checkpoint 是哪个参数训出来的？”

---

# 2. 技术栈

我建议：

```text
Python 3.11
PyTorch
Transformers
TRL
PEFT
bitsandbytes
datasets
accelerate
safetensors
PyYAML
pydantic
pandas
numpy
matplotlib
pytest
psutil
GPUtil / pynvml
```

**Windows 上建议 Python 3.11。**

依赖尽量由 `uv` 管理；PyTorch 本身则根据你 NVIDIA 驱动支持的 CUDA wheel 安装。

---

# 3. `pyproject.toml`

不要让 Cursor 把所有版本写死成某个网上教程里的旧版本。ML stack 变化很快。

第一版：

```toml
[project]
name = "local-llm-lab"
version = "0.1.0"
description = "Small-scale LLM capability research on a consumer GPU"
readme = "README.md"
requires-python = ">=3.11,<3.13"

dependencies = [
    "transformers>=4.55",
    "trl>=0.20",
    "peft>=0.17",
    "bitsandbytes>=0.46",
    "accelerate>=1.10",
    "datasets>=4.0",
    "safetensors>=0.5",
    "pyyaml>=6.0",
    "pydantic>=2.10",
    "numpy>=2.0",
    "pandas>=2.2",
    "matplotlib>=3.9",
    "psutil>=6.0",
    "pynvml>=12.0",
    "pytest>=8.0",
    "orjson>=3.10",
    "tqdm>=4.67",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.12",
    "mypy>=1.16",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"
```

注意：**PyTorch 不放这里面**，因为它需要根据 NVIDIA/CUDA 环境单独安装。

bitsandbytes 官方目前要求 Python >=3.10、PyTorch >=2.4，并提供 NVIDIA CUDA 支持。([Hugging Face][3])

---

# 4. Windows 环境安装

PowerShell：

```powershell
mkdir local-llm-lab
cd local-llm-lab

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

或者使用 `uv`：

```powershell
pip install uv

uv venv --python 3.11
.venv\Scripts\Activate.ps1
```

然后：

### 第一步单独安装 PyTorch

**不要让 Cursor 猜 CUDA 版本。**

打开 PyTorch 官方安装页面，选择：

```text
Windows
Pip
Python
CUDA = your supported version
```

然后执行官方给你的安装命令。

之后：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory / 1024**3)"
```

你应该看到：

```text
True
<你的 NVIDIA GPU>
~8 GB
```

---

# 5. 安装其余依赖

```powershell
pip install -e ".[dev]"
```

然后：

```powershell
python -c "import transformers, trl, peft, bitsandbytes; print('OK')"
```

最后：

```powershell
pytest
```

---

# 6. 我强烈建议加一个 `doctor.py`

建立：

```text
scripts/doctor.py
```

它检查：

```text
Python
PyTorch
CUDA
GPU name
VRAM
system RAM
Transformers
TRL
PEFT
bitsandbytes
disk free space
Docker
```

输出：

```text
=== Local LLM Lab Environment ===

Python              3.11.x        OK
PyTorch             x.x.x         OK
CUDA available      True          OK
GPU                 RTX xxxx
VRAM                7.9 GB
System RAM          47.8 GB

Transformers        x.x.x         OK
TRL                 x.x.x         OK
PEFT                x.x.x         OK
bitsandbytes        x.x.x         OK

Docker              available     OK

STATUS: READY
```

---

# 7. Model 选择

第一阶段：

# `Qwen/Qwen3-1.7B`

官方模型页目前仍提供该模型及其 Transformers 用法，许可证 Apache-2.0。([Hugging Face][1])

下载代码：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
)
```

但是 **不要直接默认开启它的 thinking 模式** 做第一轮 coding baseline。

你的第一个研究问题是：

> RLVR 能否提升可验证 coding capability？

不是：

> thinking mode + sampling + RLVR 混在一起会不会更好？

变量一次只改一个。

---

# 8. Data Schema

我建议每道任务都定义成：

```json
{
  "id": "py_000001",
  "version": 1,
  "category": "array",
  "difficulty": 0.42,

  "prompt": "Write a Python function called merge_intervals(...)",

  "starter_code": "",

  "visible_tests": [
    {
      "input": "[[1,3],[2,6]]",
      "expected": "[[1,6]]"
    }
  ],

  "hidden_tests": [
    {
      "input": "[]",
      "expected": "[]"
    },
    {
      "input": "[[1,1]]",
      "expected": "[[1,1]]"
    }
  ],

  "reference_solution": "def merge_intervals(...): ...",

  "metadata": {
    "source": "synthetic",
    "generator": "teacher_model_v1"
  }
}
```

但是：

## `reference_solution` 不进入 prompt。

也不要进入 test set 的训练 artifact。

---

# 9. Train / Dev / Test

固定：

```text
train = 5,000
dev   = 500
test  = 1,000
```

然后永远：

```text
test.jsonl
```

**freeze。**

以后：

```text
Base
SFT
RL
Mining
Agent
4B
```

全部跑同一个 `test.jsonl`。

---

# 10. Dataset 设计

第一版：

| Category         | Count |
| ---------------- | ----: |
| Basic Python     | 1,000 |
| Arrays / strings | 1,000 |
| Data structures  | 1,000 |
| Algorithms       | 1,250 |
| Edge cases       |   750 |

总计：

**5,000**

第二阶段再加入：

```text
debugging
multi-function
repo-level
tool use
```

---

# 11. Data 来源

不要只下载一个 benchmark 然后 fine-tune。

我建议：

### 20–30%

公开数据 / benchmark-inspired。

### 70–80%

synthetic + verified。

也就是说：

```text
Teacher model
      ↓
generate problem
      ↓
generate reference solution
      ↓
generate hidden tests
      ↓
run verifier
      ↓
pass?
 ┌────┴────┐
yes       no
 ↓         ↓
keep     discard
```

---

# 12. 非常重要：Verifier 安全

这是整个项目里我最不建议偷懒的地方。

**不要直接让模型生成的 Python 在你的 Windows host 上运行。**

因为模型输出就是不可信代码。

正确结构：

```text
Model
 ↓
candidate.py
 ↓
Docker container
 ↓
network disabled
 ↓
CPU limit
 ↓
RAM limit
 ↓
timeout
 ↓
pytest
 ↓
reward
```

Docker Desktop 在 Windows 上可以使用 WSL2 backend。

第一阶段就先把 verifier 和容器隔离做好。否则你的 RL pipeline 本身就存在安全漏洞。

---

# 13. Verifier interface

建立：

```python
class Verifier:
    def verify(
        self,
        task: dict,
        candidate_code: str,
    ) -> dict:
        ...
```

统一返回：

```json
{
  "reward": 1.0,
  "passed": true,
  "tests_passed": 12,
  "tests_total": 12,
  "timeout": false,
  "syntax_error": false,
  "runtime_error": false,
  "stderr": "",
  "duration_seconds": 0.83
}
```

---

# 14. Reward 第一版

**就这么简单：**

```python
reward = 1.0 if all_tests_pass else 0.0
```

不要一上来：

```text
+ correctness
+ elegance
+ token length
+ style
+ complexity
+ comments
```

否则你以后无法知道：

> 是模型真的变强了，还是 reward hacking。

---

# 15. Reward 第二版

等基础 RL 跑通以后再：

```python
reward = (
    1.0 * correctness
    + 0.05 * format_valid
)
```

之后再实验：

```text
efficiency
test coverage
tool usage
steps
```

每次只增加一个 reward component。

---

# 16. Baseline Evaluator

建立：

```text
src/llm_lab/evaluation/evaluator.py
```

调用方式：

```powershell
python -m llm_lab.evaluation.evaluator `
  --model Qwen/Qwen3-1.7B `
  --dataset data/test.jsonl `
  --num-samples 8 `
  --output experiments/exp00_baseline
```

输出：

```text
=== Evaluation ===

Tasks:              1000

Pass@1:             0.XX
Pass@4:             0.XX
Pass@8:             0.XX

Mean completion:    XXX tokens
Median completion:  XXX tokens

Timeout rate:       X.X%
Syntax error rate:  X.X%

Mean latency:       X.XX sec
Peak VRAM:          X.XX GB
Peak RAM:           XX.X GB
```

---

# 17. Metrics 的定义

你以后所有实验最核心的指标：

## Pass@1

一次生成就正确：

$$
Pass@1=\frac{\text{successful tasks}}{\text{tasks}}
$$

---

## Pass@K

K 次生成至少有一个成功。

：

```text
K = 1
K = 4
K = 8
```

---

# 18. 还要记录这些

### Capability

```text
Pass@1
Pass@4
Pass@8
OOD Pass@1
```

### Efficiency

```text
generation tokens
seconds / task
GPU hours
tokens / second
```

### Training health

```text
loss
reward
reward_std
entropy
KL
grad_norm
completion length
```

### Hardware

```text
peak VRAM
peak RAM
GPU utilization
temperature
```

---

# 19. Experiment tracking 文件

每个实验都生成：

```json
{
  "experiment_id": "exp02_grpo_v1",
  "model": "Qwen/Qwen3-1.7B",

  "base_checkpoint": "...",

  "dataset": "train_v1",
  "test_dataset": "test_v1",

  "method": "GRPO",

  "hyperparameters": {
    "num_generations": 4,
    "learning_rate": 1e-5,
    "max_prompt_length": 1024,
    "max_completion_length": 384
  },

  "metrics": {
    "pass_at_1": 0.0,
    "pass_at_4": 0.0,
    "pass_at_8": 0.0
  },

  "hardware": {
    "peak_vram_gb": 0.0,
    "peak_ram_gb": 0.0
  },

  "training": {
    "wall_hours": 0.0,
    "gpu_hours": 0.0,
    "tokens": 0
  }
}
```

---

# 20. `baseline.yaml`

```yaml
experiment:
  id: exp00_baseline
  description: "Qwen3-1.7B baseline"

model:
  id: Qwen/Qwen3-1.7B
  load_in_4bit: true
  device_map: auto

generation:
  max_new_tokens: 384
  temperature: 0.7
  top_p: 0.9
  do_sample: true

evaluation:
  dataset: data/test.jsonl
  num_samples: [1, 4, 8]

logging:
  output_dir: experiments/exp00_baseline
```

这里建议**固定 random seed**，同时另外做少量 seed robustness test。

---

# 21. Phase 0 的目标

第一阶段只回答：

> **我的电脑能否稳定完成本地 inference + evaluation？**

成功标准：

```text
[ ] model loads
[ ] GPU works
[ ] 1,000 tests execute
[ ] Pass@1/4/8 works
[ ] metrics saved
[ ] experiment reproducible
```

**没做到这一阶段，不要进入训练。**

---

# 22. Phase 1 — SFT

`configs/sft_v1.yaml`：

```yaml
experiment:
  id: exp01_sft_v1

model:
  id: Qwen/Qwen3-1.7B

quantization:
  enabled: true
  bits: 4
  quant_type: nf4
  compute_dtype: float16

lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05

training:
  dataset: data/train.jsonl
  eval_dataset: data/dev.jsonl

  max_seq_length: 2048

  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

  learning_rate: 0.00001
  num_train_epochs: 1

  gradient_checkpointing: true

  logging_steps: 10
  save_steps: 250

output:
  dir: experiments/exp01_sft_v1
```

QLoRA 使用 4-bit base model + LoRA 是这里的核心，NF4 是 Hugging Face 当前 bitsandbytes 文档推荐的 4-bit training 配置。([Hugging Face][3])

---

# 23. SFT 的实验不要只跑一次

第一轮：

```text
5K data
1 epoch
```

然后第二轮才做：

```text
10K data
```

比较：

```text
5K → 10K
```

---

# 24. SFT 需要记录什么？

每个 checkpoint：

```text
train loss
eval loss

Pass@1
Pass@4
Pass@8

OOD Pass@1

avg output tokens
GPU hours
VRAM
```

最终表：

| Model   | Pass@1 | Pass@4 | Pass@8 | OOD |
| ------- | -----: | -----: | -----: | --: |
| Base    |        |        |        |     |
| SFT 5K  |        |        |        |     |
| SFT 10K |        |        |        |     |

---

# 25. Phase 2 — GRPO / RLVR

TRL 当前 `GRPOTrainer` 原生支持 reward functions；它也支持 tools 和 environment-based agent training。([Hugging Face][2])

`grpo_v1.yaml`：

```yaml
experiment:
  id: exp02_grpo_v1

model:
  base_checkpoint: experiments/exp01_sft_v1/checkpoint-final

quantization:
  enabled: true
  bits: 4
  quant_type: nf4

lora:
  enabled: true
  r: 16
  alpha: 32

grpo:
  num_generations: 4

  max_prompt_length: 1024
  max_completion_length: 384

  learning_rate: 0.00001

  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4

  gradient_checkpointing: true

reward:
  type: pytest
  pass_reward: 1.0
  fail_reward: 0.0

dataset:
  train: data/train.jsonl
  eval: data/dev.jsonl

logging:
  output_dir: experiments/exp02_grpo_v1
  log_completions: true
```

---

# 26. 为什么第一版 GRPO 只用 4 generations？

你的 8GB VRAM 是硬约束。

TRL 文档明确说明，GRPO 的 generation 数、context length 等都会显著影响显存和训练成本。([Hugging Face][2])

所以：

```text
num_generations = 4
```

作为 baseline。

以后实验：

```text
4 vs 8
```

而不是一开始就 16。

---

# 27. GRPO 每一步应该保存 rollout

例如：

```json
{
  "task_id": "py_00234",

  "responses": [
    {
      "text": "...",
      "reward": 0.0
    },
    {
      "text": "...",
      "reward": 1.0
    },
    {
      "text": "...",
      "reward": 0.0
    },
    {
      "text": "...",
      "reward": 1.0
    }
  ]
}
```

这个文件非常重要。

以后你研究：

> 为什么 RL 有效？

可以直接分析它。

---

# 28. GRPO 需要重点关注这些训练指标

每 N steps：

```text
reward_mean
reward_std
completion_length_mean
entropy
KL
grad_norm
```

尤其注意：

### reward_mean

应该逐渐改善。

### reward_std

不能长期直接塌成：

```text
0
```

否则 GRPO group 内没有学习信号。

---

# 29. Phase 3 — Hard Example Mining

这是我建议你重点投入时间的实验。

建立：

```text
src/llm_lab/data/miner.py
```

执行：

```text
Current model
      ↓
10K tasks
      ↓
sample 4
      ↓
verify
      ↓
calculate Pass@4 per task
```

得到：

```text
task A = 1.00
task B = 0.75
task C = 0.25
task D = 0.00
```

---

# 30. 分类

```text
Solved:
Pass@4 >= 0.75

Borderline:
0.25 <= Pass@4 < 0.75

Hard:
0 < Pass@4 < 0.25

Impossible:
Pass@4 == 0
```

然后重点研究：

# Borderline

而不是：

# Impossible

因为 borderline 最容易产生有效学习信号。

---

# 31. Hard Mining 实验

你要明确做三个实验：

### A

Random

```text
100% random
```

### B

Hard

```text
100% hard/borderline
```

### C

Mixed

```text
70% hard
30% random
```

最终：

| Strategy | Pass@1 | OOD | GPU-hours |
| -------- | -----: | --: | --------: |
| Random   |        |     |           |
| Hard     |        |     |           |
| Mixed    |        |     |           |

这样你才能真正回答：

> **hard-example mining 是否值得？**

---

# 32. Phase 4 — Test-Time Compute

这一步**完全不训练**。

只改变 inference：

```text
K = 1
K = 2
K = 4
K = 8
K = 16
```

记录：

```text
Pass@K
tokens / problem
seconds / problem
```

最终画：

```text
Pass@K
   │
   │                  ●
   │             ●
   │         ●
   │      ●
   │   ●
   └────────────────────────
         inference compute
```

这会告诉你：

> 你的模型是否具有 test-time scaling potential。

---

# 33. Phase 5 — Tool-use

最后再加入：

```text
read_file()
write_file()
run_tests()
```

你的 environment 可以设计成：

```text
workspace/
    main.py
    tests/
```

模型：

```text
inspect
 ↓
edit
 ↓
run_tests
 ↓
observe
 ↓
fix
 ↓
run_tests
```

TRL 当前 GRPOTrainer 已经提供 environment/tool training 接口，可以直接往这个方向扩展。([Hugging Face][2])

---

# 34. Agent metrics

新增：

### Task completion rate

```text
completed tasks / total tasks
```

### Steps to success

```text
平均多少 tool calls
```

### Recovery rate

```text
first attempt failed
    ↓
second attempt succeeded
```

计算：

$$
RecoveryRate =
\frac{failure\rightarrow success}
{initial\ failures}
$$

这个指标特别有价值。

---

# 35. 4B Scaling Experiment

当 1.7B 全流程跑完：

```text
Base
 ↓
SFT
 ↓
RLVR
 ↓
Hard mining
```

再换：

```text
Qwen3-4B
```

只重复最核心实验：

```text
Base
SFT
RLVR
```

不用从头把所有 agent experiment 再做一遍。

---

# 36. 最重要的一张实验总表

建议直接存：

```text
reports/experiment_log.csv
```

字段：

```text
experiment_id
date
model
method
dataset_version
train_examples

pass_at_1
pass_at_4
pass_at_8
ood_pass_at_1

mean_reward
reward_std
eval_loss

mean_completion_tokens
mean_latency
timeout_rate

gpu_hours
tokens_trained
tokens_generated

peak_vram_gb
peak_ram_gb

learning_rate
batch_size
grad_accum
num_generations
context_length
max_completion_length
lora_rank
```

---

# 37. 你真正应该画的图

Cursor 应该自动生成：

### Figure 1

```text
Pass@1 vs Experiment
```

### Figure 2

```text
Pass@4 vs Experiment
```

### Figure 3

```text
OOD Pass@1 vs Experiment
```

### Figure 4

```text
Capability gain / GPU-hour
```

### Figure 5

```text
Pass@K vs inference compute
```

### Figure 6

```text
Reward vs training step
```

---

# 38. 最重要的一个指标：ROI

每次实验：

$$
ROI =
\frac{Pass@1_{new}-Pass@1_{base}}
{GPUHours}
$$

例如：

```text
SFT:
+4.2 points / 3 hours
= 1.4 points/hour

RLVR:
+9.5 points / 14 hours
= 0.68 points/hour

Hard Mining:
+6.1 points / 4 hours
= 1.53 points/hour
```

那么：

> **Hard Mining 反而比 RLVR 更高 ROI。**

这就是你真正要发现的东西。

---

# 39. Git strategy

我强烈建议：

```text
main
 │
 ├── exp/baseline
 ├── exp/sft-v1
 ├── exp/grpo-v1
 ├── exp/mining-v1
 └── exp/agent-v1
```

每个 experiment 的 code + config 都能复现。

**不要把 hyperparameter 改来改去然后继续用同一个 checkpoint。**

---

# 40. 给 Cursor 的总规则

把下面这段先放进项目：

## `AGENTS.md`

```markdown
# Local LLM Lab Development Rules

## Goal

Build a reproducible small-scale LLM capability research framework
for an 8GB NVIDIA GPU and 48GB system RAM.

Primary model:
Qwen/Qwen3-1.7B

Secondary model:
Qwen/Qwen3-4B

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
```

---

# 41. Cursor Prompt #1 — Environment

直接复制：

```text
Implement Phase 0 of this repository.

Goal:
Create a reproducible Windows environment diagnostic for an 8GB NVIDIA GPU
and 48GB system RAM.

Create:
- scripts/doctor.py
- src/llm_lab/hardware.py
- tests/test_smoke.py

The doctor must report:
- Python version
- PyTorch version
- CUDA availability
- CUDA version
- GPU name
- VRAM
- system RAM
- Transformers version
- TRL version
- PEFT version
- bitsandbytes version
- Docker availability

Do not install packages automatically.

Add clear error messages.

Run the tests and fix all failures.

Do not implement training yet.
```

---

# 42. Cursor Prompt #2 — Model Loading

```text
Implement local inference for Qwen/Qwen3-1.7B.

Requirements:
- 4-bit quantized loading
- device_map=auto
- configurable generation parameters
- no hardcoded absolute paths
- support model ID from YAML config
- save generation metadata
- record peak GPU memory

Create:
- src/llm_lab/inference/generate.py
- configs/baseline.yaml
- scripts/download_model.py
- tests/test_smoke.py

The implementation must work on an 8GB NVIDIA GPU.

Add a CLI:
python -m llm_lab.inference.generate --config configs/baseline.yaml

Do not implement training.
```

---

# 43. Cursor Prompt #3 — Dataset

```text
Implement the dataset layer.

Create:
- src/llm_lab/data/schema.py
- src/llm_lab/data/loader.py
- src/llm_lab/data/generator.py
- scripts/validate_dataset.py
- tests/test_dataset.py

Define a versioned JSONL schema for Python coding tasks.

Each task must contain:
- id
- version
- category
- difficulty
- prompt
- visible_tests
- hidden_tests
- optional starter_code
- metadata

Support train/dev/test split validation.

The test set must be treated as immutable.

Add checks for:
- duplicate IDs
- duplicate prompts
- train/test leakage
- malformed JSON
- missing hidden tests
- invalid difficulty values

Do not implement model training.
```

---

# 44. Cursor Prompt #4 — Verifier

这个要特别认真：

```text
Implement a secure Python coding verifier.

IMPORTANT:
Never execute model-generated Python directly on the Windows host.

Implement:
- src/llm_lab/verifier/interface.py
- src/llm_lab/verifier/docker_verifier.py
- tests/test_verifier.py

Use Docker for execution.

The verifier must:
- disable network access
- use an ephemeral container
- mount only the candidate workspace
- enforce CPU limits
- enforce memory limits
- enforce timeout
- capture stdout/stderr
- return structured results
- never mutate the source dataset
- clean up containers after execution

Return:
reward
passed
tests_passed
tests_total
timeout
syntax_error
runtime_error
stderr
duration_seconds

Implement unit tests using trusted toy examples.
```

---

# 45. Cursor Prompt #5 — Baseline Evaluation

```text
Implement the benchmark evaluator.

Create:
- src/llm_lab/evaluation/evaluator.py
- src/llm_lab/evaluation/metrics.py
- src/llm_lab/reporting/experiment.py

Requirements:
- evaluate Pass@1, Pass@4, Pass@8
- run hidden tests
- support deterministic seed
- record generation latency
- record token counts
- record peak VRAM
- record peak RAM
- save per-task results as JSONL
- save experiment summary as JSON

CLI:

python -m llm_lab.evaluation.evaluator \
  --config configs/eval.yaml

Do not modify the test dataset.

Do not implement SFT or RL.
```

---

# 46. Cursor Prompt #6 — SFT

```text
Implement QLoRA SFT for Qwen/Qwen3-1.7B using Hugging Face TRL.

Requirements:
- 4-bit NF4 quantization
- PEFT LoRA
- gradient checkpointing
- micro batch size 1
- configurable gradient accumulation
- YAML configuration
- checkpoint saving
- resume support
- experiment manifest
- deterministic seed

Start with:
max_seq_length=2048
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-5
num_train_epochs=1
LoRA r=16
LoRA alpha=32

Do not modify evaluation code.
Do not modify test.jsonl.
Log:
- train loss
- learning rate
- GPU memory
- training time

Add a smoke test that does not perform a full training run.
```

---

# 47. Cursor Prompt #7 — GRPO

```text
Implement GRPO/RLVR using the current TRL GRPOTrainer API.

Starting setup:
- base model = SFT checkpoint
- QLoRA
- num_generations=4
- max_prompt_length=1024
- max_completion_length=384
- micro batch size=1
- gradient accumulation=4

Reward:
1.0 if all hidden pytest tests pass
0.0 otherwise

Do NOT add reward shaping.

Requirements:
- record all rollouts
- record reward mean/std
- record completion length
- record training time
- record peak VRAM/RAM
- support checkpoint resume
- save YAML config with experiment
- never modify test.jsonl

Build a reward adapter around the existing verifier.

Use the current TRL API rather than inventing a custom GRPO implementation.
```

TRL 官方当前文档已经提供 `GRPOTrainer` 和自定义 reward function 接口，因此这里让 Cursor **遵循当前 API，而不是自己实现 GRPO** 很重要。([Hugging Face][2])

---

# 48. Cursor Prompt #8 — Hard Mining

```text
Implement hard-example mining.

For each task:
- sample 4 model responses
- verify all responses
- calculate per-task Pass@4

Classify:
solved: Pass@4 >= 0.75
borderline: 0.25 <= Pass@4 < 0.75
hard: 0 < Pass@4 < 0.25
impossible: Pass@4 == 0

Create:
- src/llm_lab/data/miner.py
- tests/test_miner.py

Output:
- mining_report.json
- hard.jsonl
- borderline.jsonl
- mixed.jsonl

Never include test.jsonl examples.

Add experiment IDs and dataset versions to every output.
```

---

# 49. Cursor Prompt #9 — Test-Time Search

```text
Implement test-time sampling evaluation.

Evaluate:
K = 1, 2, 4, 8, 16

For each K:
- generate K candidates
- verify each candidate
- record whether at least one succeeds
- record total tokens
- record latency

Produce:
- Pass@K table
- tokens per task
- seconds per task
- capability-vs-compute plot

Do not train the model.
```

---

# 50. Cursor Prompt #10 — Final Research Dashboard

最后：

```text
Build a research reporting pipeline.

Read all experiment manifests and metrics.

Produce:
1. experiment_log.csv
2. comparison table
3. Pass@1 vs experiment plot
4. OOD Pass@1 vs experiment plot
5. capability gain per GPU-hour
6. Pass@K vs inference compute
7. reward vs training step
8. hardware utilization summary

Never silently discard failed experiments.

A failed experiment must appear in the report with status=failed
and an error summary.
```

---

# 51. 你的实际执行顺序

我建议你严格这样做：

```text
DAY 1
│
├── Environment
├── doctor.py
└── Qwen inference
```

↓

```text
DAY 2
│
├── Dataset schema
├── 100 toy tasks
└── verifier
```

↓

```text
DAY 3
│
├── 1K baseline evaluation
└── metrics/reporting
```

↓

```text
DAY 4-5
│
└── 5K dataset + QLoRA SFT
```

↓

```text
DAY 6-7
│
└── SFT evaluation
```

↓

```text
WEEK 2
│
└── GRPO/RLVR
```

↓

```text
WEEK 3
│
├── Hard mining
├── RLVR v2
└── ablation
```

↓

```text
WEEK 4
│
├── Test-time scaling
└── tool-use prototype
```

↓

```text
之后
│
└── 4B scaling
```

---

# 52. 一个非常重要的调整：不要马上准备 5K 数据

这里我会比前面的计划更加谨慎。

实际执行顺序应该是：

### Dataset v0

**100 tasks**

目的：

```text
验证 schema
验证 verifier
验证 evaluator
```

---

### Dataset v1

**500 tasks**

目的：

```text
baseline
debug
第一次 SFT
```

---

### Dataset v2

**5,000 tasks**

目的：

```text
正式实验
```

这会节省你大量时间。

因为最容易发生的事情是：

> 你花 10 小时生成 5,000 道题，然后发现 verifier 有 bug。

😅

---

# 53. 第一周的真正 milestone

我不会把 milestone 定成：

> “模型开始训练。”

而定成：

# **M1：可靠实验闭环**

即：

```text
prompt
 ↓
Qwen3-1.7B
 ↓
candidate code
 ↓
Docker verifier
 ↓
reward
 ↓
metric
 ↓
experiment log
```

整个闭环能稳定跑 **100 tasks**。

这才是第一大里程碑。

---

# 54. 第二大 milestone

```text
Base
  ↓
SFT
```

必须证明：

```text
OOD Pass@1
```

没有明显恶化。

---

# 55. 第三大 milestone

```text
SFT
  ↓
RLVR
```

必须证明：

```text
RLVR Pass@1 > SFT Pass@1
```

否则先不要加任何 fancy trick。

---

# 56. 第四大 milestone

验证：

```text
RLVR
+
Hard Mining
```

是否优于：

```text
RLVR
```

---

# 57. 第五大 milestone

验证：

```text
1 → 2 → 4 → 8 → 16 samples
```

是否出现：

```text
Pass@K ↑
```

然后研究：

> 模型到底是“不会”，还是“知道但不会稳定找到”。

---

# 58. 最后一个很重要的设计：不要把你的模型和“官方 benchmark score”绑定

你的核心实验最好使用：

# **自己生成、自己冻结的 benchmark**

比如：

```text
test_v1
1,000 tasks
```

每道题：

```text
hidden tests
randomized inputs
```

这样你能更可靠地判断：

> 模型是不是能力真的增加。

否则你非常容易出现：

```text
benchmark ↑
real capability ?
```

---

# 59. 最终项目的科学问题

我建议把整个 repo 的 README 开头就写成：

> **Research Question**
>
> Can small, consumer-hardware-scale post-training methods improve the verifiable coding capability of a 1.7B language model, and which intervention gives the highest capability gain per unit compute?

然后下面写：

```text
Interventions

1. SFT
2. RLVR
3. Hard-example mining
4. Test-time compute
5. Tool-use
```

最后：

$$
\text{Capability Gain / GPU Hour}
$$

作为你的核心研究指标之一。

---

## 🎯 你的第一版配置，我最终建议锁死

| 项目             | 第一版                                         |
| -------------- | ------------------------------------------- |
| GPU            | 8GB NVIDIA                                  |
| RAM            | 48GB                                        |
| Model          | **Qwen3-1.7B**                              |
| Quantization   | 4-bit NF4                                   |
| Fine-tuning    | QLoRA                                       |
| SFT data       | 500 → 5K                                    |
| Context        | 2048                                        |
| RL             | GRPO                                        |
| GRPO group     | **4**                                       |
| Completion     | 384 tokens                                  |
| Reward         | **hidden pytest pass/fail**                 |
| Test           | 1,000 frozen tasks                          |
| Primary metric | **Pass@1**                                  |
| Secondary      | Pass@4 / Pass@8                             |
| Generalization | OOD Pass@1                                  |
| Efficiency     | Gain / GPU-hour                             |
| Later          | Hard mining → test-time search → agent → 4B |

这个配置的核心优点是：**你的 8GB 显存会限制实验规模，但不会阻止你研究真正重要的东西。** QLoRA/4-bit 是为了让训练进入你的硬件预算；TRL/GRPO 是为了让你能研究 RLVR；Docker verifier 是为了让 reward 真正可自动验证；统一 experiment log 则让你最终能回答“哪个方法真的有效”。([Hugging Face][2])

我建议**第一轮只让 Cursor 实现到 `Phase 0 + Phase 1`，不要一次生成整个 repo**。这样最容易发现 Windows/CUDA/quantization/verifier 的实际兼容问题。

[1]: https://huggingface.co/Qwen/Qwen3-1.7B/tree/main?utm_source=chatgpt.com "Qwen/Qwen3-1.7B at main"
[2]: https://huggingface.co/docs/trl/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer · Hugging Face"
[3]: https://huggingface.co/docs/bitsandbytes/installation?utm_source=chatgpt.com "Installation Guide · Hugging Face"
