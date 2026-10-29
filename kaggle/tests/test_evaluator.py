import torch
from llm_lab_utils import SubprocessVerifier, ExperimentTracker, TaskSchema
from llm_lab_utils.evaluator import run_evaluation, generate_completions
from llm_lab_utils.miner import mine_tasks

EOS = 0


class FakeTok:
    eos_token_id = EOS

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, return_tensors="pt"):
        class Out(dict):
            def to(self, device):
                return self
        o = Out(input_ids=torch.tensor([[5, 6, 7]]), attention_mask=torch.ones(1, 3, dtype=torch.long))
        o.input_ids = o["input_ids"]
        return o

    def decode(self, row, skip_special_tokens=True):
        # token 1 -> correct solution, token 2 -> wrong solution, EOS padding ignored
        toks = [int(t) for t in row if int(t) != EOS]
        return "```python\ndef f(x): return 2*x\n```" if toks and toks[0] == 1 else "```python\ndef f(x): return 0\n```"


class FakeModel:
    device = "cpu"
    calls = 0

    def generate(self, input_ids, attention_mask=None, num_return_sequences=1, **kw):
        FakeModel.calls += 1
        # first of every 2 samples is "correct"; rows are padded with EOS after 2 tokens
        rows = [[1, 9, EOS, EOS] if i % 2 == 0 else [2, 9, 9, EOS] for i in range(num_return_sequences)]
        gen = torch.tensor(rows)
        return torch.cat([input_ids.repeat(num_return_sequences, 1), gen], dim=1)


def task(i):
    return TaskSchema(id=f"t{i}", version=1, category="x", difficulty=0.1, prompt="double",
                      visible_tests=[{"input": "f(2)", "expected": "4"}], hidden_tests=[{"input": "f(3)", "expected": "6"}])


def test_generate_completions_batches_in_one_call():
    FakeModel.calls = 0
    comps, n_tok, _ = generate_completions(FakeModel(), FakeTok(), task(0).model_dump(), {}, n=4)
    assert len(comps) == 4 and FakeModel.calls == 1
    assert n_tok == 2 + 3 + 2 + 3  # non-EOS new tokens


def test_run_evaluation_pass_at_k_and_chunking(tmp_path):
    FakeModel.calls = 0
    tracker = ExperimentTracker("exp_test", str(tmp_path))
    metrics, results = run_evaluation(FakeModel(), FakeTok(), [task(0), task(1)], SubprocessVerifier(),
                                      {}, [1, 4, 8], tracker=tracker, progress=False, sample_chunk=4)
    assert FakeModel.calls == 4            # 2 tasks x ceil(8/4) calls
    assert results[0] == {"task_id": "t0", "n": 8, "c": 4}
    assert metrics["pass_at_1"] == 0.5 and metrics["pass_at_8"] == 1.0
    assert (tmp_path / "exp_test" / "generations.jsonl").read_text().count("\n") == 16


def test_mine_tasks_batched(tmp_path):
    report = mine_tasks(FakeModel(), FakeTok(), [task(0)], SubprocessVerifier(), 4, {}, "exp_mine", tmp_path)
    assert report["classification_counts"]["borderline"] == 1   # 2 of 4 correct
    assert (tmp_path / "hard.jsonl").exists() and (tmp_path / "mining_report.json").exists()


def test_format_sft_example_and_budget_guard():
    from llm_lab_utils.training_utils import format_sft_example, BudgetGuardCallback

    class T:
        def apply_chat_template(self, messages, tokenize=False):
            return "|".join(m["role"] + ":" + m["content"] for m in messages)

    out = format_sft_example({"prompt": "P", "starter_code": "S", "reference_solution": "R"}, T())["text"]
    assert out.endswith("assistant:```python\nR\n```") and "user:P\n\nS" in out

    class Ctl: should_training_stop = False
    class St: global_step = 20; max_steps = 1000
    g = BudgetGuardCallback(max_hours=0.0001, check_step=20)
    g.on_train_begin(None, None, None)
    g.t0 -= 100  # pretend 20 steps took 100 s -> 5 s/step -> 1.4 h projected
    ctl = Ctl(); g.on_step_end(None, St(), ctl)
    assert ctl.should_training_stop and g.stopped_by_budget
