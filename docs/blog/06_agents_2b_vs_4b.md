# 6. From code generator to agent: it works at 4B and not at 2B

*Series part 6. Experiments: notebooks 10 (2B, 1.11 GPU h) and 11 (4B, 0.71 GPU h).*

## The agent
A ReAct-style loop with four tools: `read_file`, `write_file`, `run_tests` and `submit`, at most 5 steps. The model writes a solution, runs the tests, reads the failure and tries again.

**A leak we fixed before trusting any number.** The first version let `run_tests` execute the hidden tests, so the agent could "see" the grader. Now `run_tests` only runs the visible example, and the final grade uses visible plus hidden tests, never shown to the agent. A unit test (`test_agent_feedback_does_not_leak_hidden_tests`) pins this behaviour.

## 2B: the loop does not help
60 held-out tasks, same model as single-shot generation:

| Model | Single-shot | Agent | Difference (95% CI) |
| :--- | ---: | ---: | :--- |
| base | 45.0% | 50.0% | +5.0 [-5.6, +15.0] |
| GRPO from base | 60.4% | 58.3% | -2.1 [-11.3, +6.7] |

No interval excludes zero. The overheads explain why: 6-12% of steps were unparseable, 11-17% truncated, and the recovery rate (failing, then passing after feedback) was about a third. Compare with post 5: picking among 16 samples using the visible test gave about +18 points; five agent steps gave about nothing.

## 4B: the loop helps a lot
On 39 of the 60 tasks (the run reached its time budget) the 4B base model reached **74.4%** as an agent against **51.3%** single-shot on the same tasks: **+23.1 pt, CI [+13.1, +34.0]**. Even though 15% of steps were unparseable and 27% truncated (4B writes long reasoning), feedback let it recover.

## Reading it
Tool use is a capability that appears with scale: the 4B model can use failure messages, the 2B model mostly cannot. The caveats are real: 39 tasks, the first 39 of the list rather than a random subset, and one run. A bigger model also makes single-shot a weaker baseline here because of the 384-token cap (post 9).

Next: [is 2B plus RL better than a 4B model?](09_train_smarter_or_bigger.md)
