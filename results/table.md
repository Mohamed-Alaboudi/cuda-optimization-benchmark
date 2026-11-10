# Results by model

Built by `report/make_report.py` from 54 tries in `trials` and the grade logs in `grade_logs`. Pass = correct on every input and call, integrity checks kept, sanitizer clean. Speed columns use passing tries only (timed input of each task, medians).

| Model (version, settings) | Trials per task | pass^3 | Median speedup vs PyTorch eager | Median % of reference speed |
|---|---|---|---|---|
| gpt-6.1-sol: id gpt-6.1-sol; `{"reasoning":{"effort":"high"},"max_output_tokens":32000}` | 3 | 1.00 | 13.81x | 92% |
| claude-opus-5-5: id claude-opus-5-5; `{"output_config":{"effort":"high"},"max_tokens":32000}` | 3 | 1.00 | 9.70x | 40% |
| kimi-k3: id k3; `{"reasoning_effort":"high","max_tokens":32000}` | 3 | 1.00 | 9.34x | 49% |

pass^3 = mean over tasks of C(passes, 3) / C(tries, 3).
