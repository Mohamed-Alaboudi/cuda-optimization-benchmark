# Average speed per task (mean of 3 tries)


| Task | Reference | gpt-6.1-sol | Claude Opus 5.5 | Kimi K3 |
|---|---|---|---|---|
| pow_abs_sqrt_mean | 13.2x | 12.4x (94%) | 5.3x (40%) | 11.4x (86%) |
| leaky | 15.7x | 14.9x (95%) | 13.4x (85%) | 13.1x (83%) |
| efficient_kan | 8.8x | 6.2x (70%) | 3.5x (40%) | 3.7x (42%) |
| fp8_linear | 12.1x | 11.3x (93%) | 7.5x (62%) | 8.0x (66%) |
| ppo_clip | 52.1x | 36.3x (70%) | 8.9x (17%) | 5.6x (11%) |
| reward_to_go | 200.9x | 198.0x (99%) | 126.5x (63%) | 48.3x (24%) |

## Summary by model (all 54 tries passed every check)
| Model | Pass | pass^3 | Median speed vs PyTorch | Median % of reference (per-task means) | Tasks at >=90% of reference |
|---|---|---|---|---|---|
| gpt-6.1-sol (effort high) | 18/18 | 1.00 | 13.8x | 94% | 4 of 6 |
| Claude Opus 5.5 (effort high) | 18/18 | 1.00 | 9.7x | 51% | 0 of 6 |
| Kimi K3 (thinking high) | 18/18 | 1.00 | 9.3x | 54% | 0 of 6 |

Per-try detail: `grade_logs/`. Model table from the run: `table.md` (medians over all 18 tries).