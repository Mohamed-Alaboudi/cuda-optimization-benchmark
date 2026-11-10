| Kernel | Task | Status | Label | Failed case / check | x PyTorch eager | % of expert speed | Identical repeat | Allocs per call |
|---|---|---|---|---|---|---|---|---|
| claude-opus-5-5/leaky/try1 | leaky | pass |  |   | 12.254 | 93.2 | True | 0 |
| claude-opus-5-5/ppo_clip/try1 | ppo_clip | pass |  |   | 7.189 | 510.5 | True | 0 |
| expert/efficient_kan/try1 | efficient_kan | pass |  |   | 3.607 | 99.9 | True | 2 |
| expert/fp8_linear/try1 | fp8_linear | pass |  |   | 2.881 | 100.2 | True | 5 |
| expert/leaky/try1 | leaky | pass |  |   | 12.903 | 100.1 | True | 0 |
| expert/pow_abs_sqrt_mean/try1 | pow_abs_sqrt_mean | pass |  |   | 12.45 | 100.0 | True | 0 |
| expert/ppo_clip/try1 | ppo_clip | pass |  |   | 0.558 | 58.3 | True | 2 |
| expert/reward_to_go/try1 | reward_to_go | pass |  |   | 72.38 | 100.2 | True | 0 |
| gpt-6.1-sol/fp8_linear/try1 | fp8_linear | pass |  |   | 12.003 | 417.1 | True | 5 |
| gpt-6.1-sol/pow_abs_sqrt_mean/try1 | pow_abs_sqrt_mean | pass |  |   | 13.188 | 105.2 | True | 0 |
| kimi-k3/efficient_kan/try1 | efficient_kan | pass |  |   | 3.471 | 97.0 | True | 1 |
| kimi-k3/reward_to_go/try1 | reward_to_go | pass |  |   | 31.184 | 42.8 | True | 0 |
