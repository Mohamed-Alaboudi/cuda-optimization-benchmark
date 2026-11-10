| Kernel | Task | Status | Label | Failed case / check | x PyTorch eager | % of expert speed | Identical repeat | Allocs per call |
|---|---|---|---|---|---|---|---|---|
| expert/efficient_kan | efficient_kan | pass |  |   | 8.915 | 100.0 | True | 4 |
| expert/fp8_linear | fp8_linear | pass |  |   | 12.657 | 99.9 | True | 5 |
| expert/leaky | leaky | pass |  |   | 31.085 | 100.0 | True | 0 |
| expert/pow_abs_sqrt_mean | pow_abs_sqrt_mean | pass |  |   | 13.14 | 98.8 | True | 0 |
| expert/ppo_clip | ppo_clip | pass |  |   | 85.676 | 99.4 | True | 0 |
| expert/reward_to_go | reward_to_go | pass |  |   | 262.854 | 97.8 | True | 0 |
| planted/efficient_kan/cached_result | efficient_kan | fail | stale state | B1_I256_O512 call2:out |  |  | False | 4 |
| planted/efficient_kan/oob_write | efficient_kan | fail | out-of-bounds write | B1_I256_O512 call1:guards |  |  | True | 4 |
| planted/efficient_kan/race | efficient_kan | fail | race | B64_I100_O300 sanitizer:racecheck |  |  | True | 4 |
| planted/efficient_kan/uninit_read | efficient_kan | fail | uninitialized read | B64_I100_O300 sanitizer:initcheck |  |  | True | 6 |
| planted/efficient_kan/wrong_value | efficient_kan | fail | wrong value | B1_I256_O512 call1:out |  |  | True | 4 |
| planted/fp8_linear/cached_result | fp8_linear | fail | stale state | M1_K1024_N4096 call2:output |  |  | False | 5 |
| planted/fp8_linear/oob_write | fp8_linear | fail | out-of-bounds write | M1_K1024_N4096 call1:guards |  |  | True | 5 |
| planted/fp8_linear/race | fp8_linear | fail | race | M16_K64_N128 sanitizer:racecheck |  |  | True | 5 |
| planted/fp8_linear/uninit_read | fp8_linear | fail | uninitialized read | M16_K64_N128 sanitizer:initcheck |  |  | True | 7 |
| planted/fp8_linear/wrong_value | fp8_linear | fail | wrong value | M1_K1024_N4096 call1:output |  |  | True | 5 |
| planted/leaky/cached_result | leaky | fail | stale state | T1_B1_H1021 call2:spikes |  |  | False | 0 |
| planted/leaky/oob_write | leaky | fail | out-of-bounds write | T1_B1_H1021 call1:guards |  |  | True | 0 |
| planted/leaky/race | leaky | fail | race | T33_B65_H129_mem0 sanitizer:racecheck |  |  | True | 0 |
| planted/leaky/uninit_read | leaky | fail | uninitialized read | T33_B65_H129_mem0 sanitizer:initcheck |  |  | True | 2 |
| planted/leaky/wrong_value | leaky | fail | wrong value | T1_B1_H1021 call1:spikes |  |  | True | 0 |
| planted/pow_abs_sqrt_mean/cached_result | pow_abs_sqrt_mean | fail | stale state | b4_h4_w4 call2:output |  |  | False | 0 |
| planted/pow_abs_sqrt_mean/oob_write | pow_abs_sqrt_mean | fail | out-of-bounds write | b4_h4_w4 call1:guards |  |  | True | 0 |
| planted/pow_abs_sqrt_mean/race | pow_abs_sqrt_mean | fail | race | b3_h129_w257 sanitizer:racecheck |  |  | True | 0 |
| planted/pow_abs_sqrt_mean/uninit_read | pow_abs_sqrt_mean | fail | uninitialized read | b3_h129_w257 sanitizer:initcheck |  |  | True | 2 |
| planted/pow_abs_sqrt_mean/wrong_value | pow_abs_sqrt_mean | fail | wrong value | b4_h4_w4 call1:output |  |  | True | 0 |
| planted/ppo_clip/cached_result | ppo_clip | fail | stale state | B1_T1_H1 call2:output |  |  | False | 0 |
| planted/ppo_clip/oob_write | ppo_clip | fail | out-of-bounds write | B1_T1_H1 call1:guards |  |  | True | 0 |
| planted/ppo_clip/race | ppo_clip | fail | race | B1000_T333_H2 sanitizer:racecheck |  |  | True | 0 |
| planted/ppo_clip/uninit_read | ppo_clip | fail | uninitialized read | B1000_T333_H2 sanitizer:initcheck |  |  | True | 2 |
| planted/ppo_clip/wrong_value | ppo_clip | fail | wrong value | B1_T1_H1 call1:output |  |  | True | 0 |
| planted/reward_to_go/cached_result | reward_to_go | fail | stale state | B3_T1 call2:output |  |  | False | 0 |
| planted/reward_to_go/oob_write | reward_to_go | fail | out-of-bounds write | B3_T1 call1:guards |  |  | True | 0 |
| planted/reward_to_go/race | reward_to_go | fail | race | B7_T999 sanitizer:racecheck |  |  | True | 0 |
| planted/reward_to_go/uninit_read | reward_to_go | fail | uninitialized read | B7_T999 sanitizer:initcheck |  |  | True | 2 |
| planted/reward_to_go/wrong_value | reward_to_go | fail | wrong value | B3_T1 call1:output |  |  | True | 0 |
