# Try isolation
Each try ran in its own empty folder, as its own unix user. The prompts name the task folder `/app`; the runner maps `/app` to that try's folder (`runner/agent_loop.py`, `map_app`). Tries share no files, caches or history. The network was open on the pod but the models did not use it. Source: `runner/agent_loop.py`; per-try folders under `results/trials/`.
