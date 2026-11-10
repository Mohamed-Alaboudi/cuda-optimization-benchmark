# What the model sees in /app: reference.py, check.py, public_cases.json and the prompt.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir torch==2.9.1 numpy==2.2.6 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
WORKDIR /app
COPY reference.py check.py public_cases.json /app/
