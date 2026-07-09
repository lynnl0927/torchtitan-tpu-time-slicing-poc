FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Install HTTP dependencies
RUN pip install --no-cache-dir fastapi uvicorn requests

# 3. Install JAX and Tokamax (strict sequence)
RUN pip install --no-cache-dir "jax[tpu]==0.9.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --no-cache-dir git+https://github.com/openxla/tokamax.git@8cba6a6a1e52e9efbb7ff8facb66f18f0bfcbe4c

# 4. Install PyTorch CPU and LibTPU runtime
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir libtpu==0.0.40 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 5. Re-verify HTTP server dependencies
RUN pip install --no-cache-dir fastapi uvicorn requests grpcio

# 6. Install the internal torch_tpu wheel directly from artifact registry
ARG GCP_TOKEN
RUN pip install --no-cache-dir --pre --no-deps \
    --index-url "https://oauth2accesstoken:${GCP_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" \
    torch_tpu

# 7. Copy the torchtitan codebase
COPY . /app/torchtitan
WORKDIR /app/torchtitan

# 7.5 Download the custom UDS-enabled libtpu from GCS
RUN curl -sL -H "Authorization: Bearer ${GCP_TOKEN}" -o /app/_libtpu.so https://storage.googleapis.com/linglinll-gke-dev-libtpu/_libtpu.so

# 8. Install TorchTitan project dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir portpicker absl-py numpy gcsfs frozendict triton fairscale tamm
RUN pip install --no-cache-dir 'typeguard==2.13.3'

# 9. Set the environment variable for Python
ENV PYTHONPATH=/app/torchtitan:$PYTHONPATH
