# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install grpcurl globally
RUN curl -sL https://github.com/fullstorydev/grpcurl/releases/download/v1.9.1/grpcurl_1.9.1_linux_x86_64.tar.gz | tar -xz -C /usr/local/bin grpcurl

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

# 6. Install the internal torch_tpu wheel from the build context (pre-download
# it into ./wheels with your own credentials — see deploy/README.md; a token
# build-arg would leak the credential into the image's build history).
# The bind mount keeps the 122 MB wheel out of the image layers.
RUN --mount=type=bind,source=wheels,target=/tmp/wheels \
    pip install --no-cache-dir --no-deps /tmp/wheels/torch_tpu-*.whl

# 7. Copy the torchtitan codebase
COPY . /app/torchtitan
WORKDIR /app/torchtitan

# 8. Install TorchTitan project dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir portpicker absl-py numpy gcsfs frozendict triton fairscale tamm
RUN pip install --no-cache-dir 'typeguard==2.13.3'

# 9. Set the environment variable for Python
ENV PYTHONPATH=/app/torchtitan:$PYTHONPATH

# The custom checkpointing libtpu (_libtpuv2.so) is intentionally NOT baked
# into this image: rl-disagg.yaml stages it from GCS via the stage-libtpu
# initContainer and sets TPU_LIBRARY_PATH=/shared/_libtpuv2.so. Without that
# override, the pip-installed `libtpu` package is used (no checkpoint/restore).
