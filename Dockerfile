# forge-observatory — serving image (promoted apps only; dev stays in INFO_698_experiments).
# One recipe, CPU or CUDA via BASE_IMAGE:
#   CPU  : docker build -t forge:cpu .
#   CUDA : docker build --build-arg BASE_IMAGE=tensorflow/tensorflow:2.16.1-gpu -t forge:gpu .
#          (the TF GPU image already ships Python + CUDA + cuDNN — cleaner than nvidia/cuda,
#           which has no Python. uv then pins the rest from pyproject.)
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# librosa needs libsndfile/ffmpeg to decode audio; git/curl for tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git curl ca-certificates && rm -rf /var/lib/apt/lists/*

# uv = fast, locked installs. Keep the venv OUTSIDE /workspace so a live
# `.:/workspace` bind mount (compose) does NOT shadow the installed deps.
RUN pip install --no-cache-dir uv
ENV UV_PROJECT_ENVIRONMENT=/opt/forge-venv
ENV PATH=/opt/forge-venv/bin:$PATH

WORKDIR /workspace
# deps first -> this layer survives source edits, so rebuilds are fast
COPY pyproject.toml uv.lock* ./
RUN uv sync --extra eda --extra app --frozen || uv sync --extra eda --extra app

COPY . .
ENV PYTHONPATH=/workspace
ENV FORGE_HOST=0.0.0.0
EXPOSE 5000
# default service = the genre dashboard; `lab` service overrides to a shell
CMD ["python", "apps/genre/app/server.py"]
