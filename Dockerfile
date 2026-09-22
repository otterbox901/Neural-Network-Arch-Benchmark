# GPU image by default; build with --build-arg BASE_IMAGE=docker.io/tensorflow/tensorflow:2.21.0 for CPU.
ARG BASE_IMAGE=docker.io/tensorflow/tensorflow:2.21.0-gpu
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The -gpu base image ships CUDA 12.3 and no cuDNN, but TF 2.21 needs CUDA >= 12.5 + cuDNN 9.
# Installing TF's own [and-cuda] extra pulls matching NVIDIA wheels, which TF finds on its own.
RUN case "$BASE_IMAGE" in *-gpu) pip install --no-cache-dir "tensorflow[and-cuda]==$(python -c 'import tensorflow as tf; print(tf.__version__)')" ;; esac

COPY nnbench/ nnbench/
COPY configs/ configs/
COPY tests/ tests/

# Datasets are cached in /root/.keras (mount ./data there); results go to /app/results.
VOLUME ["/root/.keras", "/app/results"]
ENTRYPOINT ["python", "-m", "nnbench.run"]
CMD ["--config", "configs/cifar10.yaml"]
