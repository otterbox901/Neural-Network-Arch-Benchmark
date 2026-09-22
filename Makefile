# Works with docker or podman (auto-detected; override with ENGINE=docker).
ENGINE ?= $(shell command -v docker >/dev/null 2>&1 && echo docker || echo podman)
TF_VERSION ?= 2.21.0
CONFIG ?= configs/cifar10.yaml
ARGS ?=

ifeq ($(ENGINE),podman)
GPU_FLAGS ?= --device nvidia.com/gpu=all --security-opt=label=disable
else
GPU_FLAGS ?= --gpus all
endif
MOUNTS = -v $(CURDIR)/configs:/app/configs:ro,Z -v $(CURDIR)/data:/root/.keras:Z -v $(CURDIR)/results:/app/results:Z

.PHONY: dirs build build-cpu run run-cpu smoke test report shell

build:
	$(ENGINE) build --build-arg BASE_IMAGE=docker.io/tensorflow/tensorflow:$(TF_VERSION)-gpu -t nn-arch-benchmark:gpu .

build-cpu:
	$(ENGINE) build --build-arg BASE_IMAGE=docker.io/tensorflow/tensorflow:$(TF_VERSION) -t nn-arch-benchmark:cpu .

dirs:
	@mkdir -p data results

run: dirs            ## full benchmark on GPU
	$(ENGINE) run --rm $(GPU_FLAGS) $(MOUNTS) nn-arch-benchmark:gpu --config $(CONFIG) $(ARGS)

run-cpu: dirs        ## full benchmark on CPU (slow for CIFAR-10)
	$(ENGINE) run --rm $(MOUNTS) nn-arch-benchmark:cpu --config $(CONFIG) $(ARGS)

smoke: dirs          ## few-minute end-to-end check on CPU
	$(ENGINE) run --rm $(MOUNTS) nn-arch-benchmark:cpu --config configs/smoke.yaml

test:
	$(ENGINE) run --rm --entrypoint python nn-arch-benchmark:cpu -m pytest -q tests

report:              ## rebuild report: make report RUN=results/<run_dir>
	$(ENGINE) run --rm $(MOUNTS) --entrypoint python nn-arch-benchmark:cpu -m nnbench.report $(RUN)

shell: dirs
	$(ENGINE) run --rm -it $(MOUNTS) --entrypoint bash nn-arch-benchmark:cpu
