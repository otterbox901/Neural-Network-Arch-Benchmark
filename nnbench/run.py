"""Run a full architecture comparison, then build the report.

Usage: python -m nnbench.run --config configs/cifar10.yaml [--epochs 5] [--archs mlp,resnet_small]
"""

import argparse
import json
import platform
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import yaml

from . import report, train
from .data import load_splits
from .models import ARCHITECTURES

DEFAULTS = {
    "experiment_name": "arch_comparison",
    "dataset": "cifar10",
    "val_fraction": 0.1,
    "data_seed": 0,
    "train_subset": None,
    "test_subset": None,
    "seeds": [42],
    "architectures": list(ARCHITECTURES),
    "epochs": 30,
    "batch_size": 128,
    "eval_batch_size": 512,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "augment": True,
    "augment_architectures": list(ARCHITECTURES),
    "select_metric": "val_accuracy",
    "early_stopping_patience": 8,
    "lr_patience": 3,
    "test_eval_every": 1,
    "verbose": 2,
}


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow {tf.__version__} · GPUs: {[g.name for g in gpus] or 'none (CPU)'}")
    return gpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cifar10.yaml")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int)
    p.add_argument("--archs", help="comma-separated subset of architectures")
    p.add_argument("--seeds", help="comma-separated seeds, e.g. 1,2,3")
    args = p.parse_args()

    cfg = {**DEFAULTS, **(yaml.safe_load(Path(args.config).read_text()) or {})}
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.archs:
        cfg["architectures"] = args.archs.split(",")
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    unknown = set(cfg["architectures"]) - set(ARCHITECTURES)
    if unknown:
        p.error(f"unknown architectures {sorted(unknown)}; available: {sorted(ARCHITECTURES)}")

    gpus = configure_gpu()
    splits = load_splits(cfg["dataset"], cfg["val_fraction"], cfg["data_seed"],
                         cfg["train_subset"], cfg["test_subset"])
    cfg.update(n_train=len(splits.y_train), n_val=len(splits.y_val), n_test=len(splits.y_test),
               class_names=splits.class_names,
               environment={"tensorflow": tf.__version__, "python": platform.python_version(),
                            "gpus": [g.name for g in gpus]})
    print(f"{cfg['dataset']}: train {cfg['n_train']} · val {cfg['n_val']} · "
          f"test {cfg['n_test']} (independent)")

    run_dir = Path(args.out) / f"{cfg['experiment_name']}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    train.run(cfg, splits, run_dir)
    report.write_report(run_dir)


if __name__ == "__main__":
    main()
