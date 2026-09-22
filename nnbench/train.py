"""Train each architecture and track independent test-set metrics across epochs."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import metrics
from .data import make_dataset
from .models import build


class TestSetMonitor(keras.callbacks.Callback):
    """Evaluates the held-out test set every N epochs, for *reporting only*.

    Nothing here feeds back into training: checkpoint selection, early stopping
    and LR scheduling all use the validation split.
    """

    def __init__(self, x_test, y_test, batch_size, every=1):
        super().__init__()
        self.x_test, self.y_test = x_test, y_test
        self.batch_size, self.every = batch_size, every
        self.rows = []

    def on_train_begin(self, logs=None):
        self._t0 = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.perf_counter() - self._epoch_t0  # excludes test eval below
        row = {
            "epoch": epoch + 1,
            "epoch_time_s": epoch_time,
            "learning_rate": float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate)),
            **{k: float(v) for k, v in (logs or {}).items() if k != "learning_rate"},
        }
        if (epoch + 1) % self.every == 0:
            probs = self.model.predict(self.x_test, batch_size=self.batch_size, verbose=0)
            row.update({f"test_{k}": v for k, v in
                        metrics.classification_metrics(self.y_test, probs).items()})
        row["elapsed_s"] = time.perf_counter() - self._t0
        self.rows.append(row)


def _inference_speed(model, x, batch_size, repeats=50):
    model.predict(x[:batch_size], verbose=0)  # warm-up / graph trace
    t0 = time.perf_counter()
    model.predict(x, batch_size=batch_size, verbose=0)
    throughput = len(x) / (time.perf_counter() - t0)

    single = tf.convert_to_tensor(x[:1])
    model(single, training=False)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(single, training=False)
        times.append(time.perf_counter() - t0)
    return throughput, float(np.median(times) * 1000.0)


def train_one(arch, seed, splits, cfg, run_dir):
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)

    model = build(arch, splits.input_shape, splits.num_classes)
    model.compile(
        optimizer=keras.optimizers.AdamW(cfg["learning_rate"], weight_decay=cfg["weight_decay"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    tag = f"{arch}_seed{seed}"
    ckpt = run_dir / "models" / f"{tag}.keras"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    monitor = TestSetMonitor(splits.x_test, splits.y_test, cfg["eval_batch_size"],
                             cfg["test_eval_every"])
    select = cfg["select_metric"]
    mode = "min" if "loss" in select else "max"
    callbacks = [
        monitor,  # first, so its epoch timer excludes the other callbacks' work
        keras.callbacks.ModelCheckpoint(ckpt, monitor=select, mode=mode, save_best_only=True),
        keras.callbacks.EarlyStopping(monitor=select, mode=mode,
                                      patience=cfg["early_stopping_patience"]),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=cfg["lr_patience"], min_lr=1e-6),
    ]

    augment = cfg["augment"] and arch in cfg["augment_architectures"]
    train_ds = make_dataset(splits.x_train, splits.y_train, cfg["batch_size"],
                            shuffle=True, augment=augment, seed=seed)
    val_ds = make_dataset(splits.x_val, splits.y_val, cfg["eval_batch_size"])

    t0 = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=cfg["epochs"],
              callbacks=callbacks, verbose=cfg["verbose"])
    wall_time = time.perf_counter() - t0

    history = pd.DataFrame(monitor.rows)
    history.insert(0, "seed", seed)
    history.insert(0, "architecture", arch)
    history.to_csv(run_dir / "history" / f"{tag}.csv", index=False)

    # Final evaluation uses the checkpoint chosen on validation data only.
    best = keras.models.load_model(ckpt)
    probs = best.predict(splits.x_test, batch_size=cfg["eval_batch_size"], verbose=0)
    np.save(run_dir / "predictions" / f"{tag}.npy", probs.astype(np.float32))

    correct = probs.argmax(axis=1) == splits.y_test
    lo, hi = metrics.bootstrap_accuracy_ci(correct, seed=seed)
    throughput, latency_ms = _inference_speed(best, splits.x_test, cfg["eval_batch_size"])
    best_epoch = int(history[select].idxmin() if mode == "min" else history[select].idxmax()) + 1

    result = {
        "architecture": arch,
        "seed": seed,
        "params": int(best.count_params()),
        "augmented": augment,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "train_time_s": wall_time,
        "train_time_excl_test_eval_s": float(history["epoch_time_s"].sum()),
        "val_accuracy": float(history.loc[best_epoch - 1, "val_accuracy"]),
        **{f"test_{k}": v for k, v in metrics.classification_metrics(splits.y_test, probs).items()},
        "test_accuracy_ci_low": lo,
        "test_accuracy_ci_high": hi,
        "throughput_img_per_s": throughput,
        "latency_ms_single": latency_ms,
        "per_class_f1": metrics.per_class_f1(splits.y_test, probs),
        "confusion_matrix": metrics.confusion(splits.y_test, probs),
    }
    with open(run_dir / "final" / f"{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def run(cfg, splits, run_dir: Path):
    for sub in ("history", "predictions", "final", "models"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "predictions" / "y_test.npy", splits.y_test)

    results = []
    for arch in cfg["architectures"]:
        for seed in cfg["seeds"]:
            print(f"\n=== {arch} (seed {seed}) ===", flush=True)
            r = train_one(arch, seed, splits, cfg, run_dir)
            print(f"--> test acc {r['test_accuracy']:.4f} "
                  f"[{r['test_accuracy_ci_low']:.4f}, {r['test_accuracy_ci_high']:.4f}]  "
                  f"macro-F1 {r['test_macro_f1']:.4f}  ECE {r['test_ece']:.4f}  "
                  f"params {r['params']:,}  time {r['train_time_s']:.0f}s", flush=True)
            results.append(r)
    return results
