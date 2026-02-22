#!/usr/bin/env python3
"""
evaluate_cv_run.py

Evaluates a cross-validation run (multiple folds) on internal + external TFRecord datasets,
and writes per-fold + ensemble predictions/metrics.

IMPORTANT CHANGE vs your previous version:
- Collects a patient_id for every TFRecord example (lesion)
- Aggregates lesion-level predictions to patient-level (default: mean prob across lesions)
- Computes patient-level AUC + patient-level bootstrap confidence intervals (resampling patients)
- Saves BOTH lesion-level and patient-level arrays into the .npz files

This is the correct approach when you have multiple lesions per patient.
"""

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf

import helper_funcs as hf
import constants


# ------------------ utilities ------------------

def load_module(py_path: Path, quiet: bool = True):
    """
    Loads a python file as a module.

    IMPORTANT: Many of your training scripts execute code at import-time
    (including prints, path creation, etc.). To avoid misleading output
    (e.g. showing the script's hardcoded selected_sequences), we silence
    stdout/stderr by default.
    """
    import contextlib
    import io

    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    if quiet:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            spec.loader.exec_module(mod)  # type: ignore
    else:
        spec.loader.exec_module(mod)  # type: ignore

    return mod


def read_ids(txt_path: Path) -> list[str]:
    return [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]


def ids_to_tfrecord_paths(patient_ids: list[str], tfr_root: Path) -> list[str]:
    patient_dirs = [str(tfr_root / pid) for pid in patient_ids]
    return hf.get_tfr_paths_for_patients(patient_dirs)


def _normalize_path_sep(fn: tf.Tensor) -> tf.Tensor:
    # make Windows paths safe too
    return tf.strings.regex_replace(fn, r"\\", "/")


def _patient_id_from_filename(fn: tf.Tensor) -> tf.Tensor:
    """
    Assumes TFRecords live in:
      .../<patient_id>/<file>.tfrecord
    and returns the parent folder as patient_id.
    """
    fn = _normalize_path_sep(fn)
    parts = tf.strings.split(fn, "/")
    return parts[-2]


def build_test_ds_with_patient_ids(
    tfr_paths: list[str],
    *,
    selected_indices: list[int],
    batch_size: int,
    num_classes: int,
    rgb: bool,
    use_clinical_data: bool,
    use_layer: bool,
    dataset_type,
):
    """
    Returns a dataset of (inputs, y, patient_id) batches.

    Uses helper_funcs.parse_record() for consistent parsing with training.
    """
    if not tfr_paths:
        raise ValueError("No TFRecord paths provided.")

    files = tf.data.Dataset.from_tensor_slices(tfr_paths)

    # dataset yields (filename, serialized_example)
    ds = files.interleave(
        lambda fn: tf.data.TFRecordDataset([fn], compression_type="GZIP")
                    .map(lambda rec: (fn, rec), num_parallel_calls=tf.data.AUTOTUNE),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    # Ensure selected_indices is a Tensor so parse_record's rgb branch (tf.shape) is safe.
    sel_idx_t = tf.constant(selected_indices, dtype=tf.int32)

    parse_partial = partial(
        hf.parse_record,
        selected_indices=sel_idx_t,
        dataset_type=dataset_type,
        use_clinical_data=use_clinical_data,
        use_layer=use_layer,
        labeled=True,
        num_classes=num_classes,
        rgb=rgb,
    )

    def _map_fn(fn, rec):
        inputs, y = parse_partial(rec)
        pid = _patient_id_from_filename(fn)
        return inputs, y, pid

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(1)
    return ds


def ds_for_prediction(ds_with_pid: tf.data.Dataset) -> tf.data.Dataset:
    """Convert (x, y, pid) -> x only, for model.predict()."""
    return ds_with_pid.map(lambda x, y, pid: x)


def collect_y_true_and_patient_ids(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect lesion-level y_true and patient_id in the exact order produced by the dataset.
    y_true is returned as int array shape (N,).
    patient_ids is returned as object array shape (N,) with strings.
    """
    ys, pids = [], []
    for x, y, pid in ds:
        ys.append(y.numpy().reshape(-1))
        pids.append(pid.numpy().reshape(-1))  # bytes

    y_true = np.concatenate(ys)

    # In your pipeline, binary labels are float32 with shape (N,1) -> convert to int
    y_true = y_true.astype(float)
    y_true = np.rint(y_true).astype(int).reshape(-1)

    pid_bytes = np.concatenate(pids).astype("S")
    patient_ids = np.array([p.decode("utf-8") for p in pid_bytes], dtype=object)

    return y_true, patient_ids


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> dict:
    y_true = y_true.reshape(-1).astype(int)
    y_prob = y_prob.reshape(-1)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = float((tp + tn) / max(tp + tn + fp + fn, 1))
    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))

    auc_m = tf.keras.metrics.AUC(curve="ROC")
    auc_m.update_state(y_true, y_prob)
    auc = float(auc_m.result().numpy())

    return {
        "n": int(len(y_true)),
        "accuracy": acc,
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": float(thr),
    }


def summarize(metrics_list: list[dict]) -> dict:
    keys = ["accuracy", "auc", "sensitivity", "specificity"]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0}
    return out


# ------------------ patient-level aggregation + bootstrap ------------------

def aggregate_to_patient_level(
    y_true_lesion: np.ndarray,
    y_prob_lesion: np.ndarray,
    patient_ids_lesion: np.ndarray,
    how: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate lesion-level arrays into patient-level arrays.

    Returns:
      y_true_patient: (N_patients,)
      y_prob_patient: (N_patients,)
      patient_ids_unique: (N_patients,) object array of patient IDs

    Assumes y_true is constant within each patient.
    """
    y_true_lesion = np.asarray(y_true_lesion).reshape(-1).astype(int)
    y_prob_lesion = np.asarray(y_prob_lesion).reshape(-1).astype(float)
    patient_ids_lesion = np.asarray(patient_ids_lesion).reshape(-1)

    uniq = np.unique(patient_ids_lesion)
    yt_p, yp_p = [], []

    for pid in uniq:
        idx = np.where(patient_ids_lesion == pid)[0]
        yt = y_true_lesion[idx]
        yp = y_prob_lesion[idx]

        if yt.size == 0:
            continue

        # sanity: should be one label per patient
        if not np.all(yt == yt[0]):
            raise ValueError(f"Inconsistent labels within patient {pid}: {np.unique(yt)}")

        yt_p.append(int(yt[0]))

        if how == "mean":
            yp_p.append(float(np.mean(yp)))
        elif how == "max":
            yp_p.append(float(np.max(yp)))
        elif how == "median":
            yp_p.append(float(np.median(yp)))
        else:
            raise ValueError("how must be one of: mean, max, median")

    return np.array(yt_p, dtype=int), np.array(yp_p, dtype=float), uniq


def roc_auc_tf(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    m = tf.keras.metrics.AUC(curve="ROC")
    m.update_state(y_true.reshape(-1).astype(int), y_prob.reshape(-1).astype(float))
    return float(m.result().numpy())


def bootstrap_auc_ci_patient_level(
    y_true_patient: np.ndarray,
    y_prob_patient: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Bootstrap CI for AUC by resampling patients (NOT lesions).
    """
    rng = np.random.default_rng(seed)
    y_true_patient = np.asarray(y_true_patient).reshape(-1).astype(int)
    y_prob_patient = np.asarray(y_prob_patient).reshape(-1).astype(float)

    n = len(y_true_patient)
    if n == 0:
        return {
            "auc": float("nan"),
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "n_patients": 0,
            "n_boot_used": 0,
            "n_boot_failed_one_class": n_boot,
        }

    aucs = []
    n_fail = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # sample patients with replacement
        yt = y_true_patient[idx]
        yp = y_prob_patient[idx]

        # AUC undefined if only one class present
        if len(np.unique(yt)) < 2:
            n_fail += 1
            continue

        aucs.append(roc_auc_tf(yt, yp))

    aucs = np.asarray(aucs, dtype=float)
    auc_point = roc_auc_tf(y_true_patient, y_prob_patient)

    if len(aucs) == 0:
        return {
            "auc": auc_point,
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "n_patients": int(n),
            "n_boot_used": 0,
            "n_boot_failed_one_class": int(n_fail),
        }

    return {
        "auc": auc_point,
        "auc_ci_low": float(np.quantile(aucs, alpha / 2)),
        "auc_ci_high": float(np.quantile(aucs, 1 - alpha / 2)),
        "n_patients": int(n),
        "n_boot_used": int(len(aucs)),
        "n_boot_failed_one_class": int(n_fail),
    }


# ------------------ weights + model helpers ------------------

def find_weight_for_fold(run_dir: Path, fold: int, weights_name: str, prefer_first: bool = False) -> Path | None:
    # Find all fold_k weight files anywhere under run_dir
    pattern = f"**/fold_{fold}/{weights_name}"
    matches = list(run_dir.glob(pattern))
    if not matches:
        return None
    if prefer_first:
        return sorted(matches)[0]
    # otherwise pick newest by mtime (best for resumed training / crashes)
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def safe_load_weights(model: tf.keras.Model, weight_path: Path, allow_mismatch: bool):
    """
    Try strict loading first. If it fails and allow_mismatch=True, retry with by_name+skip_mismatch.
    Also prints an approximate 'how many tensors updated' indicator for mismatch mode.
    Returns (loaded_ok: bool, used_mismatch_mode: bool, updated_tensors: int, total_tensors: int).
    """
    try:
        model.load_weights(str(weight_path))
        print(f"[EVAL] Weights loaded STRICT OK: {weight_path.name}")
        return True, False, -1, -1
    except Exception as e:
        print("\n" + "!" * 90)
        print(f"[EVAL][WEIGHT LOAD ERROR] Strict load failed for: {weight_path}")
        print(f"[EVAL][WEIGHT LOAD ERROR] Exception: {type(e).__name__}: {e}")
        print("!" * 90 + "\n")

        if not allow_mismatch:
            return False, False, -1, -1

    # Retry mismatch mode
    try:
        before = [w.numpy().copy() for w in model.weights]
        model.load_weights(str(weight_path), by_name=True, skip_mismatch=True)
        after = [w.numpy() for w in model.weights]

        changed = 0
        for b, a in zip(before, after):
            if b.shape != a.shape:
                changed += 1
                continue
            if not np.array_equal(b, a):
                changed += 1

        total = len(after)

        print("\n" + "-" * 90)
        print(f"[EVAL][WEIGHT LOAD WARNING] Loaded with by_name=True, skip_mismatch=True: {weight_path.name}")
        print(f"[EVAL][WEIGHT LOAD WARNING] Approx. updated tensors: {changed}/{total}")
        print("[EVAL][WEIGHT LOAD WARNING] This suggests at least some mismatch vs the architecture used in training.")
        print("-" * 90 + "\n")

        return True, True, changed, total

    except Exception as e2:
        print("\n" + "!" * 90)
        print(f"[EVAL][WEIGHT LOAD ERROR] Mismatch-mode load ALSO failed for: {weight_path}")
        print(f"[EVAL][WEIGHT LOAD ERROR] Exception: {type(e2).__name__}: {e2}")
        print("!" * 90 + "\n")
        return False, True, -1, -1


def check_model_input_shape(model: tf.keras.Model, expected_hw: int, expected_channels: int):
    """
    Warn if the model's image input shape doesn't match the expected TFRecord image shape.
    This catches the common mistake: evaluating a model trained on different channels/sequences.
    """
    # Find an input tensor that looks like an image (rank 4)
    img_shape = None
    if isinstance(model.input_shape, list):
        for shp in model.input_shape:
            if isinstance(shp, tuple) and len(shp) == 4:
                img_shape = shp
                break
    else:
        if isinstance(model.input_shape, tuple) and len(model.input_shape) == 4:
            img_shape = model.input_shape

    if img_shape is None:
        print("[EVAL][SHAPE WARNING] Could not identify a rank-4 image input in model.input_shape.")
        print(f"[EVAL][SHAPE WARNING] model.input_shape = {model.input_shape}")
        return

    # shape is typically (None, H, W, C)
    _, h, w, c = img_shape
    ok = (h == expected_hw and w == expected_hw and c == expected_channels)

    if not ok:
        print("\n" + "!" * 90)
        print("[EVAL][SHAPE WARNING] Model image input shape mismatch!")
        print(f"[EVAL][SHAPE WARNING] Expected image input: (None, {expected_hw}, {expected_hw}, {expected_channels})")
        print(f"[EVAL][SHAPE WARNING] Detected image input: {img_shape}")
        print("[EVAL][SHAPE WARNING] This usually means selected_sequences/rgb flag does not match the run.")
        print("!" * 90 + "\n")
    else:
        print(f"[EVAL] Model image input shape OK: {img_shape}")


def set_builder_globals(
    mod,
    *,
    seqs: list[str],
    selected_indices: list[int],
    img_size: int,
    rgb: bool,
    clinical_data: bool,
    use_layer: bool,
    num_classes: int,
    dropout_rate: float,
    l2_regularization: float,
    image_size_override: int | None,
    learning_rate: float | None,
    activation_func: str | None,
    contrast_DA: bool | None,
):
    """
    Many of the builders depend on module-level globals.
    We overwrite those globals and ALSO recompute derived variables
    (selected_indices/num_selected_channels/input_shape) the same way
    the training scripts do.

    This ensures the CLI --selected_sequences actually changes the
    model graph at evaluation time.
    """

    # ---- Core flags used by the builders ----
    mod.dataset_type = constants.Dataset.NORMAL

    # Your scripts typically use rgb_images, not rgb
    mod.rgb_images = bool(rgb)

    mod.clinical_data = bool(clinical_data)
    mod.use_layer = bool(use_layer)
    mod.num_classes = int(num_classes)

    mod.dropout_rate = float(dropout_rate)
    mod.l2_regularization = float(l2_regularization)

    # Inception/ResNet50V2 transfer scripts resize internally using image_size
    if image_size_override is not None:
        mod.image_size = int(image_size_override)

    if contrast_DA is not None:
        mod.contrast_DA = bool(contrast_DA)
    elif not hasattr(mod, "contrast_DA"):
        mod.contrast_DA = False

    # compile-only (but referenced)
    if learning_rate is not None:
        mod.learning_rate = float(learning_rate)
    elif not hasattr(mod, "learning_rate"):
        mod.learning_rate = 1e-3

    if activation_func is not None:
        mod.activation_func = activation_func
        # some builders reference constants.activation_func explicitly
        constants.activation_func = activation_func
    else:
        if not hasattr(mod, "activation_func"):
            mod.activation_func = constants.activation_func

    # Ensure hf exists if script expects it as module var
    if not hasattr(mod, "hf"):
        mod.hf = hf

    # ---- Sequence globals + derived globals (matches your training script logic) ----
    mod.selected_sequences = seqs
    mod.selected_indices = selected_indices
    mod.num_selected_channels = len(selected_indices)

    if mod.num_selected_channels == 0:
        raise ValueError("[EVAL] selected_sequences cannot be empty.")

    # mirror your script's shape derivation
    if mod.num_selected_channels == 1 and mod.rgb_images is True:
        mod.input_shape = (img_size, img_size, 3)
    else:
        if mod.rgb_images is True:
            raise ValueError("[EVAL] rgb_images cannot be used when multiple sequences are selected.")
        mod.input_shape = (img_size, img_size, mod.num_selected_channels)

    # ---- Confirmation prints (so you can verify what the builder sees) ----
    print("\n" + "=" * 90)
    print("[EVAL] Module globals AFTER override (THIS is what the builder will use):")
    print(f"[EVAL] selected_sequences    = {mod.selected_sequences}")
    print(f"[EVAL] selected_indices      = {mod.selected_indices}")
    print(f"[EVAL] num_selected_channels = {mod.num_selected_channels}")
    print(f"[EVAL] rgb_images            = {mod.rgb_images}")
    print(f"[EVAL] input_shape           = {mod.input_shape}")
    print(f"[EVAL] clinical_data         = {mod.clinical_data}")
    print(f"[EVAL] use_layer             = {mod.use_layer}")
    print(f"[EVAL] num_classes           = {mod.num_classes}")
    print(f"[EVAL] dropout_rate          = {mod.dropout_rate}")
    print(f"[EVAL] l2_regularization     = {mod.l2_regularization}")
    if hasattr(mod, "image_size"):
        print(f"[EVAL] image_size            = {getattr(mod, 'image_size')}")
    print("=" * 90 + "\n")


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--weights_name", default="saved_weights.weights.h5", type=str)
    ap.add_argument("--prefer_first_weight", action="store_true",
                    help="If set: pick first matching weight file instead of newest")

    ap.add_argument("--model_py", required=True, type=str)
    ap.add_argument("--builder_fn", required=True, type=str)
    ap.add_argument("--builder_kwargs", default="{}", type=str)

    ap.add_argument("--internal_tfr_root", required=True, type=str)
    ap.add_argument("--internal_ids", required=True, type=str)

    ap.add_argument("--external_tfr_root", required=True, type=str)
    ap.add_argument("--external_ids", required=True, type=str)

    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--selected_sequences", default="t1c", type=str,
                    help="comma-separated: t1,t1c,t2,flair,(mask)")
    ap.add_argument("--rgb", action="store_true")

    ap.add_argument("--use_clinical_data", action="store_true")
    ap.add_argument("--use_layer", action="store_true")

    ap.add_argument("--dropout_rate", required=True, type=float)
    ap.add_argument("--l2_regularization", required=True, type=float)

    # for transfer models (InceptionV3, ResNet50V2) that use internal resizing
    ap.add_argument("--image_size", default=None, type=int,
                    help="e.g. 299 for InceptionV3, 224 for ResNet50V2")

    ap.add_argument("--learning_rate", default=None, type=float)
    ap.add_argument("--activation_func", default=None, type=str)
    ap.add_argument("--contrast_DA", action="store_true")
    ap.add_argument("--no_contrast_DA", action="store_true")

    # weight loading behavior flags
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="Retry weight loading with by_name=True, skip_mismatch=True if strict loading fails.")
    ap.add_argument("--stop_on_weight_error", action="store_true",
                    help="Stop evaluation immediately if any fold weight loading fails.")

    ap.add_argument("--batch_size", default=50, type=int)
    ap.add_argument("--num_classes", default=2, type=int)
    ap.add_argument("--threshold", default=0.5, type=float)
    ap.add_argument("--max_folds", default=10, type=int)

    # NEW: patient aggregation + bootstrap settings
    ap.add_argument("--patient_agg", default="mean", type=str, choices=["mean", "max", "median"],
                    help="How to aggregate lesion probabilities to patient level.")
    ap.add_argument("--n_boot", default=2000, type=int,
                    help="Number of bootstrap resamples for patient-level AUC CI.")
    ap.add_argument("--boot_seed", default=42, type=int,
                    help="Random seed for bootstrap.")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = [s.strip() for s in args.selected_sequences.split(",") if s.strip()]
    selected_indices = [constants.SEQUENCE_TO_INDEX[s] for s in seqs]

    img_size = constants.IMG_SIZE  # TFRecord image size in your pipeline (240)

    # Expected model input channels (image input only)
    expected_channels = 3 if args.rgb else len(selected_indices)

    contrast_flag = None
    if args.contrast_DA and args.no_contrast_DA:
        raise ValueError("Use only one of --contrast_DA or --no_contrast_DA")
    if args.contrast_DA:
        contrast_flag = True
    if args.no_contrast_DA:
        contrast_flag = False

    # Quiet import prevents misleading prints from the training script
    mod = load_module(Path(args.model_py), quiet=True)
    if not hasattr(mod, args.builder_fn):
        raise ValueError(f"{args.builder_fn} not found in {args.model_py}")
    builder = getattr(mod, args.builder_fn)

    set_builder_globals(
        mod,
        seqs=seqs,
        selected_indices=selected_indices,
        img_size=img_size,
        rgb=args.rgb,
        clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        l2_regularization=args.l2_regularization,
        image_size_override=args.image_size,
        learning_rate=args.learning_rate,
        activation_func=args.activation_func,
        contrast_DA=contrast_flag,
    )

    builder_kwargs = json.loads(args.builder_kwargs)

    # ------------------ datasets (collect y_true + patient_id once) ------------------

    internal_ids = read_ids(Path(args.internal_ids))
    internal_tfr_paths = ids_to_tfrecord_paths(internal_ids, Path(args.internal_tfr_root))
    print(f"[EVAL] Internal: {len(internal_ids)} patients -> {len(internal_tfr_paths)} TFRecord files")

    internal_ds_pid = build_test_ds_with_patient_ids(
        internal_tfr_paths,
        selected_indices=selected_indices,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        rgb=args.rgb,
        use_clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
        dataset_type=constants.Dataset.NORMAL,
    )
    y_true_internal_lesion, pid_internal_lesion = collect_y_true_and_patient_ids(internal_ds_pid)
    internal_ds = ds_for_prediction(internal_ds_pid)

    print(f"[EVAL] Internal: collected lesion-level y_true length {len(y_true_internal_lesion)} "
          f"({len(np.unique(pid_internal_lesion))} unique patients)")

    external_ids = read_ids(Path(args.external_ids))
    external_tfr_paths = ids_to_tfrecord_paths(external_ids, Path(args.external_tfr_root))
    print(f"[EVAL] External: {len(external_ids)} patients -> {len(external_tfr_paths)} TFRecord files")

    external_ds_pid = build_test_ds_with_patient_ids(
        external_tfr_paths,
        selected_indices=selected_indices,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        rgb=args.rgb,
        use_clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
        dataset_type=constants.Dataset.NORMAL,
    )
    y_true_external_lesion, pid_external_lesion = collect_y_true_and_patient_ids(external_ds_pid)
    external_ds = ds_for_prediction(external_ds_pid)

    print(f"[EVAL] External: collected lesion-level y_true length {len(y_true_external_lesion)} "
          f"({len(np.unique(pid_external_lesion))} unique patients)")

    # ------------------ fold evaluation ------------------

    internal_fold_metrics_patient, external_fold_metrics_patient = [], []
    internal_probs_lesion, external_probs_lesion = [], []

    for fold in range(args.max_folds):
        w = find_weight_for_fold(run_dir, fold, args.weights_name, prefer_first=args.prefer_first_weight)
        if w is None:
            print(f"[EVAL] Stopping at fold {fold}: no weights found under {run_dir} matching fold_{fold}/{args.weights_name}")
            break

        print(f"[EVAL] Fold {fold}: using weights: {w.resolve()}")
        model = builder(**builder_kwargs) if builder_kwargs else builder()

        # sanity-check the model's image input shape before loading weights
        check_model_input_shape(model, expected_hw=img_size, expected_channels=expected_channels)

        ok, mismatch_mode, changed, total = safe_load_weights(model, w, allow_mismatch=args.allow_mismatch)
        if not ok:
            msg = f"[EVAL] Fold {fold}: weight loading failed. Skipping fold."
            if args.stop_on_weight_error:
                raise RuntimeError(msg)
            print(msg)
            continue

        # lesion-level probabilities (aligned with pid_*_lesion)
        p_int_lesion = model.predict(internal_ds, verbose=1).reshape(-1)
        p_ext_lesion = model.predict(external_ds, verbose=1).reshape(-1)

        # aggregate to patient-level
        yt_int_p, yp_int_p, pid_int_p = aggregate_to_patient_level(
            y_true_internal_lesion, p_int_lesion, pid_internal_lesion, how=args.patient_agg
        )
        yt_ext_p, yp_ext_p, pid_ext_p = aggregate_to_patient_level(
            y_true_external_lesion, p_ext_lesion, pid_external_lesion, how=args.patient_agg
        )

        # patient-level metrics + patient-level AUC CI
        m_int_p = compute_binary_metrics(yt_int_p, yp_int_p, args.threshold)
        m_ext_p = compute_binary_metrics(yt_ext_p, yp_ext_p, args.threshold)

        ci_int = bootstrap_auc_ci_patient_level(
            yt_int_p, yp_int_p, n_boot=args.n_boot, seed=args.boot_seed
        )
        ci_ext = bootstrap_auc_ci_patient_level(
            yt_ext_p, yp_ext_p, n_boot=args.n_boot, seed=args.boot_seed
        )

        m_int_p.update({
            "auc_ci_low": ci_int["auc_ci_low"],
            "auc_ci_high": ci_int["auc_ci_high"],
            "n_patients": ci_int["n_patients"],
            "patient_agg": args.patient_agg,
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_boot_used": ci_int["n_boot_used"],
            "n_boot_failed_one_class": ci_int["n_boot_failed_one_class"],
        })
        m_ext_p.update({
            "auc_ci_low": ci_ext["auc_ci_low"],
            "auc_ci_high": ci_ext["auc_ci_high"],
            "n_patients": ci_ext["n_patients"],
            "patient_agg": args.patient_agg,
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_boot_used": ci_ext["n_boot_used"],
            "n_boot_failed_one_class": ci_ext["n_boot_failed_one_class"],
        })

        internal_fold_metrics_patient.append(m_int_p)
        external_fold_metrics_patient.append(m_ext_p)
        internal_probs_lesion.append(p_int_lesion)
        external_probs_lesion.append(p_ext_lesion)

        # Write per-fold metrics (patient-level)
        (out_dir / f"internal_metrics_fold_{fold}.json").write_text(json.dumps(m_int_p, indent=2))
        (out_dir / f"external_metrics_fold_{fold}.json").write_text(json.dumps(m_ext_p, indent=2))

        # Write per-fold predictions with BOTH lesion and patient arrays
        np.savez_compressed(
            out_dir / f"internal_preds_fold_{fold}.npz",
            y_true_lesion=y_true_internal_lesion,
            y_prob_lesion=p_int_lesion,
            patient_id_lesion=pid_internal_lesion,
            y_true_patient=yt_int_p,
            y_prob_patient=yp_int_p,
            patient_id_patient=pid_int_p,
        )
        np.savez_compressed(
            out_dir / f"external_preds_fold_{fold}.npz",
            y_true_lesion=y_true_external_lesion,
            y_prob_lesion=p_ext_lesion,
            patient_id_lesion=pid_external_lesion,
            y_true_patient=yt_ext_p,
            y_prob_patient=yp_ext_p,
            patient_id_patient=pid_ext_p,
        )

        print(
            f"[EVAL] Fold {fold} | INTERNAL patient-level acc={m_int_p['accuracy']:.4f} "
            f"auc={m_int_p['auc']:.4f} [{m_int_p['auc_ci_low']:.4f}, {m_int_p['auc_ci_high']:.4f}] | "
            f"EXTERNAL patient-level acc={m_ext_p['accuracy']:.4f} "
            f"auc={m_ext_p['auc']:.4f} [{m_ext_p['auc_ci_low']:.4f}, {m_ext_p['auc_ci_high']:.4f}]"
        )

    # Summaries across folds (patient-level)
    (out_dir / "internal_summary_across_folds.json").write_text(
        json.dumps(summarize(internal_fold_metrics_patient), indent=2)
    )
    (out_dir / "external_summary_across_folds.json").write_text(
        json.dumps(summarize(external_fold_metrics_patient), indent=2)
    )

    # ------------------ ensemble across folds (lesion-level ensemble -> patient aggregation) ------------------

    if len(internal_probs_lesion) >= 2:
        p_int_ens_lesion = np.mean(np.stack(internal_probs_lesion, axis=0), axis=0)
        p_ext_ens_lesion = np.mean(np.stack(external_probs_lesion, axis=0), axis=0)

        # patient aggregation
        yt_int_p, yp_int_p, pid_int_p = aggregate_to_patient_level(
            y_true_internal_lesion, p_int_ens_lesion, pid_internal_lesion, how=args.patient_agg
        )
        yt_ext_p, yp_ext_p, pid_ext_p = aggregate_to_patient_level(
            y_true_external_lesion, p_ext_ens_lesion, pid_external_lesion, how=args.patient_agg
        )

        m_int_p = compute_binary_metrics(yt_int_p, yp_int_p, args.threshold)
        m_ext_p = compute_binary_metrics(yt_ext_p, yp_ext_p, args.threshold)

        ci_int = bootstrap_auc_ci_patient_level(
            yt_int_p, yp_int_p, n_boot=args.n_boot, seed=args.boot_seed
        )
        ci_ext = bootstrap_auc_ci_patient_level(
            yt_ext_p, yp_ext_p, n_boot=args.n_boot, seed=args.boot_seed
        )

        m_int_p.update({
            "auc_ci_low": ci_int["auc_ci_low"],
            "auc_ci_high": ci_int["auc_ci_high"],
            "n_patients": ci_int["n_patients"],
            "patient_agg": args.patient_agg,
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_boot_used": ci_int["n_boot_used"],
            "n_boot_failed_one_class": ci_int["n_boot_failed_one_class"],
        })
        m_ext_p.update({
            "auc_ci_low": ci_ext["auc_ci_low"],
            "auc_ci_high": ci_ext["auc_ci_high"],
            "n_patients": ci_ext["n_patients"],
            "patient_agg": args.patient_agg,
            "n_boot": args.n_boot,
            "boot_seed": args.boot_seed,
            "n_boot_used": ci_ext["n_boot_used"],
            "n_boot_failed_one_class": ci_ext["n_boot_failed_one_class"],
        })

        (out_dir / "internal_ensemble.json").write_text(json.dumps(m_int_p, indent=2))
        (out_dir / "external_ensemble.json").write_text(json.dumps(m_ext_p, indent=2))

        np.savez_compressed(
            out_dir / "internal_preds_ensemble.npz",
            y_true_lesion=y_true_internal_lesion,
            y_prob_lesion=p_int_ens_lesion,
            patient_id_lesion=pid_internal_lesion,
            y_true_patient=yt_int_p,
            y_prob_patient=yp_int_p,
            patient_id_patient=pid_int_p,
        )
        np.savez_compressed(
            out_dir / "external_preds_ensemble.npz",
            y_true_lesion=y_true_external_lesion,
            y_prob_lesion=p_ext_ens_lesion,
            patient_id_lesion=pid_external_lesion,
            y_true_patient=yt_ext_p,
            y_prob_patient=yp_ext_p,
            patient_id_patient=pid_ext_p,
        )

    print("Done. Results written to:", out_dir)


if __name__ == "__main__":
    main()