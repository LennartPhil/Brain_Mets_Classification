import argparse
import importlib.util
import json
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


def build_test_ds(
    tfr_paths: list[str],
    *,
    selected_indices: list[int],
    batch_size: int,
    num_classes: int,
    rgb: bool,
    use_clinical_data: bool,
    use_layer: bool,
):
    dummy = tfr_paths[:1] if tfr_paths else []
    _, _, test_data = hf.read_data(
        train_paths=dummy,
        val_paths=dummy,
        test_paths=tfr_paths,
        selected_indices=selected_indices,
        batch_size=batch_size,
        num_classes=num_classes,
        rgb=rgb,
        use_clinical_data=use_clinical_data,
        use_layer=use_layer,
        dataset_type=constants.Dataset.NORMAL,
    )
    return test_data


def collect_y_true(ds: tf.data.Dataset) -> np.ndarray:
    ys = []
    for x, y in ds:
        ys.append(y.numpy().reshape(-1))
    return np.concatenate(ys).astype(int)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--weights_name", default="saved_weights.weights.h5", type=str)
    ap.add_argument("--prefer_first_weight", action="store_true", help="If set: pick first matching weight file instead of newest")

    ap.add_argument("--model_py", required=True, type=str)
    ap.add_argument("--builder_fn", required=True, type=str)
    ap.add_argument("--builder_kwargs", default="{}", type=str)

    ap.add_argument("--internal_tfr_root", required=True, type=str)
    ap.add_argument("--internal_ids", required=True, type=str)

    ap.add_argument("--external_tfr_root", required=True, type=str)
    ap.add_argument("--external_ids", required=True, type=str)

    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--selected_sequences", default="t1c", type=str, help="comma-separated: t1,t1c,t2,flair,(mask)")
    ap.add_argument("--rgb", action="store_true")

    ap.add_argument("--use_clinical_data", action="store_true")
    ap.add_argument("--use_layer", action="store_true")

    ap.add_argument("--dropout_rate", required=True, type=float)
    ap.add_argument("--l2_regularization", required=True, type=float)

    # for transfer models (InceptionV3, ResNet50V2) that use internal resizing
    ap.add_argument("--image_size", default=None, type=int, help="e.g. 299 for InceptionV3, 224 for ResNet50V2")

    ap.add_argument("--learning_rate", default=None, type=float)
    ap.add_argument("--activation_func", default=None, type=str)
    ap.add_argument("--contrast_DA", action="store_true")
    ap.add_argument("--no_contrast_DA", action="store_true")

    # NEW: weight loading behavior flags
    ap.add_argument("--allow_mismatch", action="store_true",
                    help="If set: retry weight loading with by_name=True, skip_mismatch=True when strict loading fails.")
    ap.add_argument("--stop_on_weight_error", action="store_true",
                    help="If set: stop evaluation immediately if any fold weight loading fails.")

    ap.add_argument("--batch_size", default=50, type=int)
    ap.add_argument("--num_classes", default=2, type=int)
    ap.add_argument("--threshold", default=0.5, type=float)
    ap.add_argument("--max_folds", default=10, type=int)

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

    # Quiet import prevents misleading "Using sequences: ..." prints from the training script
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

    # datasets
    internal_ids = read_ids(Path(args.internal_ids))
    internal_tfr_paths = ids_to_tfrecord_paths(internal_ids, Path(args.internal_tfr_root))
    print(f"[EVAL] Internal: {len(internal_ids)} patients -> {len(internal_tfr_paths)} TFRecord files")
    internal_ds = build_test_ds(
        internal_tfr_paths,
        selected_indices=selected_indices,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        rgb=args.rgb,
        use_clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
    )
    y_true_internal = collect_y_true(internal_ds)
    print(f"[EVAL] Internal: collected y_true of length {len(y_true_internal)}")

    external_ids = read_ids(Path(args.external_ids))
    external_tfr_paths = ids_to_tfrecord_paths(external_ids, Path(args.external_tfr_root))
    print(f"[EVAL] External: {len(external_ids)} patients -> {len(external_tfr_paths)} TFRecord files")
    external_ds = build_test_ds(
        external_tfr_paths,
        selected_indices=selected_indices,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        rgb=args.rgb,
        use_clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
    )
    y_true_external = collect_y_true(external_ds)
    print(f"[EVAL] External: collected y_true of length {len(y_true_external)}")

    # fold evaluation
    internal_fold_metrics, external_fold_metrics = [], []
    internal_probs, external_probs = [], []

    for fold in range(args.max_folds):
        w = find_weight_for_fold(run_dir, fold, args.weights_name, prefer_first=args.prefer_first_weight)
        if w is None:
            print(f"[EVAL] Stopping at fold {fold}: no weights found under {run_dir} matching fold_{fold}/{args.weights_name}")
            break

        print(f"[EVAL] Fold {fold}: using weights: {w.resolve()}")
        model = builder(**builder_kwargs) if builder_kwargs else builder()

        # sanity-check the model's image input shape before loading weights
        check_model_input_shape(model, expected_hw=img_size, expected_channels=expected_channels)

        # safe weight loading with notifier for failures/mismatches
        ok, mismatch_mode, changed, total = safe_load_weights(model, w, allow_mismatch=args.allow_mismatch)
        if not ok:
            msg = f"[EVAL] Fold {fold}: weight loading failed. Skipping fold."
            if args.stop_on_weight_error:
                raise RuntimeError(msg)
            print(msg)
            continue

        p_int = model.predict(internal_ds, verbose=1).reshape(-1)
        p_ext = model.predict(external_ds, verbose=1).reshape(-1)

        m_int = compute_binary_metrics(y_true_internal, p_int, args.threshold)
        m_ext = compute_binary_metrics(y_true_external, p_ext, args.threshold)

        internal_fold_metrics.append(m_int)
        external_fold_metrics.append(m_ext)
        internal_probs.append(p_int)
        external_probs.append(p_ext)

        (out_dir / f"internal_metrics_fold_{fold}.json").write_text(json.dumps(m_int, indent=2))
        (out_dir / f"external_metrics_fold_{fold}.json").write_text(json.dumps(m_ext, indent=2))
        np.savez_compressed(out_dir / f"internal_preds_fold_{fold}.npz", y_true=y_true_internal, y_prob=p_int)
        np.savez_compressed(out_dir / f"external_preds_fold_{fold}.npz", y_true=y_true_external, y_prob=p_ext)

        print(
            f"[EVAL] Fold {fold} | INTERNAL acc={m_int['accuracy']:.4f} auc={m_int['auc']:.4f} | "
            f"EXTERNAL acc={m_ext['accuracy']:.4f} auc={m_ext['auc']:.4f}"
        )

    (out_dir / "internal_summary_across_folds.json").write_text(json.dumps(summarize(internal_fold_metrics), indent=2))
    (out_dir / "external_summary_across_folds.json").write_text(json.dumps(summarize(external_fold_metrics), indent=2))

    if len(internal_probs) >= 2:
        p_int_ens = np.mean(np.stack(internal_probs, axis=0), axis=0)
        p_ext_ens = np.mean(np.stack(external_probs, axis=0), axis=0)

        (out_dir / "internal_ensemble.json").write_text(
            json.dumps(compute_binary_metrics(y_true_internal, p_int_ens, args.threshold), indent=2)
        )
        (out_dir / "external_ensemble.json").write_text(
            json.dumps(compute_binary_metrics(y_true_external, p_ext_ens, args.threshold), indent=2)
        )

        np.savez_compressed(out_dir / "internal_preds_ensemble.npz", y_true=y_true_internal, y_prob=p_int_ens)
        np.savez_compressed(out_dir / "external_preds_ensemble.npz", y_true=y_true_external, y_prob=p_ext_ens)

    print("Done. Results written to:", out_dir)


if __name__ == "__main__":
    main()
