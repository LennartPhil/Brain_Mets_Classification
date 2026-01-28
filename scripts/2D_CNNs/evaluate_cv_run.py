# evaluate_cv_run.py
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

import helper_funcs as hf
import constants


# ------------------ utilities ------------------

def load_module(py_path: Path):
    spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
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
    # Use test_paths -> test_data for clean semantics
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
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }
    return out


def set_builder_globals(
    mod,
    *,
    channels: int,
    img_size: int,
    clinical_data: bool,
    use_layer: bool,
    num_classes: int,
):
    """
    Your builders rely on global variables inside the training script module.
    We set the critical ones here so the architecture matches training.
    """
    # input shape used in Input(...)
    mod.input_shape = (img_size, img_size, channels)

    # flags controlling inputs/concatenation
    mod.clinical_data = clinical_data
    mod.use_layer = use_layer
    mod.num_classes = num_classes

    # ensure dataset_type exists (used in builder to decide input list)
    mod.dataset_type = constants.Dataset.NORMAL

    # set sensible defaults if missing; these usually exist already in your script
    if not hasattr(mod, "activation_func"):
        mod.activation_func = constants.activation_func
    if not hasattr(mod, "dropout_rate"):
        mod.dropout_rate = constants.REGULAR_DROPOUT_RATE
    if not hasattr(mod, "l2_regularization"):
        mod.l2_regularization = constants.REGULAR_L2_REGULARIZATION
    if not hasattr(mod, "learning_rate"):
        mod.learning_rate = 1e-3  # only relevant for compile; weights define behavior anyway
    if not hasattr(mod, "contrast_DA"):
        mod.contrast_DA = False

    # make sure helper funcs reference exists if builder expects hf
    if not hasattr(mod, "hf"):
        mod.hf = hf


# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--weights_name", default="saved_weights.weights.h5", type=str)

    ap.add_argument("--model_py", required=True, type=str, help="e.g. /mnt/data/2D_CNN_resnet152.py")
    ap.add_argument("--builder_fn", required=True, type=str, help="e.g. build_resnet152_model or build_resnext_model")
    ap.add_argument("--builder_kwargs", default="{}", type=str, help='JSON string, e.g. {"architecture":"ResNeXt101"}')

    ap.add_argument("--internal_tfr_root", required=True, type=str, help="/home/lennart/work/tfrs/all_pats_single_slice_gray")
    ap.add_argument("--internal_ids", required=True, type=str, help="/home/lennart/work/tfrs/split_text_files/test_ids.txt")

    ap.add_argument("--external_tfr_root", required=True, type=str, help="/home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray")
    ap.add_argument("--external_ids", required=True, type=str, help="external_ids.txt (created by make_external_ids.py)")

    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--selected_sequences", default="t1c", type=str, help="comma-separated, e.g. t1c or t1,t1c,t2,flair,mask")
    ap.add_argument("--rgb", action="store_true")
    ap.add_argument("--use_clinical_data", action="store_true")
    ap.add_argument("--use_layer", action="store_true")

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

    channels = 3 if args.rgb else len(selected_indices)
    img_size = constants.IMG_SIZE

    # Load training-script module and get builder
    mod = load_module(Path(args.model_py))
    if not hasattr(mod, args.builder_fn):
        raise ValueError(f"{args.builder_fn} not found in {args.model_py}")
    builder = getattr(mod, args.builder_fn)

    # Set required globals so builder matches the training config
    set_builder_globals(
        mod,
        channels=channels,
        img_size=img_size,
        clinical_data=args.use_clinical_data,
        use_layer=args.use_layer,
        num_classes=args.num_classes,
    )

    builder_kwargs = json.loads(args.builder_kwargs)

    # ------------------ build datasets ------------------
    internal_ids = read_ids(Path(args.internal_ids))
    internal_tfr_paths = ids_to_tfrecord_paths(internal_ids, Path(args.internal_tfr_root))
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

    external_ids = read_ids(Path(args.external_ids))
    external_tfr_paths = ids_to_tfrecord_paths(external_ids, Path(args.external_tfr_root))
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

    # ------------------ evaluate folds ------------------
    internal_fold_metrics, external_fold_metrics = [], []
    internal_probs, external_probs = [], []

    for fold in range(args.max_folds):
        w = run_dir / f"fold_{fold}" / args.weights_name
        if not w.exists():
            print(f"Stopping at fold {fold}: missing {w}")
            break

        print(f"[Fold {fold}] building model via {args.builder_fn}({builder_kwargs})")
        model = builder(**builder_kwargs) if builder_kwargs else builder()

        print(f"[Fold {fold}] loading weights: {w}")
        model.load_weights(str(w))

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

        print(f"[Fold {fold}] INTERNAL acc={m_int['accuracy']:.4f} auc={m_int['auc']:.4f} | EXTERNAL acc={m_ext['accuracy']:.4f} auc={m_ext['auc']:.4f}")

    # ------------------ summaries ------------------
    internal_summary = summarize(internal_fold_metrics)
    external_summary = summarize(external_fold_metrics)

    (out_dir / "internal_summary_across_folds.json").write_text(json.dumps(internal_summary, indent=2))
    (out_dir / "external_summary_across_folds.json").write_text(json.dumps(external_summary, indent=2))

    # Ensemble across folds (mean probability)
    if len(internal_probs) >= 2:
        p_int_ens = np.mean(np.stack(internal_probs, axis=0), axis=0)
        p_ext_ens = np.mean(np.stack(external_probs, axis=0), axis=0)

        m_int_ens = compute_binary_metrics(y_true_internal, p_int_ens, args.threshold)
        m_ext_ens = compute_binary_metrics(y_true_external, p_ext_ens, args.threshold)

        (out_dir / "internal_ensemble.json").write_text(json.dumps(m_int_ens, indent=2))
        (out_dir / "external_ensemble.json").write_text(json.dumps(m_ext_ens, indent=2))

        np.savez_compressed(out_dir / "internal_preds_ensemble.npz", y_true=y_true_internal, y_prob=p_int_ens)
        np.savez_compressed(out_dir / "external_preds_ensemble.npz", y_true=y_true_external, y_prob=p_ext_ens)

    print("Done. Results written to:", out_dir)


if __name__ == "__main__":
    main()
