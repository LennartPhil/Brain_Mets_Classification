# make_external_ids.py
from pathlib import Path

EXTERNAL_TFR_ROOT = Path("/home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray")
OUT_TXT = Path("/home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt")

def main():
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

    patient_ids = sorted([p.name for p in EXTERNAL_TFR_ROOT.iterdir() if p.is_dir()])
    if not patient_ids:
        raise RuntimeError(f"No patient folders found in {EXTERNAL_TFR_ROOT}")

    OUT_TXT.write_text("\n".join(patient_ids) + "\n")
    print(f"Wrote {len(patient_ids)} IDs -> {OUT_TXT}")

if __name__ == "__main__":
    main()
