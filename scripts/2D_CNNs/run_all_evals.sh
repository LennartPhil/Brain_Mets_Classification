#!/usr/bin/env bash
set -u  # error on undefined vars

LOG_ROOT="/home/lennart/work/eval_logs"
mkdir -p "$LOG_ROOT"

run_eval () {
  local name="$1"; shift
  echo "============================================================"
  echo "[START] $name  $(date)"
  echo "============================================================"

  # run + log, do NOT stop whole script if one fails:
  ( "$@" ) 2>&1 | tee "$LOG_ROOT/${name}.log"
  local code=${PIPESTATUS[0]}

  if [ $code -ne 0 ]; then
    echo "[FAIL]  $name (exit code $code)  $(date)"
  else
    echo "[OK]    $name  $(date)"
  fi

  echo
  return 0
}


# run_eval "4seq_scratch_resnext50" \
#     python evaluate_cv_run.py \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/4seq/scratch/resnext50/resnext50_00_2cls_slice_no_clin_no_layer_gray_seq[[]t1-t1c-t2-flair[]]_normal_DA_kfold_run_2025_09_27_19_59_45 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt50"}' \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_4seq_scratch_resnext50 \
#         --selected_sequences t1,t1c,t2,flair \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0005 \
#         --no_contrast_DA

# run_eval "4seq_scratch_resnext101" \
#   python evaluate_cv_run.py \
#     --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/4seq/scratch/resnext101/resnext101_00_2cls_slice_no_clin_no_layer_gray_seq[[]t1-t1c-t2-flair[]]_normal_DA_kfold_run_2025_09_28_14_32_12 \
#     --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
#     --builder_fn build_resnext_model \
#     --builder_kwargs '{"architecture":"ResNeXt101"}' \
#     --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#     --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#     --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#     --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#     --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_4seq_scratch_resnext101 \
#     --selected_sequences t1,t1c,t2,flair \
#     --dropout_rate 0.45 \
#     --l2_regularization 0.0005 \
#     --no_contrast_DA

# run_eval "4seq_scratch_resnet152" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/4seq/scratch/resnet152/resnet152_00_2cls_slice_no_clin_no_layer_gray_seq[[]t1-t1c-t2-flair[]]_normal_DA_kfold_run_2025_09_26_12_47_30 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
#         --builder_fn build_resnet152_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_4seq_scratch_resnet152 \
#         --selected_sequences t1,t1c,t2,flair \
#         --dropout_rate 0.4 \
#         --l2_regularization 0.0005 \
#         --no_contrast_DA


run_eval "maxinfo_scratch_resnext50" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/maxinfo/scratch/resnext50/resnext50_00_2cls_slice_clin_layer_gray_seq[[]t1-t1c-t2-flair-mask[]]_normal_DA_kfold_run_2026_01_07_15_25_06 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
        --builder_fn build_resnext_model \
        --builder_kwargs '{"architecture":"ResNeXt50"}' \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_maxinfo_scratch_resnext50 \
        --selected_sequences t1,t1c,t2,flair,mask \
        --dropout_rate 0.4 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \
        --use_layer \

run_eval "maxinfo_scratch_resnext101" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/maxinfo/scratch/resnext101/resnext101_00_2cls_slice_clin_layer_gray_seq[[]t1-t1c-t2-flair-mask[]]_normal_DA_kfold_run_2026_01_12_14_08_53 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
        --builder_fn build_resnext_model \
        --builder_kwargs '{"architecture":"ResNeXt101"}' \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_maxinfo_scratch_resnext101 \
        --selected_sequences t1,t1c,t2,flair,mask \
        --dropout_rate 0.4 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \
        --use_layer \

run_eval "maxinfo_scratch_resnet152" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/maxinfo/scratch/resnet152/resnet152_00_2cls_slice_clin_layer_gray_seq[[]t1-t1c-t2-flair-mask[]]_normal_DA_kfold_run_2026_01_03_03_08_42 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
        --builder_fn build_resnet152_model \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_maxinfo_scratch_resnet152 \
        --selected_sequences t1,t1c,t2,flair,mask \
        --dropout_rate 0.45 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \
        --use_layer \


run_eval "t1c_clin_scratch_resnext50" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c_clin/scratch/resnext50/resnext50_00_2cls_slice_clin_no_layer_gray_seq[[]t1c[]]_normal_DA_kfold_run_2025_10_03_14_28_34 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
        --builder_fn build_resnext_model \
        --builder_kwargs '{"architecture":"ResNeXt50"}' \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_clin_scratch_resnext50 \
        --selected_sequences t1c \
        --dropout_rate 0.45 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \

run_eval "t1c_clin_scratch_resnext101" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c_clin/scratch/resnext101/resnext101_00_2cls_slice_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2026_02_07_11_05_01 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
        --builder_fn build_resnext_model \
        --builder_kwargs '{"architecture":"ResNeXt101"}' \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_clin_scratch_resnext101 \
        --selected_sequences t1c \
        --dropout_rate 0.45 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \

run_eval "t1c_clin_scratch_resnet152" \
    python evaluate_cv_run.py \
        --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c_clin/scratch/resnet152/resnet152_00_2cls_slice_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_10_09_10_26_28 \
        --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
        --builder_fn build_resnet152_model \
        --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
        --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
        --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
        --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
        --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_clin_scratch_resnet152 \
        --selected_sequences t1c \
        --dropout_rate 0.45 \
        --l2_regularization 0.0001 \
        --no_contrast_DA \
        --use_clinical_data \


# run_eval "t1c_scratch_resnext50" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/resnext50/resnext50_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_07_31_12_48_20 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt50"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_resnext50 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_scratch_resnext101" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/resnext101/resnext101_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_08_01_14_38_29 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt101"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_resnext101 \
#         --selected_sequences t1c \
#         --dropout_rate 0.4 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_scratch_resnet152" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/resnet152/resnet152_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_08_17_19_31_38 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
#         --builder_fn build_resnet152_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_resnet152 \
#         --selected_sequences t1c \
#         --dropout_rate 0.4 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_scratch_base_conv" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/base_conv/conv_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_07_30_14_14_03 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_conv.py \
#         --builder_fn build_conv_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_base_conv \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_scratch_resnet34" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/resnet34/resnet34_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_07_30_19_52_45 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet34.py \
#         --builder_fn build_resnet34_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_resnet34 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_scratch_inceptionv3" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/inceptionv3/transfer_inceptionv3_00_2cls_slice_no_clin_no_layer_rgb_seq[t1c]_normal_DA_kfold_run_2025_09_01_16_03_27 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_transfer_inceptionv3.py \
#         --builder_fn build_transfer_inceptionv3_model \
#         --builder_kwargs '{"trainable":"False"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_inceptionv3 \
#         --selected_sequences t1c \
#         --dropout_rate 0.4 \
#         --l2_regularization 0.001 \
#         --no_contrast_DA \
#         --rgb \

# run_eval "t1c_scratch_resnet50v2" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/scratch/resnet50v2/transfer_resnet50v2_00_2cls_slice_no_clin_no_layer_rgb_seq[t1c]_normal_DA_kfold_run_2025_09_10_07_02_26 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_transfer_resnet50v2.py \
#         --builder_fn build_transfer_resnet50_model \
#         --builder_kwargs '{"trainable":"False"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_scratch_resnet50v2 \
#         --selected_sequences t1c \
#         --dropout_rate 0.4 \
#         --l2_regularization 0.0003 \
#         --no_contrast_DA \
#         --rgb \


# run_eval "t1c_rough_resnext50" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/rough/resnext50/resnext50_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_combined \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt50"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_rough_resnext50 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.001 \
#         --no_contrast_DA \

# run_eval "t1c_rough_resnext101" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/rough/resnext101/resnext101_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_combined \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt101"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_rough_resnext101 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.001 \
#         --no_contrast_DA \

# run_eval "t1c_rough_resnet152" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/rough/resnet152/resnet152_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_12_04_12_52_31 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
#         --builder_fn build_resnet152_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_rough_resnet152 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.001 \
#         --no_contrast_DA \

# run_eval "t1c_fine_resnext50" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/fine/resnext50/resnext50_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_10_25_18_15_06 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext50.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt50"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_fine_resnext50 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_fine_resnext101" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/fine/resnext101/resnext101_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_10_26_22_44_43 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnext101.py \
#         --builder_fn build_resnext_model \
#         --builder_kwargs '{"architecture":"ResNeXt101"}' \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_fine_resnext101 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \

# run_eval "t1c_fine_resnet152" \
#     python evaluate_cv_run.py \
#         --run_dir /home/lennart/work/runs/lung_vs_nonlung_2cls/t1c/fine/resnet152/resnet152_00_2cls_slice_no_clin_no_layer_gray_seq[t1c]_normal_DA_kfold_run_2025_10_24_09_13_57 \
#         --model_py /home/lennart/work/Brain_Mets_Classification/scripts/2D_CNNs/2D_CNN_resnet152.py \
#         --builder_fn build_resnet152_model \
#         --internal_tfr_root /home/lennart/work/tfrs/all_pats_single_slice_gray \
#         --internal_ids /home/lennart/work/tfrs/split_text_files/test_ids.txt \
#         --external_tfr_root /home/lennart/work/tfrs/yale_slices_tfrecords/all_pats_single_slice_gray \
#         --external_ids /home/lennart/work/tfrs/yale_slices_tfrecords/split_text_files/external_ids.txt \
#         --out_dir /home/lennart/work/eval_internal_external/lung_vs_nolung_2cls_t1c_fine_resnet152 \
#         --selected_sequences t1c \
#         --dropout_rate 0.45 \
#         --l2_regularization 0.0001 \
#         --no_contrast_DA \