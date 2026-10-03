import argparse
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


TASK_CONFIGS = {
    "2cls": {
        "classes": ["CHC", "WBC"],
    },
    "3cls": {
        "classes": ["CD14", "CD3", "CD66b"],
    },
}


def setup_logging(output_dir, log_file):
    if log_file is None:
        log_file = "postprocess_track.log"
    if not os.path.isabs(log_file):
        log_file = os.path.join(output_dir, log_file)
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_file


def resolve_output_path(output_dir, path):
    return path if os.path.isabs(path) else os.path.join(output_dir, path)


def log_metrics(metrics, per_class_df, confusion_df):
    logging.info("Track-level metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logging.info("%s: %.4f", key, value)
        else:
            logging.info("%s: %s", key, value)

    logging.info("\nTrack-level per-class metrics:\n%s", per_class_df.to_string(index=False))
    logging.info("\nTrack-level confusion matrix:\n%s", confusion_df.to_string())


def compute_metrics(y_true, y_pred, class_names):
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, true in zip(y_pred, y_true):
        pred = int(pred)
        true = int(true)
        if not 0 <= pred < num_classes or not 0 <= true < num_classes:
            raise ValueError(
                f"Class index out of range: pred={pred}, true={true}, "
                f"num_classes={num_classes}"
            )
        matrix[pred, true] += 1

    total = int(np.sum(matrix))
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    per_class_rows = []

    for class_index, class_name in enumerate(class_names):
        true_positive = matrix[class_index, class_index]
        false_positive = np.sum(matrix[class_index, :]) - true_positive
        false_negative = np.sum(matrix[:, class_index]) - true_positive
        true_negative = np.sum(matrix) - true_positive - false_positive - false_negative
        support = np.sum(matrix[:, class_index])

        precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive else 0.0
        )
        class_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(class_f1)
        supports.append(support)

        per_class_rows.append({
            "class_index": class_index,
            "class_name": class_name,
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "specificity": round(float(specificity), 3),
            "f1": round(float(class_f1), 3),
            "support": int(support),
        })

    class_indices = list(range(num_classes))

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=class_indices,
        average="weighted",
        zero_division=0,
    )
    
    support_sum = np.sum(supports)

    metrics = {
        "acc": accuracy,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(macro_f1),
        "weighted_precision": (
            float(np.average(precisions, weights=supports))
            if support_sum else 0.0
        ),
        "weighted_recall": (
            float(np.average(recalls, weights=supports))
            if support_sum else 0.0
        ),
        "weighted_f1": float(weighted_f1),
        "total": total,
    }
    
    confusion_df = pd.DataFrame(
        matrix,
        index=[f"pred_{name}" for name in class_names],
        columns=[f"true_{name}" for name in class_names],
    )
    return metrics, pd.DataFrame(per_class_rows), confusion_df


def normalize_manifest(manifest_df, task):
    required_columns = {"track_id", "frame_idx", "cell_type"}
    missing_columns = required_columns.difference(manifest_df.columns)
    if missing_columns:
        raise ValueError(
            f"Test CSV is missing required columns: {sorted(missing_columns)}"
        )

    manifest_df = manifest_df.copy()
    manifest_df["cell_type"] = manifest_df["cell_type"].replace({"CD16": "CD66b"})

    if task == "2cls":
        manifest_df["cell_type"] = manifest_df["cell_type"].replace({
            "CD3": "WBC",
            "CD14": "WBC",
            "CD66b": "WBC",
        })

    allowed_classes = TASK_CONFIGS[task]["classes"]
    manifest_df = manifest_df[
        manifest_df["cell_type"].isin(allowed_classes)
    ].copy()
    if manifest_df.empty:
        raise ValueError(
            f"No {task} samples remain after filtering for {allowed_classes}"
        )
    return manifest_df


def load_fold_directory(csv_dir, test_folds):
    frames = []
    for fold_index in test_folds:
        csv_path = os.path.join(csv_dir, f"fold_{fold_index}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise ValueError("No fold CSV files were loaded")
    return pd.concat(frames, ignore_index=True)


def load_manifest(args):
    if args.test_csv is not None:
        conflicting = [
            args.test_csv_dir,
            args.chc_csv_dir,
            args.wbc_csv_dir,
            args.test_folds,
        ]
        if any(value is not None for value in conflicting):
            raise ValueError(
                "--test_csv cannot be combined with fold-directory arguments"
            )
        if not os.path.isfile(args.test_csv):
            raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
        return normalize_manifest(pd.read_csv(args.test_csv), args.task), "single test CSV"

    uses_binary_directories = args.chc_csv_dir is not None or args.wbc_csv_dir is not None
    if uses_binary_directories:
        if args.task != "2cls":
            raise ValueError("--chc_csv_dir and --wbc_csv_dir are only valid for --task 2cls")
        if args.chc_csv_dir is None or args.wbc_csv_dir is None or args.test_folds is None:
            raise ValueError(
                "--chc_csv_dir, --wbc_csv_dir, and --test_folds must be provided together"
            )
        chc_df = load_fold_directory(args.chc_csv_dir, args.test_folds)
        wbc_df = load_fold_directory(args.wbc_csv_dir, args.test_folds)
        manifest_df = pd.concat([chc_df, wbc_df], ignore_index=True)
        return normalize_manifest(manifest_df, args.task), "separate CHC and WBC fold directories"

    if args.test_csv_dir is not None or args.test_folds is not None:
        if args.test_csv_dir is None or args.test_folds is None:
            raise ValueError("--test_csv_dir and --test_folds must be provided together")
        manifest_df = load_fold_directory(args.test_csv_dir, args.test_folds)
        return normalize_manifest(manifest_df, args.task), "test fold directory"

    return None, None


def build_ordered_frame_manifest(manifest_df, max_seq_length):
    ordered_parts = []
    for track_id, group_df in manifest_df.groupby("track_id"):
        group_df = group_df.sort_values("frame_idx")
        num_sequences = (len(group_df) + max_seq_length - 1) // max_seq_length
        for sequence_chunk in range(num_sequences):
            start = sequence_chunk * max_seq_length
            end = min(start + max_seq_length, len(group_df))
            sequence_df = group_df.iloc[start:end].copy()
            sequence_df["_sequence_index"] = len(ordered_parts)
            sequence_df["_sequence_chunk"] = sequence_chunk
            sequence_df["_track_id"] = track_id
            ordered_parts.append(sequence_df)

    if not ordered_parts:
        raise ValueError("No frames were found after filtering")
    return pd.concat(ordered_parts, ignore_index=True)


def majority_label(values):
    counts = values.value_counts()
    winners = counts[counts == counts.max()].index.tolist()
    return int(sorted(winners)[0])


def validate_predictions(predictions_df, expected_num_classes):
    if "gt" not in predictions_df.columns:
        raise ValueError("preds_test.csv must contain a 'gt' column")

    probability_columns = [
        column for column in predictions_df.columns if column != "gt"
    ]
    if not probability_columns or not all(str(column).isdigit() for column in probability_columns):
        raise ValueError(
            "preds_test.csv must contain only 'gt' and numeric probability columns"
        )
    probability_columns = sorted(probability_columns, key=lambda column: int(column))
    if len(probability_columns) != expected_num_classes:
        raise ValueError(
            f"Expected {expected_num_classes} probability columns for this task, "
            f"found {len(probability_columns)}: {probability_columns}"
        )

    for column in ["gt"] + probability_columns:
        predictions_df[column] = pd.to_numeric(predictions_df[column], errors="raise")
    return probability_columns


def get_sequence_prediction(group, probability_columns, aggregation_method):
    mean_probabilities = group[probability_columns].mean(axis=0)
    if aggregation_method == "mean_prob":
        prediction = int(np.argmax(mean_probabilities.to_numpy()))
        return prediction, mean_probabilities
    if aggregation_method == "median_prob":
        median_probabilities = group[probability_columns].median(axis=0)
        prediction = int(np.argmax(median_probabilities.to_numpy()))
        return prediction, median_probabilities
    if aggregation_method == "majority_vote":
        frame_predictions = group[probability_columns].to_numpy().argmax(axis=1)
        return majority_label(pd.Series(frame_predictions)), mean_probabilities
    raise ValueError(f"Unsupported aggregation method: {aggregation_method}")


def build_track_predictions(predictions_df, frame_manifest_df, aggregation_method, num_classes):
    probability_columns = validate_predictions(predictions_df, num_classes)
    if len(predictions_df) != len(frame_manifest_df):
        raise ValueError(
            "preds_test.csv rows do not match ordered test CSV rows: "
            f"{len(predictions_df)} vs {len(frame_manifest_df)}"
        )

    merged_df = predictions_df.copy()
    merged_df["sequence_index"] = frame_manifest_df["_sequence_index"].to_numpy()
    merged_df["sequence_chunk"] = frame_manifest_df["_sequence_chunk"].to_numpy()
    merged_df["track_id"] = frame_manifest_df["_track_id"].to_numpy()
    merged_df["frame_idx"] = frame_manifest_df["frame_idx"].to_numpy()
    merged_df["cell_type"] = frame_manifest_df["cell_type"].to_numpy()
    if "image_file_path" in frame_manifest_df.columns:
        merged_df["image_file_path"] = frame_manifest_df["image_file_path"].to_numpy()

    rows = []
    for sequence_index, group in merged_df.groupby("sequence_index", sort=False):
        prediction, aggregate_probabilities = get_sequence_prediction(
            group,
            probability_columns,
            aggregation_method,
        )
        ground_truth = majority_label(group["gt"].astype(int))
        row = {
            "sequence_index": int(sequence_index),
            "track_id": group["track_id"].iloc[0],
            "sequence_chunk": int(group["sequence_chunk"].iloc[0]),
            "num_frames": len(group),
            "gt": ground_truth,
            "pred": prediction,
            "correct": int(prediction == ground_truth),
            "first_frame_idx": group["frame_idx"].iloc[0],
            "last_frame_idx": group["frame_idx"].iloc[-1],
            "cell_type": group["cell_type"].iloc[0],
        }
        if "image_file_path" in group.columns:
            row["first_image_file_path"] = group["image_file_path"].iloc[0]
        for column in probability_columns:
            row[column] = float(aggregate_probabilities[column])
        rows.append(row)

    return pd.DataFrame(rows)


def build_predictions_by_sequence_length(
    predictions_df,
    sequence_length,
    drop_incomplete,
    aggregation_method,
    num_classes,
):
    probability_columns = validate_predictions(predictions_df, num_classes)
    rows = []
    for start in range(0, len(predictions_df), sequence_length):
        end = min(start + sequence_length, len(predictions_df))
        group = predictions_df.iloc[start:end]
        if drop_incomplete and len(group) < sequence_length:
            continue

        prediction, aggregate_probabilities = get_sequence_prediction(
            group,
            probability_columns,
            aggregation_method,
        )
        ground_truth = majority_label(group["gt"].astype(int))
        row = {
            "sequence_index": len(rows),
            "start_row": start,
            "end_row": end - 1,
            "num_frames": len(group),
            "gt": ground_truth,
            "pred": prediction,
            "correct": int(prediction == ground_truth),
        }
        for column in probability_columns:
            row[column] = float(aggregate_probabilities[column])
        rows.append(row)
    return pd.DataFrame(rows)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate SFCell frame-level predictions into sequence-level results "
            "for binary or three-class classification."
        )
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIGS))
    parser.add_argument("--preds_test_csv", required=True, help="Frame-level predictions CSV.")
    parser.add_argument(
        "--test_csv",
        default=None,
        help="Recommended: one task-ready test CSV containing track_id and frame_idx.",
    )
    parser.add_argument("--test_csv_dir", default=None, help="Legacy test fold CSV directory.")
    parser.add_argument("--chc_csv_dir", default=None, help="Legacy two-class CHC fold directory.")
    parser.add_argument("--wbc_csv_dir", default=None, help="Legacy two-class WBC fold directory.")
    parser.add_argument("--test_folds", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--output_csv", default="preds_test_track.csv")
    parser.add_argument("--metrics_csv", default="metrics_test_track.csv")
    parser.add_argument("--per_class_metrics_csv", default="per_class_metrics_test_track.csv")
    parser.add_argument("--confusion_matrix_csv", default="confusion_matrix_test_track.csv")
    parser.add_argument("--log_file", default=None)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=10,
        help="Sequence length used when the frame-level predictions were generated.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Fallback grouping length when no test manifest is provided.",
    )
    parser.add_argument(
        "--drop_incomplete",
        action="store_true",
        help="Drop the final fallback group if it is shorter than seq_len.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.max_seq_length <= 0:
        raise ValueError("--max_seq_length must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq_len must be positive")

    task_config = TASK_CONFIGS[args.task]
    aggregation_method = "majority_vote"
    num_classes = len(task_config["classes"])

    predictions_path = os.path.abspath(args.preds_test_csv)
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Predictions CSV file not found: {predictions_path}")
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(predictions_path))
    os.makedirs(output_dir, exist_ok=True)
    log_file = setup_logging(
        output_dir,
        args.log_file or f"postprocess_track_{args.task}.log",
    )

    predictions_df = pd.read_csv(predictions_path)
    manifest_df, manifest_mode = load_manifest(args)
    if manifest_df is not None:
        ordered_manifest_df = build_ordered_frame_manifest(
            manifest_df,
            args.max_seq_length,
        )
        track_df = build_track_predictions(
            predictions_df,
            ordered_manifest_df,
            aggregation_method,
            num_classes,
        )
        aggregation_mode = f"{manifest_mode}: track_id + sequence_chunk"
    else:
        track_df = build_predictions_by_sequence_length(
            predictions_df,
            args.seq_len,
            args.drop_incomplete,
            aggregation_method,
            num_classes,
        )
        aggregation_mode = f"every {args.seq_len} consecutive rows"

    if track_df.empty:
        raise ValueError("No sequence-level predictions were generated")

    probability_columns = sorted(
        [column for column in track_df.columns if str(column).isdigit()],
        key=lambda column: int(column),
    )
    class_names = list(task_config["classes"])

    if len(class_names) != len(probability_columns):
        raise ValueError(
            f"Task {args.task} expects {len(class_names)} probability columns, "
            f"but found {len(probability_columns)}: {probability_columns}"
        )
    
    metrics, per_class_df, confusion_df = compute_metrics(
        track_df["gt"].astype(int).to_numpy(),
        track_df["pred"].astype(int).to_numpy(),
        class_names,
    )

    output_paths = {
        "predictions": resolve_output_path(output_dir, args.output_csv),
        "metrics": resolve_output_path(output_dir, args.metrics_csv),
        "per_class": resolve_output_path(output_dir, args.per_class_metrics_csv),
        "confusion": resolve_output_path(output_dir, args.confusion_matrix_csv),
    }
    track_df.to_csv(output_paths["predictions"], index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(output_paths["metrics"], index=False, encoding="utf-8-sig")
    per_class_df.to_csv(output_paths["per_class"], index=False, encoding="utf-8-sig")
    confusion_df.to_csv(output_paths["confusion"], encoding="utf-8-sig")

    log_metrics(metrics, per_class_df, confusion_df)
    logging.info("Task: %s", args.task)
    logging.info("Loaded frame-level predictions: %s", predictions_path)
    logging.info("Saved sequence-level predictions: %s", output_paths["predictions"])
    logging.info("Saved metrics: %s", output_paths["metrics"])
    logging.info("Saved per-class metrics: %s", output_paths["per_class"])
    logging.info("Saved confusion matrix: %s", output_paths["confusion"])
    logging.info("Log file: %s", log_file)
    logging.info("Aggregation mode: %s", aggregation_mode)
    logging.info("Aggregation method: %s", aggregation_method)
    logging.info("Sequences: %d, sequence accuracy: %.4f", len(track_df), metrics["acc"])


if __name__ == "__main__":
    main()

'''
usage:
python postprocess_SFCell.py \
  --task 2cls \
  --preds_test_csv test/chc-wbc-external/preds_test.csv \
  --test_csv data/splits/chc_wbc/external_test.csv \
  --max_seq_length 10 \
  --output_dir test/chc-wbc-external-track
'''
