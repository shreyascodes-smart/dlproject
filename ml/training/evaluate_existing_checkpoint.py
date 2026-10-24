
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.flood_dataset import WorldFloodsClassificationDataset
from ml.inference.flood_inference import FloodInference

DATA_DIR = PROJECT_ROOT / "data/raw/flood/worldfloods_v2"
CHECKPOINT = PROJECT_ROOT / "models/flood/best_flood_model.pt"


def main():
    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")
    print("Checkpoint exists:", CHECKPOINT.exists())

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    dataset = WorldFloodsClassificationDataset(
        str(DATA_DIR),
        split="test",
        flood_threshold=0.05,
    )
    engine = FloodInference(str(CHECKPOINT))

    # Counts: actual class vs predicted class
    tn = fp = fn = tp = 0
    failures = []
    results = []

    print(f"\nPaired test samples found: {len(dataset.samples)}")

    for sample in dataset.samples:
        name = sample["name"]

        try:
            # Derive the reference label directly.
            # Do not use dataset[i], because its current error
            # handling can silently substitute another sample.
            gt = dataset._read_mask(sample["gt"])
            pw = (
                dataset._read_mask(sample["pw"])
                if sample["pw"] is not None
                else None
            )
            actual, flood_ratio = dataset._derive_label(gt, pw)

            # Predict using the same checkpoint as the application.
            result = engine.predict(
                str(sample["s2"]),
                generate_gradcam=False,
            )
            pred = result["prediction"]
            predicted = int(pred["class"])
            probability = float(pred["flood_probability"])

            if actual == 0 and predicted == 0:
                tn += 1
            elif actual == 0 and predicted == 1:
                fp += 1
            elif actual == 1 and predicted == 0:
                fn += 1
            else:
                tp += 1

            results.append((name, actual, predicted, probability, flood_ratio))

            print(
                f"{name}: actual={actual}, predicted={predicted}, "
                f"flood_score={probability:.4f}, "
                f"label_flood_ratio={flood_ratio:.4f}"
            )

        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"FAILED: {name}: {exc}")

    total = tn + fp + fn + tp
    if total == 0:
        print("\nNo samples were evaluated successfully.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    balanced_accuracy = (recall + specificity) / 2

    print("\n--- TEST RESULTS ---")
    print(f"Successfully evaluated: {total}")
    print(f"Failed samples: {len(failures)}")
    print(f"Confusion matrix counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"F1 score:          {f1:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    print("\nReference-label counts:")
    print(f"Flood: {sum(r[1] == 1 for r in results)}")
    print(f"Non-flood: {sum(r[1] == 0 for r in results)}")

    if failures:
        print("\n--- FAILED SAMPLES ---")
        for name, error in failures:
            print(f"{name}: {error}")

    print(
        "\nNOTE: These metrics evaluate the current derived labels, "
        "not independently verified real-world flood status."
    )


if __name__ == "__main__":
    main()
