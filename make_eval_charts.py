import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("C:/LLM_PROJECTR")
DATA_DIR = BASE / "data"

# Choose which CSV you have:
# - If you used label-only script: eval_results.csv
# - If you used score script: eval_scores.csv
CSV_PATH = DATA_DIR / "eval_results.csv"   # change if needed
OUT_DIR = DATA_DIR / "charts"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Normalize columns
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column (Correct / Partially Correct / Wrong).")

    # Ensure consistent label casing
    df["label"] = df["label"].astype(str).str.strip()

    # ----------------------------
    # 1) Bar chart of label counts
    # ----------------------------
    label_order = ["Correct", "Partially Correct", "Wrong"]
    counts = df["label"].value_counts().reindex(label_order, fill_value=0)

    plt.figure()
    counts.plot(kind="bar")
    plt.title("Evaluation Results (Counts)")
    plt.xlabel("Label")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out1 = OUT_DIR / "label_counts_bar.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # ----------------------------
    # 2) Pie chart distribution
    # ----------------------------
    plt.figure()
    counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Evaluation Results (Distribution)")
    plt.ylabel("")  # remove default "label" axis name
    plt.tight_layout()
    out2 = OUT_DIR / "label_distribution_pie.png"
    plt.savefig(out2, dpi=200)
    plt.close()

    # ----------------------------
    # 3) Accuracy metrics text
    # ----------------------------
    total = len(df)
    correct = int(counts.get("Correct", 0))
    partial = int(counts.get("Partially Correct", 0))
    wrong = int(counts.get("Wrong", 0))

    strict_acc = correct / total if total else 0.0
    soft_acc = (correct + 0.5 * partial) / total if total else 0.0

    metrics_txt = (
        f"Total questions: {total}\n"
        f"Correct: {correct}\n"
        f"Partially Correct: {partial}\n"
        f"Wrong: {wrong}\n"
        f"Strict Accuracy (Correct/Total): {strict_acc:.3f}\n"
        f"Soft Accuracy ((Correct+0.5*Partial)/Total): {soft_acc:.3f}\n"
    )
    out_txt = OUT_DIR / "metrics.txt"
    out_txt.write_text(metrics_txt, encoding="utf-8")

    # ----------------------------
    # 4) Cumulative accuracy curve (optional)
    #     - Uses row order (id) to show how performance evolves
    # ----------------------------
    # Create a numeric score even if CSV doesn't have "score"
    score_map = {"Correct": 1.0, "Partially Correct": 0.5, "Wrong": 0.0}
    df["auto_score"] = df["label"].map(score_map).fillna(0.0)

    # Sort by id if exists
    if "id" in df.columns:
        df = df.sort_values("id")

    df["cum_soft_acc"] = df["auto_score"].expanding().mean()
    df["cum_strict_acc"] = (df["label"].eq("Correct")).expanding().mean()

    plt.figure()
    plt.plot(df["cum_strict_acc"].values, label="Cumulative Strict Accuracy")
    plt.title("Cumulative Strict Accuracy Over Questions")
    plt.xlabel("Question Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out3 = OUT_DIR / "cumulative_strict_accuracy.png"
    plt.savefig(out3, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df["cum_soft_acc"].values, label="Cumulative Soft Accuracy")
    plt.title("Cumulative Soft Accuracy Over Questions")
    plt.xlabel("Question Index")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    out4 = OUT_DIR / "cumulative_soft_accuracy.png"
    plt.savefig(out4, dpi=200)
    plt.close()

    print("✅ Charts saved to:", OUT_DIR)
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)
    print(" -", out4)
    print(" -", out_txt)


if __name__ == "__main__":
    main()
