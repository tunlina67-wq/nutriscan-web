"""Train the Random Forest health-grade classifier on Open Food Facts.

Usage:
    python train_classifier.py                      # downloads the first 400k products (~5-10 min)
    python train_classifier.py --rows 200000        # fewer rows = faster
    python train_classifier.py --csv my_data.csv    # use a CSV/TSV you already downloaded

Saves:  models/grade_rf.joblib  and  models/classifier_report.txt  (+ confusion matrix PNG)
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

OFF_URL = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
COLS = {
    "energy-kcal_100g": "energy_kcal", "fat_100g": "total_fat_g",
    "saturated-fat_100g": "sat_fat_g", "carbohydrates_100g": "carbs_g",
    "sugars_100g": "sugar_g", "proteins_100g": "protein_g", "sodium_100g": "sodium_g",
}
FEATURES = list(COLS.values()) + ["is_beverage"]
OUT = Path(__file__).resolve().parent / "models"


def load_data(csv, rows):
    src = csv or OFF_URL
    print(f"Reading {rows:,} rows from {src} …")
    usecols = list(COLS) + ["nutriscore_grade", "pnns_groups_1"]
    sep = "," if csv and csv.endswith(".csv") and "\t" not in open(csv).readline() else "\t"
    df = pd.read_csv(src, sep=sep, usecols=lambda c: c in usecols, nrows=rows,
                     low_memory=False, on_bad_lines="skip")
    df = df.rename(columns=COLS)
    df["nutriscore_grade"] = df["nutriscore_grade"].astype(str).str.lower().str.strip()
    df = df[df["nutriscore_grade"].isin(list("abcde"))]
    df["is_beverage"] = df.get("pnns_groups_1", "").astype(str).str.contains("everage").astype(float)
    df = df.dropna(subset=["energy_kcal", "sugar_g", "sodium_g", "total_fat_g"])
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # remove impossible values (per 100 g/ml)
    df = df[(df.energy_kcal.between(0, 900)) & (df.sugar_g.between(0, 100)) &
            (df.total_fat_g.between(0, 100)) & (df.sodium_g.between(0, 40)) &
            (df.protein_g.between(0, 100)) & (df.carbs_g.between(0, 100))]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="local Open Food Facts CSV/TSV")
    ap.add_argument("--rows", type=int, default=400_000)
    args = ap.parse_args()

    df = load_data(args.csv, args.rows)
    print(f"Usable products: {len(df):,}")
    print(df["nutriscore_grade"].value_counts().sort_index().to_string())

    X, y = df[FEATURES].values, df["nutriscore_grade"].str.upper().values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=2,
                                   n_jobs=-1, random_state=42, class_weight="balanced")
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    within_one = np.mean(np.abs(np.array([ord(a) for a in pred]) - np.array([ord(b) for b in y_te])) <= 1)

    report = classification_report(y_te, pred, digits=3)
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    text = (f"Random Forest health-grade classifier\n"
            f"Training products: {len(X_tr):,} | Test products: {len(X_te):,}\n"
            f"Test accuracy: {acc:.3f} | Within one grade: {within_one:.3f}\n\n{report}\n"
            "Feature importance:\n" + "\n".join(f"  {f:<12} {v:.3f}" for f, v in importances))
    print("\n" + text)

    OUT.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "test_accuracy": round(acc, 3)},
                OUT / "grade_rf.joblib", compress=("xz", 9))  # ~50 MB, under GitHub's 100 MB limit
    (OUT / "classifier_report.txt").write_text(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = list("ABCDE")
        cm = confusion_matrix(y_te, pred, labels=labels)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(5), labels); ax.set_yticks(range(5), labels)
        ax.set_xlabel("Predicted grade"); ax.set_ylabel("True grade")
        ax.set_title(f"Confusion matrix (accuracy {acc:.1%})")
        for i in range(5):
            for j in range(5):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
        plt.tight_layout(); plt.savefig(OUT / "classifier_confusion_matrix.png", dpi=150)
    except Exception as e:
        print("(skipped confusion matrix plot:", e, ")")

    print(f"\n✅ Saved {OUT / 'grade_rf.joblib'}")


if __name__ == "__main__":
    main()
