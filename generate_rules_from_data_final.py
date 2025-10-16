# generate_rules_from_data_final.py
"""
Train Decision Tree model from data_gaya_belajar.xlsx (sheet default),
do train/test split + evaluation, save model (versioned), rules and metrics.
"""

import pandas as pd
import joblib
import datetime
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import os

# CONFIG
DATAFILE = "data_gaya_belajar.xlsx"
MODEL_DIR = "."
RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_DEPTH = 6  # change if desired

def main():
    # 1. Load data
    print("🔁 Membaca dataset:", DATAFILE)
    df = pd.read_excel(DATAFILE)

    # 2. Prepare X and y (drop identifier & label)
    df = df.copy()
    y = df["LearningStyle"]
    X = df.drop(columns=["StudentID", "LearningStyle"], errors="ignore")

    # 3. Optional: scale features (minmax) to keep consistency
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. Train Decision Tree
    print("🧠 Melatih Decision Tree (max_depth=", MAX_DEPTH, ") ...")
    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n✅ Evaluasi model:\nAkurasi (test): {acc:.4f}\n")
    print("Classification report:\n", report)
    print("Confusion matrix:\n", cm)

    # 7. Save model (versioned) and scaler
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"learning_style_model_{ts}.pkl"
    scaler_name = f"scaler_{ts}.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_name))
    joblib.dump(scaler, os.path.join(MODEL_DIR, scaler_name))
    print(f"\n💾 Model dan scaler tersimpan: {model_name}, {scaler_name}")

    # 8. Save readable rules (based on scaled feature thresholds)
    rules_text = export_text(model, feature_names=list(X.columns))
    rules_file = f"learning_rules_{ts}.txt"
    with open(rules_file, "w", encoding="utf-8") as f:
        f.write(rules_text)
    print(f"📜 Rules disimpan ke: {rules_file}\n")
    print(rules_text)

    # 9. Save metrics to file
    metrics_file = f"model_metrics_{ts}.txt"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc}\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n\n")
        f.write("Confusion matrix:\n")
        f.write(str(cm) + "\n")
    print(f"📊 Metrics disimpan ke: {metrics_file}")

    # 10. Optionally save a short summary CSV of important info
    summary_df = pd.DataFrame({
        "model_file":[model_name],
        "scaler_file":[scaler_name],
        "rules_file":[rules_file],
        "metrics_file":[metrics_file],
        "accuracy":[acc],
        "created_at":[ts]
    })
    summary_df.to_csv(f"model_summary_{ts}.csv", index=False)
    print("✅ Selesai. File summary CSV disimpan.")

if __name__ == "__main__":
    main()
