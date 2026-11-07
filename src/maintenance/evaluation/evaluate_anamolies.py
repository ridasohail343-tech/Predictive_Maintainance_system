
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Load anomaly results
df = pd.read_csv("data/processed/anomaly_results.csv")

# Actual and predicted labels
y_true = df["known_failure"]
y_pred = df["predicted_anomaly"]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Failure"]
    )
)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nConfusion Matrix Details:")

tn, fp, fn, tp = cm.ravel()

print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# Analyze false positives
false_positives = df[
    (df["known_failure"] == 0) &
    (df["predicted_anomaly"] == 1)
]

print("False positive count:", len(false_positives))

print("\nFalse positive sensor averages:")
print(
    false_positives[
        ["TP2", "TP3", "H1", "Motor_current"]
    ].mean()
)

print("\nFalse positive date range:")
print(false_positives["timestamp"].min())
print(false_positives["timestamp"].max())

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Extract month
df["month"] = df["timestamp"].dt.to_period("M")

# Monthly anomaly counts
monthly_anomalies = df.groupby("month")["predicted_anomaly"].sum()

print("\nMonthly anomaly counts:")
print(monthly_anomalies)

# Monthly failure counts
monthly_failures = df.groupby("month")["known_failure"].sum()

print("\nMonthly failure counts:")
print(monthly_failures)

print("\nMonthly anomaly counts:")
print(monthly_anomalies)

# Calculate monthly anomaly and failure rates
monthly_analysis = df.groupby("month").agg(
    total_rows=("predicted_anomaly", "count"),
    anomalies=("predicted_anomaly", "sum"),
    failures=("known_failure", "sum")
)

monthly_analysis["anomaly_rate"] = (
    monthly_analysis["anomalies"] /
    monthly_analysis["total_rows"] * 100
)

monthly_analysis["failure_rate"] = (
    monthly_analysis["failures"] /
    monthly_analysis["total_rows"] * 100
)

print("\nMonthly analysis:")
print(monthly_analysis.round(2))

import matplotlib.pyplot as plt

monthly_analysis[["anomaly_rate", "failure_rate"]].plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("Monthly Anomaly Rate vs Failure Rate")
plt.xlabel("Month")
plt.ylabel("Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

