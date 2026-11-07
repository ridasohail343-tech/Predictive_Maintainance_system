
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# Load processed data
df = pd.read_csv("data/processed/metro_plt.csv")

# Select sensor features
features = ["TP2", "TP3", "H1", "Motor_current"]

# Train anomaly detection model
def train_anomaly_model(df):
    train = df[features].sample(
        n=min(50000, len(df)),
        random_state=42
    )

    model = IsolationForest(
        n_estimators=50,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(train)

    return model


# Detect anomalies
def detect_anomalies(model, df):
    df = df.copy()

    df["predicted_anomaly"] = (
        model.predict(df[features]) == -1
    ).astype(int)

    return df


# Train model
model = train_anomaly_model(df)

# Predict anomalies
df = detect_anomalies(model, df)

# Save trained model
joblib.dump(model, "artifacts/anomaly_model.pkl")

# Save anomaly results
df.to_csv(
    "data/processed/anomaly_results.csv",
    index=False,
    chunksize=10000
)

# Display results
print(df["predicted_anomaly"].value_counts())

print("\nAnomaly summary:")
print(df.groupby("predicted_anomaly")[features].mean())

anomaly_rate = df["predicted_anomaly"].mean() * 100
print(f"Anomaly rate: {anomaly_rate:.2f}%")

print("Anomaly detection completed!")
