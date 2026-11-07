
import streamlit as st
import pandas as pd
from pathlib import Path
import sqlite3
import requests

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide"
)

st.title("Predictive Maintenance Dashboard")

# Database path
db_path = Path(__file__).resolve().parents[1] / "maintenance.db"


# Create database and table
def create_database():

    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            TP2 REAL,
            TP3 REAL,
            H1 REAL,
            Motor_current REAL,
            prediction TEXT
        )
    """)

    conn.commit()
    conn.close()


create_database()


# Sensor input form
st.subheader("Enter Sensor Readings")

with st.form("sensor_form"):

    tp2 = st.number_input("TP2", value=5.5)
    tp3 = st.number_input("TP3", value=9.0)
    h1 = st.number_input("H1", value=3.3)
    motor = st.number_input("Motor Current", value=4.4)

    submitted = st.form_submit_button(
        "Predict Equipment Status"
    )

    if submitted:

        sensor_data = {
            "TP2": tp2,
            "TP3": tp3,
            "H1": h1,
            "Motor_current": motor
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=sensor_data,
                timeout=15
            )

            if response.status_code == 200:

                result = response.json()["prediction"]

                st.success(f"Prediction: {result}")

                st.rerun()

            else:
                st.error("Prediction failed.")

        except requests.exceptions.ConnectionError:
            st.error("FastAPI server is not running.")

        except requests.exceptions.Timeout:
            st.error("Prediction request timed out.")


# Fetch prediction history
conn = sqlite3.connect(db_path)

df = pd.read_sql_query(
    """
    SELECT *
    FROM predictions
    ORDER BY timestamp DESC
    """,
    conn
)

conn.close()


# Calculate metrics
total = len(df)

anomalies = (
    df["prediction"] == "Anomaly Detected"
).sum()

normal = (
    df["prediction"] == "Normal"
).sum()


# Display metrics
st.subheader("Prediction Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Predictions", total)

col2.metric("Anomalies Detected", anomalies)

col3.metric("Normal Readings", normal)


# Anomaly distribution chart
st.subheader("Equipment Anomaly Analysis")

if total > 0:

    chart_data = pd.DataFrame({
        "Status": ["Normal", "Anomaly Detected"],
        "Count": [normal, anomalies]
    })

    st.bar_chart(
        chart_data,
        x="Status",
        y="Count"
    )

else:
    st.info("No predictions available yet.")


# Equipment health status
st.subheader("Equipment Health Status")

if total > 0:

    latest_prediction = df.iloc[0]["prediction"]

    if latest_prediction == "Anomaly Detected":

        st.error(
            "WARNING: Anomaly Detected"
        )

    else:

        st.success(
            "Equipment Status: NORMAL"
        )

else:
    st.info("Waiting for sensor predictions.")


# Prediction history
st.subheader("Prediction History")

st.dataframe(
    df,
    use_container_width=True
)


# Download prediction history
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Prediction History",
    data=csv,
    file_name="prediction_history.csv",
    mime="text/csv"
)
