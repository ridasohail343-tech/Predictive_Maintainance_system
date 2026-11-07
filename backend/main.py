from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import sqlite3
app = FastAPI()

model = joblib.load("artifacts/anomaly_model.pkl")


class Sensor_data(BaseModel):
    TP2: float
    TP3: float
    H1: float
    Motor_current: float


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(data: Sensor_data):

    input_data = [[
        data.TP2,
        data.TP3,
        data.H1,
        data.Motor_current
    ]]

    prediction = model.predict(input_data)

    if prediction[0] == -1:
        result = "Anomaly Detected"
    else:
        result = "Normal"

    conn = sqlite3.connect("maintenance.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (TP2, TP3, H1, Motor_current, prediction)
        VALUES (?, ?, ?, ?, ?)
    """, (
        data.TP2,
        data.TP3,
        data.H1,
        data.Motor_current,
        result
    ))

    conn.commit()
    conn.close()

    return {"prediction": result}

@app.get("/predictions")
def get_predictions():

    conn = sqlite3.connect("maintenance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions")

    records = cursor.fetchall()

    conn.close()

    return {"predictions": records}




