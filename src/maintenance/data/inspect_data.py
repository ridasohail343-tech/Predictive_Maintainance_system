import pandas as pd
from pathlib import Path

file_path = Path(r"C:\Users\ridas\Documents\Predictive-maintaince-system\data\raw\metro_plt.csv")
df=pd.read_csv(file_path)


# print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df)

# preprocessing
print(df.isnull().sum())
print(df.describe())
print(df.info())
print("Duplicate rows:", df.duplicated().sum())
df['timestamp']=pd.to_datetime(df['timestamp'])
print(df["timestamp"].min())
print(df["timestamp"].max())

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.dates as mdates

# df load karne ke baad:
df['timestamp'] = pd.to_datetime(df['timestamp'])

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['timestamp'][:500], df['TP2'][:500])
ax.set_title('TP2 Sensor Readings')
ax.set_xlabel('Timestamp')
ax.set_ylabel('TP2')
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

plt.tight_layout()
plt.show()



print(df.isnull().sum())

df = df.drop(columns=["Unnamed: 0"])

print(df.columns)

df.to_csv("data/processed/metro_plt.csv", index=False)

print(df["timestamp"].is_monotonic_increasing)
print(df[["TP2", "TP3", "H1", "Motor_current"]].corr())
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df[["TP2", "TP3", "H1", "Motor_current"]].corr(), annot=True)

plt.show()
print(df.columns.tolist())

failure_periods = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00")
]

df["known_failure"] = 0

for start, end in failure_periods:
    mask = df["timestamp"].between(start, end)
    df.loc[mask, "known_failure"] = 1
import os

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/metro_plt.csv",
    index=False,
    chunksize=10000
)

print(df["known_failure"].value_counts())

split_date = pd.Timestamp("2020-04-18")

train = df[df["timestamp"] < split_date]
test = df[df["timestamp"] >= split_date]

print("Training data:", train.shape)
print("Testing data:", test.shape)
print("Training failures:", train["known_failure"].sum())
print("Testing failures:", test["known_failure"].sum())

