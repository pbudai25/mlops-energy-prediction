import pandas as pd
import numpy as np

# Reprodukálható véletlenszámok
np.random.seed(42)

# 1 év, óránkénti adatok
dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31 23:00:00",
    freq="h"
)

df = pd.DataFrame({"timestamp": dates})

# Időbeli jellemzők
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# Alapfogyasztás
base_consumption = 2.5

# Napi fogyasztási minta
daily_pattern = (
    1.5
    * np.sin((df["hour"] - 7) * 2 * np.pi / 24)
)

# Reggeli és esti fogyasztási csúcs
morning_peak = np.where(
    (df["hour"] >= 6) & (df["hour"] <= 9),
    2.0,
    0
)

evening_peak = np.where(
    (df["hour"] >= 17) & (df["hour"] <= 22),
    3.0,
    0
)

# Hétvégi többletfogyasztás
weekend_effect = np.where(
    df["day_of_week"] >= 5,
    0.8,
    0
)

# Véletlen zaj
noise = np.random.normal(0, 0.4, len(df))

# Végső fogyasztás
df["consumption_kwh"] = (
    base_consumption
    + daily_pattern
    + morning_peak
    + evening_peak
    + weekend_effect
    + noise
)

# Negatív értékek elkerülése
df["consumption_kwh"] = df["consumption_kwh"].clip(lower=0.1)

# CSV mentése
df.to_csv("data/energy.csv", index=False)

print("Adathalmaz elkészült!")
print(f"Sorok száma: {len(df)}")
print(df.head())