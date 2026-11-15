"""
generate_traffic_data.py
Simulates 5-minute loop-detector / camera readings for 12 junctions of
Tiruchirappalli for 60 days (2026-08-01 .. 2026-09-29).
Output: data/raw/traffic_readings.csv (the raw feed used by Spark)
"""
import os
import numpy as np
import pandas as pd

def generate_traffic_dataset(output_path=None):
    if output_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_path = os.path.join(base_dir, "data", "raw", "traffic_readings.csv")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rng = np.random.default_rng(42)

    JUNCTIONS = [  # id, name, zone, capacity veh/5min, free-flow speed km/h, demand weight
        (1,  "Central Bus Stand",   "Central",     210, 38, 1.00),
        (2,  "Chathiram Bus Stand", "Central",     190, 36, 0.95),
        (3,  "Palakkarai",          "Central",     170, 32, 0.92),
        (4,  "Thillai Nagar",       "Commercial",  180, 40, 0.90),
        (5,  "Cantonment",          "Commercial",  200, 42, 0.88),
        (6,  "Srirangam Gate",      "Temple",      160, 34, 0.85),
        (7,  "Woraiyur",            "Residential", 150, 40, 0.70),
        (8,  "Tennur",              "Residential", 140, 42, 0.62),
        (9,  "Ariyamangalam",       "Industrial",  220, 48, 0.80),
        (10, "Ponmalai",            "Industrial",  200, 46, 0.72),
        (11, "Airport Road",        "Suburban",    180, 55, 0.55),
        (12, "Samayapuram Toll",    "Highway",     240, 60, 0.65),
    ]

    def hour_profile(h, weekend):
        # smooth double-peak commuting curve (0..1)
        morning = np.exp(-((h - 9.0) ** 2) / (2 * 1.3 ** 2))
        evening = np.exp(-((h - 18.2) ** 2) / (2 * 1.6 ** 2))
        mid = 0.38 * np.exp(-((h - 13.5) ** 2) / (2 * 2.5 ** 2))
        night = 0.06
        if weekend:
            return night + 0.55 * mid + 0.45 * np.exp(-((h - 17.5) ** 2) / (2 * 2.2 ** 2))
        return night + 0.95 * morning + 1.0 * evening + mid

    idx = pd.date_range("2026-08-01 00:00", "2026-09-29 23:55", freq="5min")
    T = len(idx)
    hours = idx.hour.values + idx.minute.values / 60.0
    dow = idx.dayofweek.values
    weekend = dow >= 5

    # Holidays (Independence Day, Vinayagar Chathurthi-like festival day)
    holidays = {"2026-08-15", "2026-09-14"}
    is_holiday = np.array([str(d.date()) in holidays for d in idx])

    # Daily weather Markov chain: Clear / Cloudy / Rain, hourly variation of rain
    days = pd.date_range("2026-08-01", periods=60, freq="D")
    P = {
        "Clear": [0.60, 0.30, 0.10],
        "Cloudy": [0.30, 0.40, 0.30],
        "Rain": [0.15, 0.35, 0.50]
    }
    states = ["Clear", "Cloudy", "Rain"]
    cur = "Clear"
    daily = []
    for _ in days:
        cur = rng.choice(states, p=P[cur])
        daily.append(cur)

    weather = np.empty(T, dtype=object)
    for di, d in enumerate(days):
        m = (idx.normalize() == d)
        w = daily[di]
        arr = np.full(m.sum(), w, dtype=object)
        if w == "Rain":   # afternoon/evening rain bursts
            hh = hours[m]
            arr = np.where((hh >= 14) & (hh <= 21), "Rain", np.where(rng.random(m.sum()) < 0.25, "Cloudy", "Cloudy"))
        weather[m] = arr

    rain = (weather == "Rain").astype(float)
    rows = []

    for jid, name, zone, cap, ff, wt in JUNCTIONS:
        base = np.array([hour_profile(h, w) for h, w in zip(hours, weekend)])
        base = base * wt * 1.30
        base = np.where(is_holiday, base * 0.6, base)

        # Temporally correlated noise (AR(1))
        e = rng.normal(0, 0.06, T)
        ar = np.zeros(T)
        for t in range(1, T):
            ar[t] = 0.92 * ar[t - 1] + e[t]
        demand = np.clip(base * (1 + rain * 0.08) + ar, 0.02, 1.6)

        # Incidents: random onset, duration 6-14 slots (30-70 min)
        inc = np.zeros(T)
        onsets = np.where(rng.random(T) < 0.0009)[0]
        for o in onsets:
            inc[o:o + rng.integers(6, 15)] = 1

        d_eff = demand * (1 + 0.25 * inc)
        speed = ff / (1 + 0.9 * (d_eff ** 3.2))
        speed *= (1 - 0.14 * rain) * (1 - 0.12 * inc)
        speed = np.clip(speed + rng.normal(0, 1.6, T), 3, ff)
        count = np.clip((np.minimum(d_eff, 1.05) * cap + rng.normal(0, 5, T)), 0, None).astype(int)
        occ = np.clip(100 * (d_eff ** 1.1) * 0.62 + rng.normal(0, 2.5, T), 0, 100)
        queue = np.clip((d_eff - 0.75) * 260 + rng.normal(0, 8, T), 0, None)

        df = pd.DataFrame({
            "timestamp": idx,
            "junction_id": jid,
            "junction_name": name,
            "zone": zone,
            "vehicle_count": count,
            "avg_speed_kmph": speed.round(2),
            "occupancy_pct": occ.round(2),
            "queue_length_m": queue.round(1),
            "weather": weather,
            "is_holiday": is_holiday.astype(int),
            "incident_flag": inc.astype(int),
            "free_flow_kmph": ff,
        })
        rows.append(df)

    out = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "junction_id"]).reset_index(drop=True)
    out.insert(0, "reading_id", np.arange(1, len(out) + 1))

    # Inject realistic sensor defects: sensor dropouts / negative values / duplicates
    n = len(out)
    miss = rng.choice(n, int(n * 0.004), replace=False)
    out.loc[miss, "avg_speed_kmph"] = np.nan

    bad = rng.choice(n, int(n * 0.0015), replace=False)
    out.loc[bad, "vehicle_count"] = -1

    dup = out.sample(int(n * 0.002), random_state=1)
    out = pd.concat([out, dup], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

    out.to_csv(output_path, index=False)
    print(f"Generated {len(out):,} raw readings with {out.shape[1]} columns -> {output_path}")
    return output_path

if __name__ == "__main__":
    generate_traffic_dataset()
