"""
Generates a synthetic media-viewership dataset (~1.2M events) that mimics the
kind of audience-measurement data Nielsen works with: who watched what, for
how long, on which device, in which region.

Two files are produced:
  - viewership_events.csv  (the "fact" table, ~1.2M rows)
  - shows_metadata.csv     (the "dimension" table, ~500 shows)
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

N_EVENTS = 1_200_000
N_SHOWS = 500
N_DEVICES = 80_000

GENRES = ["Drama", "Comedy", "News", "Sports", "Reality", "Documentary", "Kids", "Talk Show"]
NETWORKS = ["NetA", "NetB", "NetC", "NetD", "NetE"]
REGIONS = ["North", "South", "East", "West", "Central"]
AGE_GROUPS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55+"]
DEVICE_TYPES = ["TV", "Mobile", "Tablet", "Desktop", "Connected TV"]

# ---- shows_metadata.csv ----
with open("shows_metadata.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["show_id", "genre", "network", "is_original", "premiere_year"])
    for show_id in range(1, N_SHOWS + 1):
        w.writerow([
            show_id,
            random.choice(GENRES),
            random.choice(NETWORKS),
            random.choice([0, 1]),
            random.randint(2015, 2026),
        ])

# ---- viewership_events.csv ----
start = datetime(2026, 1, 1)
with open("viewership_events.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["event_id", "device_id", "show_id", "timestamp", "watch_duration_sec",
                "region", "age_group", "device_type"])
    for event_id in range(1, N_EVENTS + 1):
        ts = start + timedelta(minutes=random.randint(0, 60 * 24 * 200))
        # occasionally inject nulls / bad rows to make cleaning meaningful
        duration = random.choice([None]) if random.random() < 0.01 else random.randint(10, 7200)
        w.writerow([
            event_id,
            random.randint(1, N_DEVICES),
            random.randint(1, N_SHOWS),
            ts.isoformat(),
            duration if duration is not None else "",
            random.choice(REGIONS),
            random.choice(AGE_GROUPS),
            random.choice(DEVICE_TYPES),
        ])

print("Done: viewership_events.csv, shows_metadata.csv")
