"""
generate_dataset.py
-------------------
Creates a small SAMPLE network-traffic dataset (data/network_traffic.csv).

IMPORTANT: this data is SYNTHETIC (made up with random numbers that follow
realistic patterns). It is only for learning/demo purposes - it is not
captured from a real network. Run it with:  python generate_dataset.py

Each row describes ONE network connection using 4 features:
    duration            - how long the connection lasted (seconds)
    packet_count        - how many packets were sent
    bytes_transferred   - total bytes sent
    failed_connections  - failed attempts (e.g. wrong passwords, closed ports)
The label is "normal" (0) or "suspicious" (1).
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_FILE = Path(__file__).parent / "data" / "network_traffic.csv"
rng = np.random.default_rng(seed=42)  # fixed seed -> same dataset every run


def make_normal(n):
    """Everyday traffic: browsing, downloads, API calls."""
    packets = np.clip(rng.lognormal(mean=3.7, sigma=0.8, size=n), 2, 800)
    duration = np.clip(rng.lognormal(mean=2.5, sigma=0.9, size=n), 0.1, 400)
    bytes_total = packets * rng.uniform(200, 1500, size=n)
    failed = np.clip(rng.poisson(0.25, size=n), 0, 2)
    # A few "busy but legitimate" users (big downloads, a mistyped password)
    busy = rng.random(n) < 0.12
    packets[busy] *= rng.uniform(3, 14, busy.sum())
    bytes_total[busy] *= rng.uniform(3, 14, busy.sum())
    failed[busy] = rng.integers(1, 8, busy.sum())
    return pd.DataFrame({"duration": duration, "packet_count": packets,
                         "bytes_transferred": bytes_total,
                         "failed_connections": failed, "label": "normal"})


def make_ddos(n):
    """Flood: very many small packets in a short time."""
    packets = rng.lognormal(mean=6.8, sigma=1.0, size=n)
    return pd.DataFrame({"duration": rng.uniform(0.1, 6, n),
                         "packet_count": packets,
                         "bytes_transferred": packets * rng.uniform(40, 400, n),
                         "failed_connections": rng.poisson(0.8, n),
                         "label": "suspicious"})


def make_port_scan(n):
    """Probe: many quick, tiny connections that mostly fail."""
    packets = rng.uniform(15, 250, n)
    return pd.DataFrame({"duration": rng.uniform(0.5, 20, n),
                         "packet_count": packets,
                         "bytes_transferred": packets * rng.uniform(40, 140, n),
                         "failed_connections": rng.integers(3, 70, n),
                         "label": "suspicious"})


def make_brute_force(n):
    """Password guessing: repeated login attempts, many failures."""
    packets = rng.uniform(60, 500, n)
    return pd.DataFrame({"duration": rng.uniform(15, 300, n),
                         "packet_count": packets,
                         "bytes_transferred": packets * rng.uniform(100, 350, n),
                         "failed_connections": rng.integers(3, 45, n),
                         "label": "suspicious"})


def main():
    data = pd.concat([make_normal(360), make_ddos(80), make_port_scan(80),
                      make_brute_force(80)], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    data["duration"] = data["duration"].round(2)
    data["packet_count"] = data["packet_count"].round().astype(int)
    data["bytes_transferred"] = data["bytes_transferred"].round().astype(int)
    data["failed_connections"] = data["failed_connections"].astype(int)
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(data)} rows to {OUTPUT_FILE}")
    print(data["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
