"""
SentinelOps - Demo Traffic Generator
====================================
Builds the CSV that the live SOC feed replays.

What is genuine and what is synthesized
---------------------------------------
GENUINE  Every flow feature value and every ``Label`` is sampled verbatim from
         the cleaned CIC-IDS2017 corpus (``data/interim/clean.parquet``). No
         feature is invented, perturbed, or interpolated, so the detections the
         demo shows are the model's real behaviour on real attack traffic.

SYNTHETIC  The *display envelope* only -- ``src_ip``, ``dst_ip``, ``dst_port``
         and ``protocol_name``. The cleaning stage drops the identifier columns
         (they leak the label), so the flows carry no addressing of their own
         and a packet-log view has nothing to render. The envelope is rebuilt
         from the published CIC-IDS2017 testbed topology and the per-class
         attack profile, so it is representative rather than arbitrary -- but
         it is reconstructed, it never reaches the model, and the UI labels it
         as such.

Composition
-----------
Sampling targets a realistic enterprise mix (~8% hostile by default) rather
than the balanced split used for training. A SOC dashboard that sees 50%
attacks is not showing anything a real analyst would recognise.

Usage
-----
    python scripts/generate_demo_traffic.py
    python scripts/generate_demo_traffic.py --rows 12000 --attack-rate 0.12
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("generate_demo_traffic")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "interim" / "clean.parquet"
DEFAULT_OUT = PROJECT_ROOT / "samples" / "live_traffic_stream.csv"

LABEL_COLUMNS = ["label_raw", "label_class"]

# Relative weights for how the hostile share is divided between attack classes.
# Reconnaissance and volumetric floods dominate real perimeter telemetry;
# infiltration is rare by construction (36 flows exist in the whole corpus).
ATTACK_MIX: dict[str, float] = {
    "DoS": 0.30,
    "DDoS": 0.26,
    "PortScan": 0.18,
    "BruteForce": 0.13,
    "WebAttack": 0.08,
    "Botnet": 0.04,
    "Infiltration": 0.01,
}

# CIC-IDS2017 testbed topology, as published with the dataset.
VICTIM_SUBNET = "192.168.10"
WEB_SERVER = "192.168.10.50"
ATTACKER = "172.16.0.1"
C2_SERVER = "205.174.165.73"

# Workstations that generate the benign background traffic.
WORKSTATIONS = [f"{VICTIM_SUBNET}.{h}" for h in (3, 5, 8, 9, 12, 14, 15, 16, 17, 19, 25, 51)]

# Public services the workstations talk to during a normal working day.
EXTERNAL_HOSTS = [
    "142.250.67.14", "13.107.42.14", "104.16.132.229", "151.101.65.69",
    "52.96.165.34", "20.42.65.92", "185.199.108.153", "199.232.69.194",
]

# Benign destination ports, chosen per-flow from the service the destination
# plausibly runs and the transport the flow actually used. A single flat pool
# produced obvious nonsense -- SMB to a Google address, RDP over UDP -- which is
# the first thing a security audience notices in a packet log.
BENIGN_PORTS: dict[tuple[str, str], list[int]] = {
    ("external", "TCP"): [443, 443, 443, 443, 80, 80, 993],
    ("external", "UDP"): [443, 53, 53, 123],          # QUIC, DNS, NTP
    ("internal", "TCP"): [443, 80, 445, 3389, 22, 139],
    ("internal", "UDP"): [53, 137, 123],              # DNS, NetBIOS name service, NTP
}

# Per-class envelope profile: (source pool, destination pool, port pool).
# "sweep" ports are drawn per-flow to reproduce a scanner walking a port range;
# "service" defers to BENIGN_PORTS above.
ENVELOPE_PROFILES: dict[str, dict[str, object]] = {
    "BENIGN": {"src": WORKSTATIONS, "dst": EXTERNAL_HOSTS + [WEB_SERVER], "ports": "service"},
    "DoS": {"src": [ATTACKER], "dst": [WEB_SERVER], "ports": [80]},
    "DDoS": {"src": [ATTACKER], "dst": [WEB_SERVER], "ports": [80]},
    "PortScan": {"src": [ATTACKER], "dst": [WEB_SERVER], "ports": "sweep"},
    "BruteForce": {"src": [ATTACKER], "dst": [WEB_SERVER], "ports": [21, 22]},
    "WebAttack": {"src": [ATTACKER], "dst": [WEB_SERVER], "ports": [80]},
    "Botnet": {"src": WORKSTATIONS[:6], "dst": [C2_SERVER], "ports": [8080]},
    "Infiltration": {"src": [f"{VICTIM_SUBNET}.8", f"{VICTIM_SUBNET}.25"], "dst": [C2_SERVER], "ports": [444, 8080]},
}

# IANA protocol numbers as they appear in the Protocol feature.
PROTOCOL_NAMES = {0: "HOPOPT", 6: "TCP", 17: "UDP"}


def build_targets(rows: int, attack_rate: float, available: dict[str, int]) -> dict[str, int]:
    """
    Decide how many flows to draw from each class.

    Requests are clipped to what the corpus actually holds, and any shortfall
    in the hostile classes is absorbed by BENIGN so the row count is honoured.
    """
    n_attack = int(round(rows * attack_rate))
    targets: dict[str, int] = {}

    total_weight = sum(w for c, w in ATTACK_MIX.items() if available.get(c, 0) > 0)
    for cls, weight in ATTACK_MIX.items():
        if available.get(cls, 0) == 0:
            continue
        want = int(round(n_attack * weight / total_weight))
        targets[cls] = min(want, available[cls])

    targets["BENIGN"] = min(rows - sum(targets.values()), available.get("BENIGN", 0))
    return {c: n for c, n in targets.items() if n > 0}


def sample_indices(labels: pd.Series, targets: dict[str, int], rng: np.random.Generator) -> np.ndarray:
    """Draw the requested number of row positions per class, without replacement."""
    picked = []
    for cls, n in targets.items():
        pool = np.flatnonzero((labels == cls).to_numpy())
        picked.append(rng.choice(pool, size=n, replace=False))
    return np.sort(np.concatenate(picked))


def read_rows(source: Path, positions: np.ndarray, columns: list[str]) -> pd.DataFrame:
    """
    Materialize the sampled rows one row group at a time.

    The corpus is 2.2M rows; loading it whole to then discard 99.7% of it would
    cost roughly a gigabyte of RAM for no benefit.
    """
    pf = pq.ParquetFile(source)
    frames = []
    offset = 0

    for group in range(pf.metadata.num_row_groups):
        size = pf.metadata.row_group(group).num_rows
        in_group = positions[(positions >= offset) & (positions < offset + size)] - offset
        if in_group.size:
            table = pf.read_row_group(group, columns=columns)
            frames.append(table.to_pandas().iloc[in_group])
        offset += size

    return pd.concat(frames, ignore_index=True)


def add_display_envelope(df: pd.DataFrame, labels: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    """
    Prepend the reconstructed addressing a packet-log view needs.

    These four columns are presentation only. They are written ahead of the
    feature block so the file reads like a flow log, and the inference path
    ignores them entirely -- the champion consumes named features and never
    sees a positional column.
    """
    protocol = df["Protocol"] if "Protocol" in df.columns else pd.Series(6, index=df.index)
    names = protocol.round().astype(int).map(PROTOCOL_NAMES).fillna("OTHER")

    src, dst, port = [], [], []

    for cls, transport in zip(labels, names):
        profile = ENVELOPE_PROFILES.get(cls, ENVELOPE_PROFILES["BENIGN"])
        src.append(rng.choice(profile["src"]))
        destination = str(rng.choice(profile["dst"]))
        dst.append(destination)

        ports = profile["ports"]
        if ports == "sweep":
            # A scanner walks the low port range one destination at a time; a
            # fixed port would make PortScan indistinguishable from a normal
            # session in the log view, which is the signal an analyst reads first.
            port.append(int(rng.integers(1, 1024)))
        elif ports == "service":
            scope = "internal" if destination.startswith(VICTIM_SUBNET) else "external"
            # Anything that is not TCP or UDP falls back to the TCP service set
            # rather than inventing a port profile for a rare transport.
            pool = BENIGN_PORTS.get((scope, transport)) or BENIGN_PORTS[(scope, "TCP")]
            port.append(int(rng.choice(pool)))
        else:
            port.append(int(rng.choice(ports)))

    envelope = pd.DataFrame(
        {"src_ip": src, "dst_ip": dst, "dst_port": port, "protocol_name": names.to_numpy()},
        index=df.index,
    )
    return pd.concat([envelope, df], axis=1)


def generate(source: Path, out: Path, rows: int, attack_rate: float, seed: int) -> pd.DataFrame:
    """Sample the corpus, attach the display envelope, and write the stream CSV."""
    if not source.exists():
        raise FileNotFoundError(
            f"Source corpus not found: {source}\n"
            "Run the pipeline first (`dvc repro prepare_data`) to produce it."
        )

    rng = np.random.default_rng(seed)

    logger.info(f"Reading class labels from {source.name}...")
    labels = pq.ParquetFile(source).read(columns=["label_class"]).to_pandas()["label_class"]
    available = labels.value_counts().to_dict()

    targets = build_targets(rows, attack_rate, available)
    logger.info("Sampling plan: " + ", ".join(f"{c}={n:,}" for c, n in sorted(targets.items())))

    positions = sample_indices(labels, targets, rng)

    schema = pq.ParquetFile(source).schema_arrow.names
    feature_columns = [c for c in schema if c not in LABEL_COLUMNS]

    logger.info(f"Materializing {len(positions):,} flows across {len(feature_columns)} features...")
    df = read_rows(source, positions, feature_columns + ["label_class"])

    # Shuffle so the replay interleaves attacks with background traffic instead
    # of playing each class as a contiguous block.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sampled_labels = df.pop("label_class")
    df = add_display_envelope(df, sampled_labels, rng)
    df["Label"] = sampled_labels.to_numpy()

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    hostile = int((df["Label"] != "BENIGN").sum())
    logger.info(f"Wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  {len(df):,} flows | {hostile:,} hostile ({hostile / len(df):.1%}) | {len(df.columns)} columns")
    logger.info("  Class mix: " + ", ".join(f"{c}={n:,}" for c, n in df["Label"].value_counts().items()))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Cleaned corpus to sample from.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Destination CSV.")
    parser.add_argument("--rows", type=int, default=6000, help="Total flows to generate.")
    parser.add_argument("--attack-rate", type=float, default=0.08, help="Hostile share, 0-1.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    args = parser.parse_args()

    if not 0.0 <= args.attack_rate <= 1.0:
        parser.error("--attack-rate must be between 0 and 1.")

    try:
        generate(args.source, args.out, args.rows, args.attack_rate, args.seed)
    except Exception as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
