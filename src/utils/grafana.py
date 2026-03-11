"""
Grafana Cloud metrics push utility.

Credentials are read from environment variables so secrets are never
hardcoded in source:

    GRAFANA_URL   – Influx-compatible write endpoint
    GRAFANA_USER  – Grafana numeric user ID
    GRAFANA_TOKEN – Grafana API token with MetricsPublisher role
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

def push_metrics_to_grafana(row: dict) -> int | None:
    url = os.environ.get("GRAFANA_URL")
    user = os.environ.get("GRAFANA_USER")
    token = os.environ.get("GRAFANA_TOKEN")

    if not all([url, user, token]):
        print(
            "Warning: GRAFANA_URL / GRAFANA_USER / GRAFANA_TOKEN env vars not set "
            "– Grafana push skipped."
        )
        return None

    metrics = {
        "traffic_classical_count":     row.get("classical_count"),
        "traffic_quantum_count":       row.get("quantum_count"),
        "traffic_classical_density":   row.get("classical_density"),
        "traffic_quantum_density":     row.get("quantum_density"),
        "traffic_error":               row.get("error"),
        "traffic_relative_error_pct":  row.get("relative_error_pct"),
        "traffic_density_A":           row.get("density_A"),
        "traffic_density_B":           row.get("density_B"),
        "traffic_num_detections":      row.get("num_detections"),
        "traffic_quantum_exec_ms":     row.get("quantum_execution_time_ms"),
        "traffic_count_agreement":     1 if str(row.get("count_agreement")) == "True" else 0,
        "traffic_theoretical_speedup": row.get("theoretical_speedup"),
    }

    fields = []
    for name, value in metrics.items():
        if value in (None, ""):
            continue
        try:
            fields.append(f"{name}={float(value)}")
        except (ValueError, TypeError):
            continue

    if not fields:
        return None

    line = f'quantum_traffic {",".join(fields)} {int(time.time() * 1e9)}'

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "text/plain"},
            data=line,
            auth=(user, token),
            timeout=5,
        )
        return response.status_code
    except requests.RequestException as exc:
        print(f"Warning: Grafana push failed – {exc}")
        return None
