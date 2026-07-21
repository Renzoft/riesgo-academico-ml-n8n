import argparse
import json
import sqlite3
from pathlib import Path

from monitoring import generate_monitoring_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["production", "manual", "system_test"],
        default="production",
    )
    parser.add_argument(
        "--database", type=Path, default=Path("app_data/riesgo_academico.db")
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/monitoring/latest.json"),
    )
    args = parser.parse_args()
    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    report = generate_monitoring_report(
        connection, args.dataset, source=args.source
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
