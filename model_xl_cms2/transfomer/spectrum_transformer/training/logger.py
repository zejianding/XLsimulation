"""V20 training logger."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "training_log.txt"
        self.log_path.write_text("", encoding="utf-8")
        self.history_path = self.output_dir / "training_history.csv"
        self._history_fieldnames = None

    def log(self, message, also_print=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if also_print:
            print(message)

    def console(self, message=""):
        print(message)

    def log_kv_block(self, title, mapping, also_print=False):
        self.log(title, also_print=also_print)
        for key, value in mapping.items():
            self.log(f"  - {key}: {value}", also_print=also_print)

    def log_history_row(self, row):
        if self._history_fieldnames is None:
            self._history_fieldnames = list(row.keys())
            with self.history_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._history_fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        with self.history_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._history_fieldnames)
            writer.writerow(row)
