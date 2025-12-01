"""
Lightweight statistics over raw comment CSV files without extra dependencies.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def iter_rows(path: Path):
    with path.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def summarize() -> None:
    platform_counts = Counter()
    thread_counts = defaultdict(int)
    thread_dates = defaultdict(lambda: [None, None])

    for csv_path in ROOT.glob("bilibili_comments_*.csv"):
        for row in iter_rows(csv_path):
            platform_counts["bilibili"] += 1
            thread = row["bv_id"]
            key = ("bilibili", thread)
            thread_counts[key] += 1
            ts = row.get("ctime")
            if ts:
                dt = datetime.fromtimestamp(int(ts))
                start, end = thread_dates[key]
                if not start or dt < start:
                    start = dt
                if not end or dt > end:
                    end = dt
                thread_dates[key] = [start, end]

    for csv_path in ROOT.glob("zhihu_comments_*.csv"):
        for row in iter_rows(csv_path):
            platform_counts["zhihu"] += 1
            thread = row["answer_id"]
            key = ("zhihu", thread)
            thread_counts[key] += 1
            created = row.get("created_at")
            if created:
                try:
                    dt = datetime.fromisoformat(created)
                except ValueError:
                    dt = None
                if dt:
                    start, end = thread_dates[key]
                    if not start or dt < start:
                        start = dt
                    if not end or dt > end:
                        end = dt
                    thread_dates[key] = [start, end]

    print("Platform counts:", platform_counts)
    print("Thread breakdown:")
    for key in sorted(thread_counts.keys()):
        start, end = thread_dates[key]
        print(
            key[0],
            key[1],
            thread_counts[key],
            start.isoformat() if start else "-",
            end.isoformat() if end else "-",
        )


if __name__ == "__main__":
    summarize()

