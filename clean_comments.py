"""
Normalize and clean comment datasets collected from Bilibili and Zhihu.

Usage:
    python clean_comments.py

Outputs:
    - cleaned_comments.csv  : unified, cleaned comment-level table
    - thread_summary.csv    : per-thread stats for quick validation
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parent

TEXT_BREAK_TAGS = re.compile(r"</?(p|br\s*/?)>", flags=re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Remove simple HTML tags, unescape entities, trim whitespace."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = TEXT_BREAK_TAGS.sub("\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def load_bilibili(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["platform"] = "bilibili"
    df["thread_id"] = df["bv_id"].astype(str)
    df["comment_id"] = df["rpid"].astype(str)
    df["parent_id"] = df["parent_rpid"].replace({0: pd.NA}).astype("string")
    df["user_id"] = df["uid"].astype("string")
    df["user_name"] = df["uname"].astype(str).str.strip()
    df["user_location"] = pd.NA
    df["user_level"] = df["user_level"].fillna(0).astype(int)
    df["like_count"] = df["like"].fillna(0).astype(int)
    df["created_at"] = pd.to_datetime(df["ctime"], unit="s", errors="coerce")
    df["text"] = df["message"].apply(clean_text)
    df["source_file"] = path.name
    keep_cols = [
        "platform",
        "thread_id",
        "comment_id",
        "parent_id",
        "level",
        "user_id",
        "user_name",
        "user_location",
        "user_level",
        "like_count",
        "created_at",
        "text",
        "source_file",
    ]
    return df[keep_cols]


def load_zhihu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["platform"] = "zhihu"
    df["thread_id"] = df["answer_id"].astype(str)
    df["comment_id"] = df["comment_id"].astype(str)
    df["parent_id"] = df["parent_id"].replace({0: pd.NA}).astype("string")
    df["user_id"] = pd.NA  # 匿名用户，无稳定 ID
    df["user_name"] = df["author_name"].astype(str).str.strip()
    df["user_location"] = df.get("ip_location")
    df["user_level"] = pd.NA
    df["like_count"] = df["like_count"].fillna(0).astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["text"] = df["content"].apply(clean_text)
    df["source_file"] = path.name
    keep_cols = [
        "platform",
        "thread_id",
        "comment_id",
        "parent_id",
        "level",
        "user_id",
        "user_name",
        "user_location",
        "user_level",
        "like_count",
        "created_at",
        "text",
        "source_file",
    ]
    return df[keep_cols]


def collect_files(pattern: str) -> List[Path]:
    return sorted(ROOT.glob(pattern))


def main() -> None:
    bilibili_files = collect_files("bilibili_comments_*.csv")
    zhihu_files = collect_files("zhihu_comments_*.csv")

    frames: List[pd.DataFrame] = []
    for file_path in bilibili_files:
        frames.append(load_bilibili(file_path))
    for file_path in zhihu_files:
        frames.append(load_zhihu(file_path))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text"])
    combined["text_length"] = combined["text"].str.len()
    combined = combined.sort_values("created_at").drop_duplicates(
        subset=["platform", "thread_id", "comment_id"], keep="first"
    )

    combined.to_csv(ROOT / "cleaned_comments.csv", index=False, encoding="utf-8-sig")

    summary = (
        combined.groupby(["platform", "thread_id"])
        .agg(
            comments=("comment_id", "count"),
            root_comments=("parent_id", lambda s: s.isna().sum()),
            avg_len=("text_length", "mean"),
            earliest=("created_at", "min"),
            latest=("created_at", "max"),
        )
        .reset_index()
        .sort_values(["platform", "comments"], ascending=[True, False])
    )
    summary.to_csv(ROOT / "thread_summary.csv", index=False, encoding="utf-8-sig")

    print(
        f"Cleaned {len(combined)} comments from "
        f"{len(bilibili_files)} Bilibili and {len(zhihu_files)} Zhihu threads."
    )


if __name__ == "__main__":
    main()

