"""
<<<<<<< HEAD
Normalize and clean comment datasets collected from Bilibili, Zhihu, Reddit, and V2EX.
=======
Normalize and clean comment datasets collected from Bilibili and Zhihu.
>>>>>>> upstream/main

Usage:
    python clean_comments.py

Outputs:
<<<<<<< HEAD
    - all_comments_cleaned.csv  : unified, cleaned comment-level table
    - cleaning_summary.csv      : summary stats per platform
=======
    - cleaned_comments.csv  : unified, cleaned comment-level table
    - thread_summary.csv    : per-thread stats for quick validation
>>>>>>> upstream/main
"""
from __future__ import annotations

import html
import re
<<<<<<< HEAD
import pandas as pd
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
TEXT_BREAK_TAGS = re.compile(r"</?(p|br\s*/?)>", flags=re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")

=======
from pathlib import Path
from typing import Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parent

TEXT_BREAK_TAGS = re.compile(r"</?(p|br\s*/?)>", flags=re.IGNORECASE)
WHITESPACE = re.compile(r"\s+")


>>>>>>> upstream/main
def clean_text(value: object) -> str:
    """Remove simple HTML tags, unescape entities, trim whitespace."""
    if pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = TEXT_BREAK_TAGS.sub("\n", text)
<<<<<<< HEAD
    text = re.sub(r'<[^>]+>', '', text) # Remove all other tags
=======
>>>>>>> upstream/main
    text = re.sub(r"\n{2,}", "\n", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()

<<<<<<< HEAD
def load_bilibili(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty: return pd.DataFrame()
    
    # Mapping
    out = pd.DataFrame(index=df.index)  # Initialize with matching index
    out["platform"] = "bilibili"
    out["thread_id"] = df["bv_id"].astype(str)
    out["comment_id"] = df["rpid"].astype(str)
    out["parent_id"] = df["parent_rpid"].fillna(0).astype(str)
    out["username"] = df["uname"].astype(str)
    out["content"] = df["message"].apply(clean_text)
    out["like_count"] = df["like"].fillna(0).astype(int)
    # Bilibili ctime is unix timestamp (seconds)
    out["created_at"] = pd.to_datetime(df["ctime"], unit="s", errors="coerce").dt.tz_localize(None)
    
    return out

def load_zhihu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty: return pd.DataFrame()
    
    out = pd.DataFrame(index=df.index)
    out["platform"] = "zhihu"
    out["thread_id"] = df["answer_id"].astype(str)
    # Zhihu ID logic varies, try to standardise
    if "comment_id" in df.columns:
        out["comment_id"] = df["comment_id"].astype(str)
    else:
        out["comment_id"] = df.index.astype(str) # Fallback
        
    out["parent_id"] = df["parent_id"].fillna(0).astype(str) if "parent_id" in df.columns else "0"
    out["username"] = df["author_name"].astype(str)
    out["content"] = df["content"].apply(clean_text)
    out["like_count"] = df["like_count"].fillna(0).astype(int)
    
    # Zhihu created_at is string "YYYY-MM-DD HH:MM:SS"
    out["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.tz_localize(None)
    
    return out

def load_reddit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty: return pd.DataFrame()
    
    out = pd.DataFrame(index=df.index)
    out["platform"] = "reddit"
    out["thread_id"] = df["post_id"].astype(str)
    out["comment_id"] = df["id"].astype(str)
    out["parent_id"] = df["parent_id"].fillna("root").astype(str)
    out["username"] = df["author"].astype(str)
    # Add flair to content if useful, or keep separate. Here we just keep content.
    out["content"] = df["content"].apply(clean_text)
    out["like_count"] = df["ups"].fillna(0).astype(int)
    
    # Reddit created_time is string "YYYY-MM-DD HH:MM:SS"
    out["created_at"] = pd.to_datetime(df["created_time"], errors="coerce").dt.tz_localize(None)
    
    return out

def load_v2ex(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty: return pd.DataFrame()
    
    out = pd.DataFrame(index=df.index)
    out["platform"] = "v2ex"
    out["thread_id"] = df["topic_id"].astype(str)
    out["comment_id"] = df["rpid"].astype(str)
    # V2EX csv in this project doesn't have parent_id column in fetch script
    out["parent_id"] = "0" 
    out["username"] = df["username"].astype(str)
    out["content"] = df["message"].apply(clean_text)
    
    # V2EX script doesn't fetch like count currently (removed per user request)
    out["like_count"] = 0
    
    # V2EX ctime is usually relative text like "1小时前" or absolute in title
    # Our fetch script extracts text. Parsing relative time is hard without fetch time.
    # We try best effort. If format is standard date, good.
    out["created_at"] = pd.to_datetime(df["ctime"], errors="coerce").dt.tz_localize(None)
    
    return out
=======

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

>>>>>>> upstream/main

def collect_files(pattern: str) -> List[Path]:
    return sorted(ROOT.glob(pattern))

<<<<<<< HEAD
def main():
    print("=== 开始清洗数据 ===")
    all_dfs = []

    # 1. Bilibili
    for p in collect_files("bilibili_comments_*.csv"):
        print(f"Processing Bilibili: {p.name}")
        try:
            all_dfs.append(load_bilibili(p))
        except Exception as e:
            print(f"Error loading {p.name}: {e}")

    # 2. Zhihu
    for p in collect_files("zhihu_comments_*.csv"):
        print(f"Processing Zhihu: {p.name}")
        try:
            all_dfs.append(load_zhihu(p))
        except Exception as e:
            print(f"Error loading {p.name}: {e}")
            
    # 3. Reddit
    for p in collect_files("reddit_comments_*.csv"):
        print(f"Processing Reddit: {p.name}")
        try:
            all_dfs.append(load_reddit(p))
        except Exception as e:
            print(f"Error loading {p.name}: {e}")

    # 4. V2EX
    for p in collect_files("v2ex_comments_*.csv"):
        print(f"Processing V2EX: {p.name}")
        try:
            all_dfs.append(load_v2ex(p))
        except Exception as e:
            print(f"Error loading {p.name}: {e}")

    if not all_dfs:
        print("没有找到任何数据文件。")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Basic Filtering
    combined = combined.dropna(subset=["content"])
    combined = combined[combined["content"].str.len() > 1] # Filter empty/single char
    combined["content_length"] = combined["content"].str.len()
    
    # Sort
    combined = combined.sort_values(["platform", "created_at"], ascending=[True, True])

    output_file = ROOT / "all_comments_cleaned.csv"
    combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print("\n=== 清洗完成 ===")
    print(f"总数据量: {len(combined)}")
    print(combined["platform"].value_counts())
    print(f"已保存至: {output_file}")

if __name__ == "__main__":
    main()
=======

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

>>>>>>> upstream/main
