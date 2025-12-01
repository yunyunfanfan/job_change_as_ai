"""
Simple utility to scrape Zhihu answer comments (root + child) into CSV.

Steps to use:
1. Fill ANSWER_ID, USER_AGENT, COOKIE below (or pass via CLI prompts).
2. Run: python fetch_zhihu_comments.py
3. The script writes comments to zhihu_comments_{answer_id}.csv

Reference workflow adapted from:
https://blog.csdn.net/QiuQiRuQin/article/details/143433636
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import requests
from requests.exceptions import ConnectionError, ReadTimeout
from tqdm import tqdm

API_TEMPLATE = (
    "https://www.zhihu.com/api/v4/answers/{answer_id}/root_comments"
    "?order=normal&limit={limit}&offset={offset}&status=open"
)


@dataclass
class ZhihuComment:
    answer_id: str
    level: int  # 1 root, 2 child
    comment_id: int
    parent_id: int
    author_name: str
    gender: str
    ip_location: str
    like_count: int
    created_at: str
    content: str


def prompt_if_empty(value: str, prompt: str) -> str:
    if value:
        return value
    return input(prompt).strip()


def unix_to_str(timestamp: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def parse_gender(tag: int) -> str:
    return {1: "男", 0: "女"}.get(tag, "未知")


def safe_get(url: str, headers: Dict[str, str], cookies: Dict[str, str]) -> Optional[dict]:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, cookies=cookies, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (ConnectionError, ReadTimeout, requests.HTTPError) as exc:
            wait = 2 * (attempt + 1)
            print(f"请求失败({attempt + 1}/3): {exc}; {wait}s 后重试...")
            time.sleep(wait)
    print("连续失败 3 次，放弃该请求。")
    return None


def extract_comment(answer_id: str, comment: dict, level: int, parent_id: int) -> ZhihuComment:
    author = comment.get("author", {}).get("member", {})
    gender = parse_gender(author.get("gender", -1))
    author_name = author.get("name") or author.get("headline") or "匿名用户"
    return ZhihuComment(
        answer_id=answer_id,
        level=level,
        comment_id=comment.get("id"),
        parent_id=parent_id,
        author_name=author_name,
        gender=gender,
        ip_location=comment.get("address_text", ""),
        like_count=comment.get("vote_count", 0),
        created_at=unix_to_str(comment.get("created_time", 0)),
        content=comment.get("content", "").strip(),
    )


def crawl_comments(
    answer_id: str,
    user_agent: str,
    cookie: str,
    max_pages: int = 50,
    page_size: int = 20,
    sleep_seconds: float = 1.0,
) -> List[ZhihuComment]:
    headers = {"User-Agent": user_agent}
    cookies = {"cookie": cookie}
    offset = 0
    page = 1
    results: List[ZhihuComment] = []

    with tqdm(total=max_pages, desc="知乎评论页", unit="page") as progress:
        while page <= max_pages:
            url = API_TEMPLATE.format(answer_id=answer_id, limit=page_size, offset=offset)
            data = safe_get(url, headers, cookies)
            if not data:
                break

            comments = data.get("data") or []
            if not comments:
                break

            for comment in comments:
                root = extract_comment(answer_id, comment, level=1, parent_id=0)
                results.append(root)

                # child comments
                for child in comment.get("child_comments") or []:
                    child_comment = extract_comment(
                        answer_id, child, level=2, parent_id=root.comment_id
                    )
                    results.append(child_comment)

            progress.update(1)
            offset += page_size
            page += 1
            time.sleep(sleep_seconds)
            if not data.get("paging", {}).get("is_end") and offset < 10000:
                continue
            else:
                break

    return results


def save_to_csv(answer_id: str, comments: List[ZhihuComment]) -> str:
    filename = f"zhihu_comments_{answer_id}.csv"
    with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(asdict(comments[0]).keys()))
        writer.writeheader()
        for comment in comments:
            writer.writerow(asdict(comment))
    return filename


def main() -> None:
    print("=== 知乎评论抓取工具 ===")
    answer_id = prompt_if_empty("", "请输入 answer 号：")
    user_agent = prompt_if_empty("", "请输入 User-Agent：")
    cookie = prompt_if_empty("", "请输入 cookie：")

    comments = crawl_comments(answer_id, user_agent, cookie)
    if not comments:
        print("未抓取到任何评论，请检查参数或登录状态。")
        sys.exit(1)

    output_path = save_to_csv(answer_id, comments)
    print(f"完成！共抓取 {len(comments)} 条评论，已保存到 {output_path}")


if __name__ == "__main__":
    main()

