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
import re
import html
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


def clean_html(raw_html: str) -> str:
    """简单的 HTML 清理，去除标签"""
    if not raw_html:
        return ""
    # 将 <br> 替换为换行
    text = re.sub(r'<br\s*/?>', '\n', raw_html)
    # 去除其他 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text.strip())


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


def fetch_answer_detail(answer_id: str, headers: Dict[str, str], cookies: Dict[str, str]) -> Optional[ZhihuComment]:
    """获取回答主帖详情（包括问题标题和回答内容）"""
    url = f"https://www.zhihu.com/api/v4/answers/{answer_id}?include=content,author,voteup_count,created_time,question"
    data = safe_get(url, headers, cookies)
    if not data:
        print(f"无法获取回答 {answer_id} 的主帖详情，将只抓取评论。")
        return None

    try:
        question = data.get("question", {})
        question_title = question.get("title", "")
        
        content_html = data.get("content", "")
        content_text = clean_html(content_html)
        
        full_content = f"【问题】{question_title}\n【回答】\n{content_text}"
        
        author = data.get("author", {})
        gender = parse_gender(author.get("gender", -1))
        author_name = author.get("name") or author.get("headline") or "匿名用户"
        
        return ZhihuComment(
            answer_id=answer_id,
            level=0,  # 0 表示主帖
            comment_id=0,
            parent_id=0,
            author_name=author_name,
            gender=gender,
            ip_location="", # API 通常不返回主帖 IP
            like_count=data.get("voteup_count", 0),
            created_at=unix_to_str(data.get("created_time", 0)),
            content=full_content,
        )
    except Exception as e:
        print(f"解析主帖详情失败: {e}")
        return None


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

    # 1. 尝试获取主帖信息
    print("正在获取主帖详情...")
    main_post = fetch_answer_detail(answer_id, headers, cookies)
    if main_post:
        results.append(main_post)
        print(f"成功获取主帖: {main_post.author_name} 的回答")

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

    raw_input = prompt_if_empty("", "请输入 answer 号：")
    
    # 自动清理输入，只保留数字 ID
    # 1. 去掉 URL 参数 (? 及其后面的内容)
    answer_id = raw_input.split('?')[0]
    # 2. 如果用户输入的是完整网址，尝试提取最后一部分
    if '/' in answer_id:
        answer_id = answer_id.split('/')[-1]

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

