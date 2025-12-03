import csv
import re
import time
import requests
import argparse
from typing import Optional, List, Dict
from requests.exceptions import ConnectionError, ReadTimeout
from tqdm import tqdm

# 默认请求头
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # 如果遇到需要登录才能查看的帖子，请在这里填入 cookie
    "cookie": "" 
}

# 用来收集所有评论行
rows = []

def safe_get(url: str, headers: dict, timeout: int = 15, retries: int = 3, backoff: float = 3.0) -> Optional[requests.Response]:
    """Requests wrapper with retries to reduce random connection resets."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return requests.get(url=url, headers=headers, timeout=timeout)
        except (ConnectionError, ReadTimeout) as exc:
            last_exc = exc
            print(f"请求失败 {attempt}/{retries}: {exc}. {backoff}s 后重试...")
            time.sleep(backoff)
    if last_exc:
        print(f"多次重试仍失败: {last_exc}")
    return None

def clean_html(raw_html: str) -> str:
    """简单的 HTML 清理，去除标签"""
    if not raw_html:
        return ""
    # 将 <br> 替换为换行
    text = re.sub(r'<br\s*/?>', '\n', raw_html)
    # 去除其他 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def get_max_page(html_content: str) -> int:
    """从页面 HTML 中提取最大页数"""
    # 匹配分页输入框：<input type="number" class="page_input" ... max="5" ...>
    match = re.search(r'class="page_input"[^>]*max="(\d+)"', html_content)
    if match:
        return int(match.group(1))
    
    # 如果没有输入框，尝试查找分页链接
    # 这里的逻辑是如果没有 max 属性，可能是只有 1 页
    return 1

def parse_comments(html_content: str, topic_id: str) -> List[Dict]:
    """使用正则解析 HTML 中的评论"""
    comments = []
    
    # V2EX 的评论通常包裹在 <div id="r_xxxx" class="cell"> ... </div> 中
    # 我们先分割出每个评论块
    # 注意：这种正则解析在 HTML 结构变化时可能会失效
    
    # 查找所有评论块的起始位置
    reply_blocks = re.findall(r'(<div id="r_\d+" class="cell">[\s\S]*?)(?=<div id="r_\d+" class="cell">|<div class="box">|$)', html_content)
    
    for block in reply_blocks:
        try:
            # 提取回复 ID
            rpid_match = re.search(r'id="r_(\d+)"', block)
            rpid = rpid_match.group(1) if rpid_match else ""
            
            # 提取用户名
            # <a href="/member/username" class="dark">username</a>
            user_match = re.search(r'<a href="/member/([^"]+)" class="dark">', block)
            username = user_match.group(1) if user_match else "unknown"
            
            # 提取内容
            # <div class="reply_content">content</div>
            content_match = re.search(r'<div class="reply_content">([\s\S]*?)</div>', block)
            content_html = content_match.group(1) if content_match else ""
            content = clean_html(content_html)
            
            # 提取时间
            # <span class="ago" title="2023-12-01 10:00:00">1 小时前</span>
            # 优先取 title 中的绝对时间，如果没有则取文本
            time_match = re.search(r'<span class="ago"[^>]*title="([^"]+)"[^>]*>', block)
            if time_match:
                ctime = time_match.group(1)
            else:
                # 备选：直接取 span 内容
                time_match_alt = re.search(r'<span class="ago"[^>]*>([\s\S]*?)</span>', block)
                ctime = time_match_alt.group(1).strip() if time_match_alt else ""

            comments.append({
                "topic_id": topic_id,
                "rpid": rpid,
                "username": username,
                "ctime": ctime,
                "message": content
            })
            
        except Exception as e:
            print(f"解析某条评论时出错: {e}")
            continue
            
    return comments

def parse_topic_info(html_content: str, topic_id: str) -> Optional[Dict]:
    """解析主帖信息（标题、内容、楼主、时间等）"""
    try:
        # 1. 提取标题 (<h1>Title</h1>)
        title_match = re.search(r'<h1>([\s\S]*?)</h1>', html_content)
        title = clean_html(title_match.group(1)) if title_match else "No Title"

        # 2. 提取楼主 (header 中的 <a href="/member/xxx">)
        # 通常在标题上方的 header div 里
        # <small class="gray"><a href="/member/user">user</a> · ...</small>
        # 或者直接找 header 里的第一个 member 链接
        header_match = re.search(r'<div class="header">([\s\S]*?)</div>', html_content)
        if header_match:
            header_content = header_match.group(1)
            user_match = re.search(r'<a href="/member/([^"]+)"', header_content)
            author = user_match.group(1) if user_match else "unknown"
            
            # 3. 提取时间
            # <span title="2024-01-01 12:00:00">...</span>
            time_match = re.search(r'<span[^>]*title="([^"]+)"[^>]*>', header_content)
            ctime = time_match.group(1) if time_match else ""
        else:
            author = "unknown"
            ctime = ""

        # 4. 提取内容
        # <div class="topic_content">...</div>
        # 由于正则匹配嵌套 div 很困难，我们尝试匹配到常见的结束标记
        # V2EX 内容后通常是 topic_buttons 或 sep10 或 subtle (附言)
        content_match = re.search(r'<div class="topic_content"[^>]*>([\s\S]*?)(?:<div class="sep10">|<div class="topic_buttons">|<div class="subtle">|</div>\s*<div class="box">)', html_content)
        if content_match:
            content_html = content_match.group(1)
            # 如果内容里包含了 markdown_body 的结束标签，可能残留 </div>，clean_html 会处理掉标签，但内容可能需要截断
            # 这里简单处理，只做 clean_html
            content = clean_html(content_html)
        else:
            content = ""

        # 5. 提取评论数 (元数据)
        # <span class="gray">86 回复</span>
        # 这通常在 box 的 cell 里，或者根据总楼层数估算
        # 我们可以直接返回 None，因为会在后面统计实际爬到的数量
        
        return {
            "topic_id": topic_id,
            "rpid": "topic", # 标识为主帖
            "username": author,
            "ctime": ctime,
            "message": f"【标题】{title}\n【内容】\n{content}"
        }

    except Exception as e:
        print(f"解析主帖信息失败: {e}")
        return None

def start_crawler(topic_id: str):
    base_url = f"https://www.v2ex.com/t/{topic_id}"
    
    # 1. 先请求第一页，获取总页数
    print(f"正在获取主题 {topic_id} 信息...")
    first_url = f"{base_url}?p=1"
    first_resp = safe_get(first_url, headers)
    
    if not first_resp or first_resp.status_code != 200:
        print("获取主题失败，请检查 ID 是否正确，或是否需要 Cookie (如果是登录可见节点)。")
        return

    html_content = first_resp.text
    
    # 解析并保存主帖信息
    topic_info = parse_topic_info(html_content, topic_id)
    global rows
    rows = []
    
    if topic_info:
        rows.append(topic_info)
        print(f"成功获取主帖: {topic_info['message'].splitlines()[0]}")  # 打印标题行
    
    total_pages = get_max_page(html_content)
    print(f"检测到共有 {total_pages} 页评论")
    
    # 2. 遍历所有页面
    
    for page in tqdm(range(1, total_pages + 1), desc="爬取进度"):
        url = f"{base_url}?p={page}"
        
        # 如果是第一页，直接使用刚才获取的内容，避免重复请求
        if page == 1:
            page_html = html_content
        else:
            resp = safe_get(url, headers)
            if not resp:
                print(f"第 {page} 页获取失败")
                continue
            page_html = resp.text
            
        page_comments = parse_comments(page_html, topic_id)
        rows.extend(page_comments)
        
        # 稍微停顿，礼貌爬取
        time.sleep(0.8)

    print(f"爬取结束，共获取 {len(rows)} 条评论")

if __name__ == "__main__":
    # 获取用户输入
    raw_input = input("请输入 V2EX 主题 ID (例如网址 https://www.v2ex.com/t/1097263 中的 1097263): ").strip()
    
    # 自动提取 ID
    match = re.search(r'(\d+)', raw_input)
    if match:
        topic_id = match.group(1)
    else:
        topic_id = raw_input

    if not topic_id:
        print("无效的 ID")
        exit(1)

    # 提示是否输入 Cookie
    print("\n提示: 某些节点 (如 /go/job) 需要登录才能查看。")
    use_cookie = input("是否需要输入 Cookie? (y/n, 默认 n): ").strip().lower()
    if use_cookie == 'y':
        cookie_str = input("请输入 cookie: ").strip()
        if cookie_str:
            headers['cookie'] = cookie_str

    # 开始爬取
    start_crawler(topic_id)

    # 写入 CSV
    if rows:
        out_name = f"v2ex_comments_{topic_id}.csv"
        fieldnames = [
            "topic_id", "rpid", "username", "ctime", "message"
        ]
        with open(out_name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"已保存到 {out_name}")
    else:
        print("没有抓到任何评论。")

