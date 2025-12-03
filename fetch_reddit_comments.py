import csv
import datetime
import json
import time
import requests
from typing import List, Dict, Optional

# Reddit 对 User-Agent 比较敏感，设置模拟浏览器
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def get_json(url: str) -> Optional[List]:
    """
    请求 Reddit 的 .json 接口
    Reddit 的帖子 URL 末尾加上 .json 即可获取结构化数据
    """
    # 移除可能存在的 URL 参数
    url = url.split('?')[0]
    # 移除末尾斜杠
    if url.endswith('/'):
        url = url[:-1]
    
    # 拼接 .json
    json_url = url + '.json'
    
    print(f"正在请求: {json_url}")
    try:
        resp = requests.get(json_url, headers=HEADERS, timeout=20)
        if resp.status_code == 429:
            print("错误: 请求过于频繁 (429 Too Many Requests)。请稍后重试。")
            return None
        elif resp.status_code == 403:
            print("错误: 访问被拒绝 (403 Forbidden)。可能是网络问题或 User-Agent 被屏蔽。")
            return None
            
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"请求失败: {e}")
        print("提示: 请确保您的网络环境可以访问 Reddit。")
        return None

def parse_timestamp(ts: float) -> str:
    """将 Unix 时间戳转换为易读格式"""
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def extract_comments(comment_data: Dict, post_id: str, parent_id: str = "root") -> List[Dict]:
    """递归提取评论树"""
    results = []
    
    # Reddit 的 JSON 结构中，listing 有时是 dict 有时是 list
    if isinstance(comment_data, dict):
        children = comment_data.get('data', {}).get('children', [])
    elif isinstance(comment_data, list):
        children = comment_data
    else:
        return []

    for child in children:
        kind = child.get('kind')
        data = child.get('data', {})
        
        # kind='t1' 代表这是一条评论
        if kind == 't1':
            author = data.get('author', '[deleted]')
            body = data.get('body', '[deleted]')
            ups = data.get('ups', 0)
            created_utc = data.get('created_utc', 0)
            comment_id = data.get('id')
            # author_flair_text 有时包含用户设置的身份或地点信息
            author_flair = data.get('author_flair_text') 
            
            item = {
                "post_id": post_id,
                "type": "comment",
                "id": comment_id,
                "parent_id": parent_id,
                "author": author,
                "author_flair": author_flair if author_flair else "",
                "created_time": parse_timestamp(created_utc),
                "ups": ups,
                "content": body
            }
            results.append(item)
            
            # 递归处理子回复 (replies)
            replies = data.get('replies')
            if replies and isinstance(replies, dict):
                results.extend(extract_comments(replies, post_id, parent_id=comment_id))
        
        # kind='more' 代表此处折叠了更多评论，API需要额外请求才能展开
        # 为了保持脚本轻量，这里不做深度展开
        elif kind == 'more':
            pass
            
    return results

def main():
    print("=== Reddit 帖子爬取工具 ===")
    print("示例链接: https://www.reddit.com/r/ChatGPT/comments/1gzxxxxx/title_here/")
    url = input("请输入 Reddit 帖子链接: ").strip()
    
    if not url:
        print("链接不能为空")
        return

    # 获取数据
    data = get_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        print("无法获取有效数据，请检查链接是否正确。")
        return

    # Reddit API 返回包含两个元素的列表：
    # data[0] 是主帖信息 (Listing t3)
    # data[1] 是评论列表 (Listing t1)
    
    rows = []
    post_id = "unknown"
    
    # 1. 解析主帖
    try:
        post_children = data[0]['data']['children']
        if post_children:
            post_data = post_children[0]['data']
            post_id = post_data.get('id')
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '') # 主帖正文
            author = post_data.get('author', '')
            ups = post_data.get('ups', 0)
            num_comments = post_data.get('num_comments', 0)
            created_utc = post_data.get('created_utc', 0)
            permalink = post_data.get('permalink', '')
            
            full_content = f"【标题】{title}\n【内容】\n{selftext}"
            
            rows.append({
                "post_id": post_id,
                "type": "post",
                "id": post_id,
                "parent_id": "",
                "author": author,
                "author_flair": post_data.get('author_flair_text', ''),
                "created_time": parse_timestamp(created_utc),
                "ups": ups,
                "content": full_content
            })
            
            print(f"成功获取主帖: {title[:30]}...")
            print(f"官方统计评论总数: {num_comments}")
    except Exception as e:
        print(f"解析主帖信息出错: {e}")
        return

    # 2. 解析评论
    try:
        comments_listing = data[1]
        comments = extract_comments(comments_listing, post_id, parent_id=post_id)
        rows.extend(comments)
        print(f"本次抓取到评论数 (含子评论): {len(comments)}")
    except Exception as e:
        print(f"解析评论出错: {e}")

    # 3. 保存文件
    if rows:
        filename = f"reddit_comments_{post_id}.csv"
        headers = [
            "post_id", "type", "id", "parent_id", 
            "author", "author_flair", "created_time", 
            "ups", "content"
        ]
        
        try:
            with open(filename, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"\n✅ 数据已保存至: {filename}")
        except IOError as e:
            print(f"文件保存失败: {e}")

if __name__ == "__main__":
    main()

