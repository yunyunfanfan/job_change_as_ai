import math
import time
import requests
import hashlib
import urllib
from urllib.parse import quote
import re
import json
import csv
from typing import Optional
from requests.exceptions import ConnectionError, ReadTimeout
from tqdm import tqdm

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0",
    "cookie": "buvid3=7D332C89-A3FA-C611-75E3-A47933302B3183263infoc; b_nut=1757731283; _uuid=E9E65A34-17E10-4103F-DBAC-96101899A54C283725infoc; enable_web_push=DISABLE; buvid_fp=1556cb2b47d931276c49a5c8566792ae; buvid4=941D5AAA-C348-1157-1E6F-BB7F53BEB62384355-025091310-Qn7dqxKWlDS8PG2wZhi2cw%3D%3D; CURRENT_QUALITY=0; rpdid=|(JR|ku~km)J0J'u~l~~JuJ|m; DedeUserID=294669515; DedeUserID__ckMd5=85a08c17b195b61b; theme-tip-show=SHOWED; theme-avatar-tip-show=SHOWED; home_feed_column=5; bp_t_offset_294669515=1139820625993400320; b_lsid=9F10A96DD_19AD8E0FB1C; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3NjQ4MzQ0MTEsImlhdCI6MTc2NDU3NTE1MSwicGx0IjotMX0.AxjMeeIh8y9dkZ2BGBqq6ZmIKzgVgvrS9baYKfFO6Ug; bili_ticket_expires=1764834351; SESSDATA=a7e3bb90%2C1780127212%2C51096%2Ac1CjCwY_f6D6E01mdOUdas5enZ3b0tpGSrCdC_rfFEpEqbMAlpzrdKY6iSAiytIhc10owSVmdDY2oyUFYwZ1E5dDgtQUFLN2FCVjY4VU1lVkkzQXg4MExKVU80SV9aelVEQW1pclV1SFlVTlpoa0dqdENpTkVZeUVqdjZyZUxZeE9WV1N4YVRub0F3IIEC; bili_jct=f4b075cd2440fafa66b5b3e0c6f46921; sid=8j0ql120; CURRENT_FNVAL=4048; browser_resolution=1910-1286"
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


# 获取 oid
def get_oid(bv_id):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bv_id}"
    oid = requests.get(url=url, headers=headers).json()['data']['aid']
    return oid


# 加密w_rid参数，md5加密算法
def get_w_rid(url):
    MD5 = hashlib.md5()
    MD5.update(url.encode('utf-8'))
    w_rid = MD5.hexdigest()
    return w_rid


def start_crawler(bv_id, pagination_str, is_second, oid, total_count=0):
    # 参数
    mode = 3  # 2=最新评论，3=热门/按时间（看博客写法）
    plat = 1
    type_ = 1
    web_location = 1315875
    seek_rpid = ""
    # 获取当下时间戳
    wts = int(time.time())
    # w_rid参数中所需的固定值
    a = "ea1db124af3c7062474693fa704f4ff8"

    if pagination_str != "":
        pagination_dict = {"offset": pagination_str}
    else:
        pagination_dict = {"offset": ""}

    pagination_json = json.dumps(pagination_dict)
    enc_pagination = urllib.parse.quote(pagination_json)

    # 生成 w_rid
    w_rid_url = (
        f"mode={mode}&oid={oid}&pagination_str={enc_pagination}"
        f"&plat={plat}&seek_rpid=&type={type_}&web_location={web_location}"
        f"&wts={wts}{a}"
    )
    w_rid = get_w_rid(w_rid_url)

    url_first = (
        "https://api.bilibili.com/x/v2/reply/wbi/main"
        f"?oid={oid}&type={type_}&mode={mode}"
        f"&pagination_str={enc_pagination}"
        f"&plat={plat}&seek_rpid={seek_rpid}"
        f"&web_location={web_location}&w_rid={w_rid}&wts={wts}"
    )
    print(f"url_first: {url_first}")

    # 一级评论返回的 json
    first_resp = safe_get(url_first, headers=headers)
    if not first_resp:
        raise RuntimeError("连续多次获取一级评论失败，终止。")
    first_comment = first_resp.json()
    time.sleep(0.5)  # 反爬

    first_replies = first_comment["data"].get("replies") or []

    for reply in tqdm(first_replies, desc=f"{bv_id} 一级评论", leave=False):
        # 一级评论
        rpid = reply["rpid"]
        uid = reply["mid"]
        name = reply["member"]["uname"]
        level = reply["member"]["level_info"]["current_level"]
        context = reply["content"]["message"]
        ctime = reply["ctime"]
        like = reply["like"]

        rows.append(
            {
                "bv_id": bv_id,
                "oid": oid,
                "level": 1,
                "root_rpid": rpid,
                "rpid": rpid,
                "parent_rpid": 0,
                "uid": uid,
                "uname": name,
                "user_level": level,
                "ctime": ctime,
                "like": like,
                "message": context,
            }
        )
        total_count += 1

        # 相关回复数（楼中楼数量）
        try:
            rereply_text = reply["reply_control"]["sub_reply_entry_text"]
            rereply = int(re.findall(r'\d+', rereply_text)[0])
        except Exception:
            rereply = 0

        # 爬二级评论
        if is_second == "true" and rereply > 0:
            ps = 10
            second_web_location = 333.788
            total_size = math.ceil(rereply / ps)  # 总页数

            for page in tqdm(range(1, total_size + 1), desc=f"{bv_id} 楼层 {rpid} 二级评论", leave=False):
                url_second = (
                    "https://api.bilibili.com/x/v2/reply/reply"
                    f"?oid={oid}&type={type_}&root={rpid}"
                    f"&ps={ps}&pn={page}&web_location={second_web_location}"
                )
                second_resp = safe_get(url_second, headers=headers, timeout=15, retries=5, backoff=3)
                if not second_resp:
                    print(f"楼层 {rpid} 第 {page} 页获取失败，跳过此页。")
                    continue
                second_comment = second_resp.json()
                try:
                    second_replies = second_comment["data"].get("replies") or []
                    for second_reply in second_replies:
                        second_rpid = second_reply["rpid"]
                        second_uid = second_reply["mid"]
                        second_name = second_reply["member"]["uname"]
                        second_level = second_reply["member"]["level_info"]["current_level"]
                        second_context = second_reply["content"]["message"]
                        second_ctime = second_reply["ctime"]
                        second_like = second_reply["like"]

                        rows.append(
                            {
                                "bv_id": bv_id,
                                "oid": oid,
                                "level": 2,
                                "root_rpid": rpid,
                                "rpid": second_rpid,
                                "parent_rpid": rpid,
                                "uid": second_uid,
                                "uname": second_name,
                                "user_level": second_level,
                                "ctime": second_ctime,
                                "like": second_like,
                                "message": second_context,
                            }
                        )
                        total_count += 1
                except Exception:
                    print("二级评论爬取失败")

                time.sleep(0.5)  # 反爬

    # 分页：拿下一页的 offset
    pagination_reply = first_comment["data"]["cursor"].get("pagination_reply") or {}
    next_offset = pagination_reply.get("next_offset")
    if next_offset:
        # 递归爬下一页
        return start_crawler(bv_id, next_offset, is_second, oid, total_count)
    else:
        print(f"当前视频一共有（代码统计到的）: {total_count} 条评论")
        return total_count


if __name__ == "__main__":
    bv_id = input("请输入视频的bv号: ").strip()
    # 获取 oid 一次传进去，避免递归里重复请求
    oid = get_oid(bv_id)
    print(f"oid = {oid}")

    # 如果不爬取二级评论则设置 is_second = 'false'
    total = start_crawler(bv_id=bv_id, pagination_str="", is_second="true", oid=oid, total_count=0)

    # 写入 CSV
    if rows:
        out_name = f"bilibili_comments_{bv_id}.csv"
        fieldnames = [
            "bv_id", "oid", "level", "root_rpid",
            "rpid", "parent_rpid", "uid", "uname",
            "user_level", "ctime", "like", "message",
        ]
        with open(out_name, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"已保存 {len(rows)} 条评论到 {out_name}")
    else:
        print("没有抓到任何评论。")