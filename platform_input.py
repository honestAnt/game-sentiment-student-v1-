"""platform/content_type 前缀输入：训练、推理、评估共用的唯一前缀真源。

背景：v6/v7/v7b/v7.1 四轮实证单模型无平台特征必然跨域干扰（微博问责黑话=负面
与B站玩梗=中性文本相似、语义按平台相反）。v8 起在 text 前拼受限枚举前缀，
使模型可学平台条件规则。前缀必须是下方 PREFIXES 的受限集合，禁止自由拼接，
保证训练与推理分布一致。
"""

from __future__ import annotations

import os
import re

PLATFORM_CN = {"weibo": "微博", "bilibili": "B站", "xhs": "小红书"}
_VERSION_RE = re.compile(r"v(\d+)")

UNKNOWN_PREFIX = "[未知] "

PREFIXES = {
    ("weibo", "博文"): "[微博博文] ",
    ("weibo", "评论"): "[微博评论] ",
    ("bilibili", "评论"): "[B站评论] ",
}

# v8b 消融粒度：只拼 platform，不区分 content_type
PLATFORM_ONLY_PREFIXES = {
    "weibo": "[微博] ",
    "bilibili": "[B站] ",
}


def derive_content_type(row: dict) -> str | None:
    """row 自带 content_type 直接用；缺失时按 source_table 推导。

    round1/round2 血统的行没有 content_type 字段，但 source_table 全覆盖：
    含 "comment" -> 评论（weibo_post_comments_details / bilibili_comments 等）；
    含 "post_details" 且不含 "comment" -> 博文（dwd_crawl_weibo_post_details_merged）。
    """
    ct = row.get("content_type")
    if ct:
        return ct
    source = (row.get("source_table") or "").lower()
    if not source:
        return None
    if "comment" in source:
        return "评论"
    if "post_details" in source or "post_detail" in source:
        return "博文"
    return None


def model_input_text(
    text: str, platform: str | None, content_type: str | None, platform_only: bool = False
) -> str:
    """拼平台前缀：受限枚举查不到（缺字段 / xhs / 未支持组合）一律 [未知] 兜底。"""
    if platform_only:
        prefix = PLATFORM_ONLY_PREFIXES.get(platform, UNKNOWN_PREFIX)
    else:
        prefix = PREFIXES.get((platform, content_type), UNKNOWN_PREFIX)
    return f"{prefix}{text}"


def prefix_enabled(model_dir: str, choice: str = "auto") -> bool:
    """推理/评估入口共用的前缀开关判定。

    auto = 目录名解析出版本号 ≥8 视为前缀模型（v8/v8b/v9… 训练时输入带平台前缀），
    其余模型（v1–v7.1）按原文推理，旧口径数字保持可复现。
    """
    if choice == "on":
        return True
    if choice == "off":
        return False
    name = os.path.basename(os.path.normpath(model_dir))
    m = _VERSION_RE.search(name)
    return bool(m and int(m.group(1)) >= 8)
