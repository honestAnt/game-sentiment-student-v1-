import json
from pathlib import Path

import pytest

from platform_input import (
    PREFIXES,
    UNKNOWN_PREFIX,
    derive_content_type,
    model_input_text,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "labeled" / "v7_1_train_mixed.jsonl"


def test_prefix_enum_covers_three_gold_slices():
    assert PREFIXES == {
        ("weibo", "博文"): "[微博博文] ",
        ("weibo", "评论"): "[微博评论] ",
        ("bilibili", "评论"): "[B站评论] ",
    }


def test_model_input_text_known_and_unknown():
    assert model_input_text("加油", "bilibili", "评论") == "[B站评论] 加油"
    assert model_input_text("残红", "weibo", "博文") == "[微博博文] 残红"
    assert model_input_text("真好看", "weibo", "评论") == "[微博评论] 真好看"
    # 缺 content_type / 不支持平台 / 缺 platform 全部走 [未知]
    assert model_input_text("x", "weibo", None) == f"{UNKNOWN_PREFIX}x"
    assert model_input_text("x", "xhs", "笔记") == f"{UNKNOWN_PREFIX}x"
    assert model_input_text("x", None, None) == f"{UNKNOWN_PREFIX}x"
    assert model_input_text("x", "unknown", None) == f"{UNKNOWN_PREFIX}x"


def test_model_input_text_platform_only():
    # v8b 消融粒度：只拼平台，不区分 content_type；未知平台仍 [未知]
    assert model_input_text("加油", "bilibili", "评论", platform_only=True) == "[B站] 加油"
    assert model_input_text("残红", "weibo", "博文", platform_only=True) == "[微博] 残红"
    assert model_input_text("真好看", "weibo", "评论", platform_only=True) == "[微博] 真好看"
    assert model_input_text("x", "xhs", "笔记", platform_only=True) == f"{UNKNOWN_PREFIX}x"
    assert model_input_text("x", None, None, platform_only=True) == f"{UNKNOWN_PREFIX}x"


def test_derive_content_type_prefers_explicit_field():
    assert derive_content_type({"content_type": "博文", "source_table": "bilibili_comments"}) == "博文"


@pytest.mark.parametrize(
    "source_table,expected",
    [
        ("dwd_crawl_weibo_post_details_merged", "博文"),
        ("weibo_post_details", "博文"),
        ("weibo_post_comments_details", "评论"),
        ("dwd_crawl_weibo_post_comments_details_merged", "评论"),
        ("bilibili_comments", "评论"),
    ],
)
def test_derive_content_type_from_source_table(source_table, expected):
    assert derive_content_type({"source_table": source_table}) == expected


def test_derive_content_type_unresolvable():
    assert derive_content_type({}) is None
    assert derive_content_type({"source_table": "something_else"}) is None


def test_v7_1_train_full_coverage():
    """训练语料 6895 行每行都能映射到非 [未知] 前缀（推导规则全覆盖断言）。"""
    if not TRAIN_FILE.exists():
        pytest.skip("train file not present")
    rows = [json.loads(line) for line in TRAIN_FILE.open(encoding="utf-8")]
    assert len(rows) == 6895
    for r in rows:
        prefix = model_input_text(r["text"], r.get("platform"), derive_content_type(r))
        assert not prefix.startswith(UNKNOWN_PREFIX), r.get("id")
