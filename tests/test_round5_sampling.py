import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling import _proportional_alloc, build_round5_pool, build_round5_sample


def _row(i, text, source="xhs_comments", sent="2"):
    return {
        "id": f"id_{source}_{i}",
        "text": text,
        "platform": "xhs" if source.startswith("xhs") else "bilibili",
        "content_type": "评论" if source.endswith("comments") else "笔记",
        "source_table": f"tbl_{source}",
        "machine_sentiment": sent,
    }


def test_pool_filters_short_truncated_excluded_and_dedups_across_sources():
    rows = [
        _row(1, "正常评论一"),
        _row(2, "x"),                                  # 长度 <2 -> 排除
        _row(3, "平台截断的文本", sent="1"),
        _row(4, "与训练重合的文本"),
        _row(5, "跨源重复文本"),
        _row(6, "正常评论二", source="bili_comments", sent="1"),
        _row(7, "跨源重复文本", source="bili_comments"),  # 与 xhs 行文本相同 -> 去重留先者
    ]
    rows[2]["truncated"] = "1"                         # 只有这条带截断标记
    df = pd.DataFrame(rows)
    out = build_round5_pool({"xhs_comments": df[df.id.str.contains("xhs")],
                             "bili_comments": df[df.id.str.contains("bili")]},
                            exclude_texts={"与训练重合的文本"})
    texts = set(out["text"])
    assert texts == {"正常评论一", "跨源重复文本", "正常评论二"}
    assert out["content_hash"].notna().all()
    assert (out["_source"] == "xhs_comments").sum() == 2   # 重复文本归先出现的 xhs 源


def test_pool_without_truncated_column_still_works():
    df = pd.DataFrame([_row(1, "无截断列的行")])
    out = build_round5_pool({"xhs_comments": df}, exclude_texts=set())
    assert len(out) == 1


def test_sample_takes_all_small_slices_and_fills_xhs_by_prior():
    frames = {
        "bili_comments": pd.DataFrame([_row(0, "稀缺片文本a", source="bili_comments")]),
        "bili_videos": pd.DataFrame([_row(0, "稀缺片文本b", source="bili_videos")]),
        "xhs_notes": pd.DataFrame([_row(0, "稀缺片文本c", source="xhs_notes")]),
        "xhs_comments": pd.DataFrame(
            [_row(100 + i, f"评论文本{i}", sent=("2" if i % 3 == 0 else ("1" if i % 3 == 1 else "3")))
             for i in range(90)]),                       # 先验 2:1:3 -> 30/30/30
    }
    pool = build_round5_pool(frames, exclude_texts=set())
    out = build_round5_sample(pool, total_target=63, seed=47)
    assert len(out) == 63
    # 稀缺片全取
    for s in ("bili_comments", "bili_videos", "xhs_notes"):
        assert (out["_source"] == s).sum() == 1
    # xhs 按先验分层：各 20 条，比例保持 1:1:1
    xhs_part = out[out["_source"] == "xhs_comments"]
    assert len(xhs_part) == 60
    prior = xhs_part["machine_sentiment"].value_counts().to_dict()
    assert prior == {"2": 20, "1": 20, "3": 20}


def test_sample_deterministic_and_no_dup():
    frames = {"xhs_comments": pd.DataFrame(
        [_row(i, f"文本{i}", sent=["1", "2", "3"][i % 3]) for i in range(50)])}
    pool = build_round5_pool(frames, exclude_texts=set())
    a = build_round5_sample(pool, total_target=20, seed=47)
    b = build_round5_sample(pool, total_target=20, seed=47)
    assert a["id"].tolist() == b["id"].tolist()
    assert not a["content_hash"].duplicated().any()


def test_proportional_alloc_largest_remainder_and_cap():
    assert _proportional_alloc(10, {"a": 80, "b": 15, "c": 5}) == {"a": 8, "b": 2, "c": 0}
    assert _proportional_alloc(100, {"a": 1}) == {"a": 1}          # 不超过组自身
    assert _proportional_alloc(0, {"a": 10}) == {"a": 0}
    assert _proportional_alloc(5, {}) == {}
