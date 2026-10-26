# -*- coding: utf-8 -*-
"""gold_eval_v6 回收：多 xlsx 一次合并、标签归一化、与 v5 去重。"""

import json

import pandas as pd
import pytest

from merge_gold_weibo_comments import (
    append_new_gold,
    labeled_records_from_xlsx,
    merge_gold_from_xlsx_paths,
)


def _xlsx(path, rows):
    pd.DataFrame(rows).to_excel(path, index=False)


def test_normalize_zhongli_and_zhengxiang(tmp_path):
    p = tmp_path / "a.xlsx"
    _xlsx(p, [
        {"序号": "id1", "类型": "评论", "文本": "恭喜", "人工标注-杠": "正向"},
        {"序号": "id2", "类型": "评论", "文本": "嗯", "人工标注-杠": "中立"},
        {"序号": "id3", "类型": "评论", "文本": "差", "人工标注-杠": "负面"},
    ])
    recs, invalid = labeled_records_from_xlsx(str(p))
    assert invalid == []
    by_id = {r["raw_id"]: r["gold_label"] for r in recs}
    assert by_id == {"id1": "正面", "id2": "中性", "id3": "负面"}


def test_skip_empty_and_garbage_labels(tmp_path):
    p = tmp_path / "a.xlsx"
    _xlsx(p, [
        {"序号": "ok", "类型": "评论", "文本": "加油", "人工标注-杠": "正面"},
        {"序号": "empty", "类型": "评论", "文本": "参考文献呢", "人工标注-杠": None},
        {"序号": "bad", "类型": "评论", "文本": "xxx", "人工标注-杠": "呵呵"},
    ])
    recs, invalid = labeled_records_from_xlsx(str(p))
    assert [r["raw_id"] for r in recs] == ["ok"]
    assert invalid == [("bad", "呵呵")]


def test_append_skips_v5_text_and_ge6_id_collision():
    gold = [
        {"id": "1", "text": "已在v5", "platform": "weibo", "content_type": "博文", "gold_label": "中性"},
        {"id": "ge6_dup", "text": "旧扩样", "platform": "weibo", "content_type": "评论", "gold_label": "正面"},
    ]
    incoming = [
        {"raw_id": "n1", "text": "已在v5", "content_type": "评论", "gold_label": "负面"},
        {"raw_id": "dup", "text": "新文本但id撞", "content_type": "评论", "gold_label": "中性"},
        {"raw_id": "n2", "text": "全新评论", "content_type": "评论", "gold_label": "正面"},
    ]
    merged, dup_text, dup_id = append_new_gold(gold, incoming)
    assert dup_text == 1 and dup_id == 1
    assert [r["id"] for r in merged] == ["1", "ge6_dup", "ge6_n2"]
    assert merged[0] == gold[0]  # v5 原行字节级不动
    assert merged[-1]["gold_label"] == "正面"
    assert merged[-1]["platform"] == "weibo"
    assert merged[-1]["content_type"] == "评论"


def test_two_xlsx_merged_once(tmp_path):
    lol = tmp_path / "lol.xlsx"
    sr = tmp_path / "sr.xlsx"
    _xlsx(lol, [
        {"序号": "h1", "类型": "评论", "文本": "恭喜G2", "人工标注-杠": "正面"},
        {"序号": "h2", "类型": "评论", "文本": "赛制如何划分", "人工标注-杠": "中立"},
    ])
    _xlsx(sr, [
        {"序号": "s1", "类型": "评论", "文本": "不值得", "人工标注-杠": "负向"},
        {"序号": "s2", "类型": "评论", "文本": "换卡", "人工标注-杠": "中性"},
    ])
    v5 = [
        {"id": "old", "text": "v5原文", "platform": "weibo", "content_type": "博文", "gold_label": "中性"},
    ]
    merged, stats = merge_gold_from_xlsx_paths(v5, [str(lol), str(sr)])
    ids = [r["id"] for r in merged]
    assert ids[0] == "old"
    assert set(ids[1:]) == {"ge6_h1", "ge6_h2", "ge6_s1", "ge6_s2"}
    labels = {r["id"]: r["gold_label"] for r in merged}
    assert labels["ge6_h1"] == "正面"
    assert labels["ge6_h2"] == "中性"
    assert labels["ge6_s1"] == "负面"
    assert labels["ge6_s2"] == "中性"
    assert stats["n_labeled"] == 4
    assert stats["n_new"] == 4
    assert stats["n_invalid"] == 0
