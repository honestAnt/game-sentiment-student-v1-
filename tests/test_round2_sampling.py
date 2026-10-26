import json
import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling import build_round2_sample, is_hard_example
from export_round2_candidates import write_round2_outputs


def _row(i, source, pred, conf, p_neg, text=None):
    return {
        "id": f"{i:032d}",
        "text": text or f"评论文本{i:04d}",
        "platform": "weibo",
        "source_table": source,
        "content_hash": f"{i:032d}",
        "student_pred": pred,
        "student_conf": conf,
        "p_neg": p_neg,
        "p_neu": 1.0 - conf if pred != "中性" else conf,
        "p_pos": 0.0,
    }


def test_is_hard_low_conf_or_neutral_with_neg_mass():
    assert is_hard_example(conf=0.59, pred="正面", p_neg=0.01) is True
    assert is_hard_example(conf=0.90, pred="中性", p_neg=0.20) is True
    assert is_hard_example(conf=0.90, pred="中性", p_neg=0.19) is False
    assert is_hard_example(conf=0.90, pred="负面", p_neg=0.90) is False


def test_round2_sample_size_post_cap_hard_frac_seed():
    comment = "weibo_post_comments_details"
    post = "weibo_post_details"
    rows = []
    for i in range(3000):
        if i < 400:
            rows.append(_row(i, post, "中性", 0.95, 0.02, text=f"官方通稿{i}"))
        elif i < 1600:
            rows.append(_row(i, comment, "中性", 0.92, 0.25, text=f"阴阳{i}"))
        elif i < 2200:
            rows.append(_row(i, comment, "中性", 0.50, 0.30, text=f"低置信{i}"))
        else:
            pred = ["负面", "中性", "正面"][i % 3]
            rows.append(_row(i, comment, pred, 0.95, 0.90 if pred == "负面" else 0.02, text=f"简单{i}"))
    df = pd.DataFrame(rows)
    a = build_round2_sample(df, sample_size=200, seed=42)
    b = build_round2_sample(df, sample_size=200, seed=42)
    c = build_round2_sample(df, sample_size=200, seed=7)
    assert len(a) == 200
    assert a["id"].tolist() == b["id"].tolist()
    assert a["id"].tolist() != c["id"].tolist()
    n_post = int((a["source_table"] == post).sum())
    assert n_post <= 20
    n_hard = int(a.apply(lambda r: is_hard_example(r["student_conf"], r["student_pred"], r["p_neg"]), axis=1).sum())
    assert n_hard >= 130
    assert a["id"].nunique() == 200
    assert set(a.columns) >= {"id", "text", "platform", "source_table", "content_hash"}


def test_round2_sample_fills_when_hard_is_scarce():
    comment = "weibo_post_comments_details"
    rows = [_row(i, comment, "正面", 0.99, 0.01) for i in range(50)]
    rows += [_row(1000 + i, comment, "中性", 0.4, 0.3) for i in range(5)]
    df = pd.DataFrame(rows)
    out = build_round2_sample(df, sample_size=20, seed=42)
    assert len(out) == 20
    assert out["id"].nunique() == 20


def test_write_round2_outputs_strips_scores_from_shards():
    comment = "weibo_post_comments_details"
    rows = []
    for i in range(15):
        rows.append(_row(i, comment, "中性", 0.5, 0.3))
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        write_round2_outputs(df, td, shard_size=10)
        shards = sorted(n for n in os.listdir(td) if n.startswith("shard_"))
        assert shards == ["shard_00.jsonl", "shard_01.jsonl"]
        with open(os.path.join(td, "shard_00.jsonl"), encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert set(rec.keys()) == {"id", "text", "platform", "source_table", "content_hash"}
        with open(os.path.join(td, "student_score_sidecar.jsonl"), encoding="utf-8") as f:
            side = [json.loads(x) for x in f]
        assert len(side) == 15
        assert "student_pred" in side[0]
        assert "hard" in side[0]
