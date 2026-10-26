import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMMENT_SOURCE, POST_SOURCE
from sampling import build_round1_sample, length_bucket, prepare_pool, sample_spread_length


def test_length_bucket():
    assert length_bucket("短") == "short"
    assert length_bucket("x" * 20) == "short"
    assert length_bucket("x" * 21) == "mid"
    assert length_bucket("x" * 128) == "mid"
    assert length_bucket("x" * 129) == "long"


def _row(i, source, aliyun, text, extra_hash=None):
    h = extra_hash or f"{i:032d}"
    return {
        "content_hash": h,
        "request_content_text": text,
        "source_table": source,
        "aliyun_final": aliyun,
        "positive_prob_aliyun": 0.5,
        "negative_prob_aliyun": 0.5,
    }


def test_prepare_pool_deterministic_row_order():
    dup_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    rows_a = [
        _row(1, POST_SOURCE, 1, "first row", extra_hash=dup_hash),
        _row(2, COMMENT_SOURCE, 3, "unique b", extra_hash="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        _row(3, POST_SOURCE, 2, "dup second", extra_hash=dup_hash),
    ]
    rows_b = list(reversed(rows_a))
    pool_a = prepare_pool(pd.DataFrame(rows_a))
    pool_b = prepare_pool(pd.DataFrame(rows_b))
    assert list(pool_a["id"]) == list(pool_b["id"])
    assert list(pool_a["aliyun_final"]) == list(pool_b["aliyun_final"])
    dup = pool_a[pool_a["content_hash"] == dup_hash].iloc[0]
    assert int(dup["aliyun_final"]) == 1


def test_prepare_pool_cleans_and_dedupes():
    df = pd.DataFrame([
        _row(1, POST_SOURCE, 1, "  你好  http://t.cn/abc  世界  "),
        _row(2, POST_SOURCE, 1, "你好 世界", extra_hash="00000000000000000000000000000001"),
        _row(3, POST_SOURCE, 1, "。"),
        _row(4, COMMENT_SOURCE, 3, "太贵了"),
    ])
    pool = prepare_pool(df)
    assert set(pool.columns) >= {
        "id", "text", "platform", "source_table", "content_hash", "aliyun_final"
    }
    assert "aliyun_final" in pool.columns
    texts = set(pool["text"])
    assert "你好 世界" in texts
    assert all(len(t) >= 2 for t in pool["text"])
    assert (pool["platform"] == "weibo").all()
    assert (pool["id"] == pool["content_hash"]).all()


def _make_pool():
    rows = []
    n = 0
    def add(source, aliyun, k, prefix):
        nonlocal n
        for j in range(k):
            n += 1
            length = 5 if j % 3 == 0 else (40 if j % 3 == 1 else 200)
            rows.append(_row(n, source, aliyun, prefix + ("哈" * length)))
    add(COMMENT_SOURCE, 1, 80, "cpos")
    add(COMMENT_SOURCE, 2, 50, "cneu")
    add(COMMENT_SOURCE, 3, 15, "cneg")
    add(POST_SOURCE, 1, 200, "ppos")
    add(POST_SOURCE, 2, 40, "pneu")
    add(POST_SOURCE, 3, 10, "pneg")
    return prepare_pool(pd.DataFrame(rows))


def test_build_round1_takes_all_negatives_and_quotas():
    pool = _make_pool()
    sampled = build_round1_sample(
        pool, sample_size=120, target_comments=40, target_neutrals=30, seed=42
    )
    assert len(sampled) == 120
    assert sampled["id"].nunique() == 120
    n_neg_pool = (pool["aliyun_final"] == 3).sum()
    assert (sampled["aliyun_final"] == 3).sum() == n_neg_pool
    n_comments = (sampled["source_table"] == COMMENT_SOURCE).sum()
    assert 35 <= n_comments <= 45
    assert (sampled["aliyun_final"] == 2).sum() >= 25
    assert "aliyun_final" in sampled.columns


def test_build_round1_deterministic():
    pool = _make_pool()
    a = build_round1_sample(pool, sample_size=80, target_comments=30, target_neutrals=25, seed=42)
    b = build_round1_sample(pool, sample_size=80, target_comments=30, target_neutrals=25, seed=42)
    assert list(a["id"]) == list(b["id"])


def test_sample_spread_length_returns_exactly_n():
    rows = []
    for i in range(30):
        length = 5 if i % 3 == 0 else (40 if i % 3 == 1 else 200)
        rows.append(_row(i, POST_SOURCE, 1, "x" + ("哈" * length)))
    df = prepare_pool(pd.DataFrame(rows))
    for n in (10, 15, 20, 25):
        rng = np.random.RandomState(42)
        sampled = sample_spread_length(df, n, rng)
        assert len(sampled) == n


def test_build_round1_keeps_all_negatives_when_exceeds_sample_size():
    rows = []
    for i in range(20):
        rows.append(_row(i, COMMENT_SOURCE, 3, f"neg{i}"))
    for i in range(20, 100):
        rows.append(_row(i, POST_SOURCE, 1, f"pos{i}" + ("哈" * 10)))
    pool = prepare_pool(pd.DataFrame(rows))
    sampled = build_round1_sample(
        pool, sample_size=15, target_comments=10, target_neutrals=5, seed=42
    )
    assert (sampled["aliyun_final"] == 3).sum() == 20
    assert len(sampled) == 20


def test_sample_spread_length_imbalanced_buckets():
    rows = []
    for i in range(50):
        rows.append(_row(i, POST_SOURCE, 1, "s" + ("哈" * 18)))
    for i in range(50, 53):
        rows.append(_row(i, POST_SOURCE, 1, "m" + ("哈" * 39)))
    for i in range(53, 56):
        rows.append(_row(i, POST_SOURCE, 1, "l" + ("哈" * 199)))
    df = prepare_pool(pd.DataFrame(rows))
    rng = np.random.RandomState(42)
    sampled = sample_spread_length(df, 20, rng)
    assert len(sampled) == 20
    buckets = sampled["text"].map(length_bucket)
    assert (buckets == "short").sum() == 14
    assert (buckets == "mid").sum() == 3
    assert (buckets == "long").sum() == 3


def test_unknown_source_does_not_consume_post_quota():
    unknown_hash = "99999999999999999999999999999999"
    rows = []
    for i in range(10):
        rows.append(_row(i, COMMENT_SOURCE, 1, f"cpos{i}" + ("哈" * 10)))
    for i in range(10, 20):
        rows.append(_row(i, POST_SOURCE, 1, f"ppos{i}" + ("哈" * 10)))
    rows.append(
        _row(
            99,
            "other_table",
            1,
            "unknown" + ("哈" * 200),
            extra_hash=unknown_hash,
        )
    )
    pool = prepare_pool(pd.DataFrame(rows))
    sampled = build_round1_sample(
        pool, sample_size=20, target_comments=10, target_neutrals=0, seed=42
    )
    assert len(sampled) == 20
    assert unknown_hash not in set(sampled["id"])
    assert (sampled["source_table"] == COMMENT_SOURCE).sum() == 10
    assert (sampled["source_table"] == POST_SOURCE).sum() == 10
