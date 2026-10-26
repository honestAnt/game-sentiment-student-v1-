# -*- coding: utf-8 -*-
from build_v9b_train_mixed import (
    bili_neg_keep_bounds,
    downsample_bili_neg,
    downsample_slice_neg,
    validate_bili_slice,
    validate_mixed,
)


def test_downsample_slice_neg_caps_output_fraction():
    rows = [{"final_label": "负面", "id": f"n{i}"} for i in range(191)]
    rows += [{"final_label": "中性", "id": f"u{i}"} for i in range(138)]
    rows += [{"final_label": "正面", "id": f"p{i}"} for i in range(24)]
    out = downsample_slice_neg(rows, neg_max_frac=0.26, seed=49)
    n_neg = sum(1 for r in out if r["final_label"] == "负面")
    assert n_neg / len(out) <= 0.26 + 1e-9
    assert sum(1 for r in out if r["final_label"] == "中性") == 138
    assert sum(1 for r in out if r["final_label"] == "正面") == 24


def test_downsample_slice_neg_noop_when_within_cap():
    rows = [{"final_label": "负面", "id": "n1"}]
    rows += [{"final_label": "中性", "id": f"u{i}"} for i in range(9)]
    out = downsample_slice_neg(rows, neg_max_frac=0.26, seed=49)
    assert len(out) == 10


def test_bili_neg_keep_bounds_feasible():
    # fixed 已有足够负面，片内 cap 不会把全集拉出 21% 下限
    lo, hi, reasons = bili_neg_keep_bounds(
        n_non_neg=162, n_neg_avail=191, fixed_n=7000, fixed_neg=1600
    )
    assert lo <= hi, (lo, hi, reasons)
    assert reasons == []


def test_bili_neg_keep_bounds_matches_real_v9b_infeasible():
    """复现 v9b 实测：压 B站≤26% 后全集负面掉出 21% 下限。"""
    lo, hi, reasons = bili_neg_keep_bounds(
        n_non_neg=162, n_neg_avail=191, fixed_n=7742, fixed_neg=1573
    )
    assert lo > hi
    assert any("片内上限" in r for r in reasons)


def test_bili_neg_keep_bounds_infeasible_when_floor_above_slice_cap():
    # fixed 几乎无负面，片内又几乎全是负 → 要抬全集到 21% 需要很多 keep，但片内 cap 很小
    lo, hi, reasons = bili_neg_keep_bounds(
        n_non_neg=10, n_neg_avail=200, fixed_n=1000, fixed_neg=0
    )
    assert lo > hi
    assert reasons


def test_validate_bili_rejects_high_neg():
    orig = [{"id": f"n{i}", "final_label": "负面"} for i in range(5)]
    orig += [{"id": f"u{i}", "final_label": "中性"} for i in range(5)]
    kept = orig  # 50% neg
    errs = validate_bili_slice(kept, orig, neg_max=0.26)
    assert any("负面占比" in e for e in errs)


def test_validate_mixed_neg_band():
    rows = []
    for i in range(79):
        rows.append({
            "id": f"a{i}", "final_label": "中性", "final_emotion": "其他",
            "platform": "weibo", "content_type": "评论", "text": "x",
        })
    for i in range(21):
        rows.append({
            "id": f"b{i}", "final_label": "负面", "final_emotion": "愤怒",
            "platform": "weibo", "content_type": "评论", "text": "x",
        })
    assert validate_mixed(rows, base_ids=set()) == []


def test_downsample_bili_neg_exact_keep():
    rows = [{"final_label": "负面", "id": f"n{i}"} for i in range(10)]
    rows += [{"final_label": "中性", "id": f"u{i}"} for i in range(10)]
    out = downsample_bili_neg(rows, keep_neg=3, seed=1)
    assert sum(1 for r in out if r["final_label"] == "负面") == 3
    assert sum(1 for r in out if r["final_label"] == "中性") == 10
