import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling import build_round3_sample, downsample_by_label, is_low_conf


def _row(i, platform, pred, conf, text=None, content_type="评论"):
    return {
        "id": f"{i:032d}",
        "text": text or f"文本{i:04d}",
        "platform": platform,
        "content_type": content_type,
        "source_table": f"{platform}_src",
        "content_hash": f"{i:032d}",
        "student_pred": pred,
        "student_conf": conf,
        "p_neg": 0.05,
        "p_neu": 0.05,
        "p_pos": 0.9,
    }


def test_is_low_conf_threshold_only():
    assert is_low_conf(0.59) is True
    assert is_low_conf(0.60) is False
    # 不再看 p_neg：判中性且负面概率高也不再算难例（round2 的捞负面偏置已删）
    assert is_low_conf(0.95) is False


def test_build_round3_sample_slice_split_and_hard_cap():
    rows = []
    i = 0
    for plat, ctype in (("bilibili", "评论"), ("weibo", "评论"), ("weibo", "博文")):
        for _ in range(600):
            rows.append(_row(i, plat, "中性", 0.95, content_type=ctype))
            i += 1
        for _ in range(60):  # 低置信池
            rows.append(_row(i, plat, "正面", 0.4, content_type=ctype))
            i += 1
    df = pd.DataFrame(rows)
    fracs = {("bilibili", "评论"): 0.4, ("weibo", "评论"): 0.3, ("weibo", "博文"): 0.3}
    a = build_round3_sample(df, sample_size=200, seed=42, slice_fracs=fracs)
    b = build_round3_sample(df, sample_size=200, seed=42, slice_fracs=fracs)
    c = build_round3_sample(df, sample_size=200, seed=7, slice_fracs=fracs)
    assert len(a) == 200
    assert a["id"].nunique() == 200
    assert a["id"].tolist() == b["id"].tolist()
    assert a["id"].tolist() != c["id"].tolist()
    comp = a.groupby(["platform", "content_type"]).size().to_dict()
    assert comp[("bilibili", "评论")] == 80  # 0.4
    assert comp[("weibo", "评论")] == 60    # 0.3
    assert comp[("weibo", "博文")] == 60    # 0.3
    n_hard = int(a["student_conf"].map(is_low_conf).sum())
    assert n_hard <= 62  # 30% 配额 + 容差（hard 池充足时不应超太多）
    assert n_hard >= 54
    assert set(a.columns) >= {"id", "text", "platform", "content_type", "source_table", "content_hash"}


def test_build_round3_sample_fills_when_easy_scarce():
    rows = [_row(i, "bilibili", "正面", 0.4) for i in range(50)]
    rows += [_row(1000 + i, "bilibili", "中性", 0.99) for i in range(5)]
    rows += [_row(2000 + i, "weibo", "中性", 0.99) for i in range(200)]
    df = pd.DataFrame(rows)
    out = build_round3_sample(df, sample_size=100, seed=42)
    assert len(out) == 100
    assert out["id"].nunique() == 100


def test_build_round3_sample_fills_from_other_slice_when_short():
    rows = [_row(i, "bilibili", "中性", 0.95) for i in range(30)]  # B站池不足 40
    rows += [_row(1000 + i, "weibo", "中性", 0.95) for i in range(400)]
    df = pd.DataFrame(rows)
    out = build_round3_sample(df, sample_size=100, seed=42)
    assert len(out) == 100
    assert int((out["platform"] == "bilibili").sum()) == 30


def test_downsample_by_label_caps_negative():
    rows = (
        [{"id": f"n{i}", "final_label": "负面"} for i in range(50)]
        + [{"id": f"z{i}", "final_label": "中性"} for i in range(50)]
    )
    df = pd.DataFrame(rows)
    out = downsample_by_label(df, neg_max_frac=0.26)
    assert len(out) == 100 - 24  # 负面 50 -> 26
    assert int((out["final_label"] == "负面").sum()) == 26


def test_downsample_by_label_noop_when_within_cap():
    rows = (
        [{"id": f"n{i}", "final_label": "负面"} for i in range(20)]
        + [{"id": f"z{i}", "final_label": "中性"} for i in range(80)]
    )
    df = pd.DataFrame(rows)
    out = downsample_by_label(df, neg_max_frac=0.26)
    assert len(out) == 100
    assert out["id"].tolist() == sorted(out["id"].tolist()) or set(out["id"]) == set(df["id"])
