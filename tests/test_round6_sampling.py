import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling import build_round6_sample


def _row(i, text, platform="weibo", ctype="评论", pred="中性", conf=0.9, src="dwd_c"):
    return {
        "id": f"{platform}_{ctype}_{i}",
        "text": text,
        "platform": platform,
        "content_type": ctype,
        "source_table": src,
        "student_pred": pred,
        "student_conf": conf,
    }


def _pool():
    rows = []
    # weibo 评论：难例（cheer / 低置信正面）+ 普通
    for i in range(40):
        rows.append(_row(i, f"普通评论正文{i}", pred="中性", conf=0.92))
    for i in range(40, 55):
        rows.append(_row(i, f"WE加油冲鸭{i}", pred="中性", conf=0.88))  # cheer
    for i in range(55, 65):
        rows.append(_row(i, f"打得还行吧{i}", pred="正面", conf=0.62))  # 低置信正面
    # weibo 博文
    for i in range(30):
        rows.append(_row(i, f"博文正文足够长{i}", platform="weibo", ctype="博文",
                         pred="中性", conf=0.9, src="dwd_p"))
    # B站：抗议 + 普通
    for i in range(20):
        rows.append(_row(i, f"普通玩梗评论{i}", platform="bilibili", ctype="评论",
                         pred="中性", conf=0.9, src="dwd_b"))
    for i in range(20, 28):
        rows.append(_row(i, f"官方出来道歉{i}", platform="bilibili", ctype="评论",
                         pred="中性", conf=0.91, src="dwd_b"))
    return pd.DataFrame(rows)


def test_round6_slice_quota_and_hard_share():
    out = build_round6_sample(_pool(), n_total=40, seed=48)
    assert len(out) == 40
    n_wc = int(((out["platform"] == "weibo") & (out["content_type"] == "评论")).sum())
    n_wp = int(((out["platform"] == "weibo") & (out["content_type"] == "博文")).sum())
    n_bc = int(((out["platform"] == "bilibili") & (out["content_type"] == "评论")).sum())
    assert (n_wc, n_wp, n_bc) == (20, 10, 10)
    wc = out[(out["platform"] == "weibo") & (out["content_type"] == "评论")]
    hard = wc["text"].str.contains("加油|打得还行")
    assert hard.sum() >= 6  # 30% of 20
    assert (out["platform"] == "weibo").any() and (out["platform"] == "bilibili").any()
    assert set(out.columns) >= {
        "id", "text", "platform", "content_type", "source_table", "content_hash",
    }


def test_round6_bili_prioritizes_protest():
    out = build_round6_sample(_pool(), n_total=40, seed=48)
    bili = out[(out["platform"] == "bilibili") & (out["content_type"] == "评论")]
    assert bili["text"].str.contains("道歉").any()


def test_round6_drops_excluded_and_short_and_dups():
    extra = pd.DataFrame([
        _row(900, "已进训练的文本"),
        _row(901, "x"),
        _row(902, "重复候选AAA"),
        _row(903, "重复候选AAA", platform="weibo", ctype="博文"),
    ])
    pool = pd.concat([_pool(), extra], ignore_index=True)
    out = build_round6_sample(pool, n_total=40, exclude_texts={"已进训练的文本"}, seed=48)
    assert "已进训练的文本" not in set(out["text"])
    assert "x" not in set(out["text"])
    assert not out["content_hash"].duplicated().any()
    assert not out["text"].duplicated().any()


def test_round6_deterministic():
    a = build_round6_sample(_pool(), n_total=40, seed=48)
    b = build_round6_sample(_pool(), n_total=40, seed=48)
    assert a["id"].tolist() == b["id"].tolist()
