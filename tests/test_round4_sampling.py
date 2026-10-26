import sys
import os

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling import build_round4_bili_hard_sample, build_round4_multigame_sample


def _blind_row(seq, text, ctype="博文", label=None):
    return {"seq": str(seq), "type": ctype, "text": text, "human_label": label}


def test_multigame_drops_labeled_and_excluded_and_dup():
    rows = [
        _blind_row(1, "金标行", "博文", label="正向"),            # 有人工标注 -> 必须排除（金标本体）
        _blind_row(2, "已训练文本", "博文"),                      # 与训练重合 -> 排除
        _blind_row(3, "短", "博文"),                             # 长度 <2 -> 排除
        _blind_row(4, "重复文本A", "评论"),
        _blind_row(5, "重复文本A", "博文"),                      # 文本重复 -> 去重
        _blind_row(6, "多游戏正文一", "博文"),
        _blind_row(7, "多游戏正文二", "评论"),
    ]
    df = pd.DataFrame(rows)
    out = build_round4_multigame_sample(df, exclude_texts={"已训练文本"}, seed=45)
    texts = set(out["text"])
    assert texts == {"重复文本A", "多游戏正文一", "多游戏正文二"}
    assert "blind2000_3" not in set(out["id"])
    assert set(out["id"]) == {"blind2000_4", "blind2000_6", "blind2000_7"}
    assert (out["platform"] == "weibo").all()
    assert set(out["content_type"]) <= {"博文", "评论"}
    assert (out["source_table"] == "weibo_sentiment_doubleblind_2000").all()
    assert out["content_hash"].notna().all()
    # 确定性
    out2 = build_round4_multigame_sample(df, exclude_texts={"已训练文本"}, seed=45)
    assert out["id"].tolist() == out2["id"].tolist()


def test_multigame_never_empty_label_leak():
    # 即使全部行都有人工标注（极端情况），也不得产出任何行
    df = pd.DataFrame([_blind_row(i, f"文本{i}", "博文", label="中性") for i in range(5)])
    out = build_round4_multigame_sample(df, exclude_texts=set())
    assert len(out) == 0


def _bili_row(i, text, agent="中立", pred="中性"):
    return {"id": f"{i:032d}", "text": text, "agent_label": agent, "v5_pred": pred}


def _score_map(rows):
    return {r["id"]: {"student_pred": r["v5_pred"], "student_conf": 0.9} for r in rows}


def test_bili_hard_prioritizes_cheer_and_sarcasm():
    rows = [
        _bili_row(1, "T1加油", pred="中性"),                    # 应援判中性 -> 第1层
        _bili_row(2, "捞批捞实至名归", pred="正面"),             # 黑话 -> 第3层
        _bili_row(3, "agent觉得好", agent="正向", pred="中性"),   # agent-v5 分歧 -> 第2层
        _bili_row(4, "普通玩梗评论一", pred="中性"),
        _bili_row(5, "普通玩梗评论二", pred="中性"),
    ]
    df = pd.DataFrame(rows)
    out = build_round4_bili_hard_sample(
        df, _score_map(rows), exclude_ids=set(), n_target=3, seed=45,
    )
    assert len(out) == 3
    assert set(out["id"]) == {f"{i:032d}" for i in (1, 2, 3)}   # 难例层优先于随机补齐
    assert (out["platform"] == "bilibili").all()
    assert (out["content_type"] == "评论").all()
    assert (out["source_table"] == "bilibili_comments_1600").all()


def test_bili_hard_pads_with_random_when_tiers_short():
    rows = [_bili_row(i, "普通评论" + "长" * i, pred="中性") for i in range(50)]
    df = pd.DataFrame(rows)
    out = build_round4_bili_hard_sample(
        df, _score_map(rows), exclude_ids=set(), n_target=20, seed=45,
    )
    assert len(out) == 20
    assert out["id"].nunique() == 20


def test_bili_hard_excludes_used_ids_and_short_texts():
    rows = [
        _bili_row(1, "T1加油", pred="中性"),
        _bili_row(2, "x", pred="中性"),                          # 太短
    ]
    df = pd.DataFrame(rows)
    out = build_round4_bili_hard_sample(
        df, _score_map(rows), exclude_ids={f"{1:032d}"}, n_target=10, seed=45,
    )
    assert f"{1:032d}" not in set(out["id"])
    assert f"{2:032d}" not in set(out["id"])
