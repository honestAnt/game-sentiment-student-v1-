# -*- coding: utf-8 -*-
"""v10 混训集构建单测：整行复制去重、标签冲突拒绝、回放语义、校验闸。"""

import pytest

from build_v10_train_mixed import (
    NEG_MAX_FRAC, NEG_MIN_FRAC, backfill_content_type, build, dedup_rows, validate,
)


def _row(rid, label="中性", platform="bilibili", ct="评论", text=None, emotion="其他"):
    return {"id": rid, "text": text or f"t_{rid}", "final_label": label,
            "final_emotion": emotion, "platform": platform, "content_type": ct}


class TestDedupRows:
    def test_removes_exact_triplicates_keeping_first(self):
        # v7_1 的真实形态：lol_* 行整行复制 3 遍
        rows = [_row("lol_bilibili_182")] * 3 + [_row("hash_a")]
        out, removed = dedup_rows(rows)
        assert [r["id"] for r in out] == ["lol_bilibili_182", "hash_a"]
        assert removed == {("bilibili", "评论"): 2}

    def test_preserves_original_order(self):
        rows = [_row("a"), _row("b"), _row("a"), _row("c")]
        out, _ = dedup_rows(rows)
        assert [r["id"] for r in out] == ["a", "b", "c"]

    def test_no_dups_is_a_noop(self):
        rows = [_row("a"), _row("b")]
        out, removed = dedup_rows(rows)
        assert out == rows and removed == {}

    def test_removal_counted_per_slice(self):
        rows = ([_row("b1")] * 3 + [_row("w1", platform="weibo", ct="评论")] * 2)
        _, removed = dedup_rows(rows)
        assert removed == {("bilibili", "评论"): 2, ("weibo", "评论"): 1}

    def test_label_conflict_refuses_to_dedup(self):
        """同 id 不同标签是数据问题，不能当复制静默丢一条。"""
        rows = [_row("x", label="中性"), _row("x", label="负面")]
        with pytest.raises(SystemExit, match="标签冲突"):
            dedup_rows(rows)

    def test_text_conflict_refuses_to_dedup(self):
        rows = [_row("x", text="原文"), _row("x", text="不同的文本")]
        with pytest.raises(SystemExit, match="标签冲突"):
            dedup_rows(rows)


class TestBackfillContentType:
    def test_derives_from_source_table_when_missing(self):
        row = {"id": "a", "text": "t", "platform": "bilibili",
               "source_table": "bilibili_comments"}
        assert backfill_content_type(row)["content_type"] == "评论"

    def test_existing_value_not_overwritten(self):
        row = _row("a", ct="评论")
        row["source_table"] = "dwd_crawl_weibo_post_details_merged"
        assert backfill_content_type(row)["content_type"] == "评论"


def _mix(n_neg, n_neu, platform="weibo", ct="评论", prefix="x"):
    return ([_row(f"{prefix}n{i}", "负面", platform, ct) for i in range(n_neg)]
            + [_row(f"{prefix}u{i}", "中性", platform, ct) for i in range(n_neu)])


class TestBuild:
    def test_dedup_reduces_base_and_reports_per_slice(self):
        base = [_row("lol_b")] * 3 + _mix(30, 70, prefix="b")
        r6 = _mix(10, 20, prefix="r")
        rows, stats = build(base, r6, dedup=True, post_replay=0)
        assert stats["dedup_removed_total"] == 2
        assert len(rows) == len(base) - 2 + len(r6)

    def test_keep_dups_preserves_every_row(self):
        base = [_row("lol_b")] * 3 + _mix(30, 70, prefix="b")
        r6 = _mix(10, 20, prefix="r")
        rows, stats = build(base, r6, dedup=False, post_replay=0)
        assert len(rows) == len(base) + len(r6)
        assert "dedup_removed" not in stats

    def test_post_replay_duplicates_only_weibo_posts(self):
        base = _mix(10, 20, ct="博文") + _mix(10, 20, platform="bilibili", prefix="b")
        rows, stats = build(base, [], dedup=True, post_replay=1)
        assert stats["post_replay_rows"] == 30
        posts = [r for r in rows if (r["platform"], r["content_type"]) == ("weibo", "博文")]
        bili = [r for r in rows if r["platform"] == "bilibili"]
        assert len(posts) == 60 and len(bili) == 30

    def test_post_replay_scales_with_n(self):
        base = _mix(10, 20, ct="博文")
        _, stats = build(base, [], dedup=True, post_replay=2)
        assert stats["post_replay_rows"] == 60


class TestValidate:
    def test_accepts_healthy_mix(self):
        rows = _mix(23, 77)          # 23% 负面，落在 [21%, 26%]
        assert validate(rows, allow_dups=False) == []

    def test_rejects_negative_fraction_below_band(self):
        errs = validate(_mix(10, 90), allow_dups=False)
        assert any("负面占比" in e for e in errs)

    def test_rejects_negative_fraction_above_band(self):
        errs = validate(_mix(40, 60), allow_dups=False)
        assert any("负面占比" in e for e in errs)

    def test_band_boundaries_inclusive(self):
        assert validate(_mix(21, 79), allow_dups=False) == []
        assert validate(_mix(26, 74), allow_dups=False) == []

    def test_rejects_leftover_duplicate_ids_when_dedup_expected(self):
        rows = _mix(23, 77) + [_row("xn0", "负面")]
        errs = validate(rows, allow_dups=False)
        assert any("重复 id" in e for e in errs)

    def test_allows_duplicate_ids_when_replay_declared(self):
        """回放的语义就是重复行，显式声明时不该被去重校验拦下。"""
        rows = _mix(23, 77) + _mix(0, 3)
        assert not any("重复 id" in e for e in validate(rows, allow_dups=True))

    def test_rejects_unresolvable_prefix(self):
        rows = _mix(23, 77) + [_row("bad", "中性", platform="xhs", ct="笔记")]
        assert any("前缀不可解析" in e for e in validate(rows, allow_dups=False))

    def test_rejects_illegal_label(self):
        rows = _mix(23, 76) + [_row("bad", "很负面")]
        assert any("final_label" in e for e in validate(rows, allow_dups=False))

    def test_rejects_illegal_emotion(self):
        rows = _mix(23, 76) + [_row("bad", "中性", emotion="暴走")]
        assert any("final_emotion" in e for e in validate(rows, allow_dups=False))

    def test_empty_mix_rejected(self):
        assert validate([], allow_dups=False) == ["混训集为空"]

    def test_band_constants_match_project_convention(self):
        assert (NEG_MIN_FRAC, NEG_MAX_FRAC) == (0.21, 0.26)
