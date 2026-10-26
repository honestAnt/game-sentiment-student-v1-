# -*- coding: utf-8 -*-
"""gold_gates 纯函数口径单测：分片切分、指标、三片闸门比对。

口径回归锚点：v8@gold_eval_v6 的三片数字（评论 0.7406 / 博文 0.8304 / B站 0.9179）
是 v9 全系列消融的比对基准，这里用真实基线文件校验闸门判定不漂。
"""

import json
import os

import pytest

from gold_gates import (
    GATE_SLICES, GATES_V6, GATES_V7, Gate, check_gates, format_gate_report, metrics,
    slice_key, slice_metrics,
)

V8_BASELINE = os.path.join("data", "validation", "gold_eval_v6_v8_baseline.json")


class TestSliceKey:
    def test_explicit_content_type(self):
        assert slice_key({"platform": "weibo", "content_type": "评论"}) == "weibo/评论"
        assert slice_key({"platform": "bilibili", "content_type": "评论"}) == "bilibili/评论"

    def test_derives_from_source_table(self):
        row = {"platform": "weibo", "source_table": "dwd_crawl_weibo_post_details_merged"}
        assert slice_key(row) == "weibo/博文"

    def test_missing_content_type_becomes_none_slice(self):
        # 不静默归到某一片：显式 None，闸门比对时会因缺片报错
        assert slice_key({"platform": "weibo"}) == "weibo/None"


class TestMetrics:
    def test_all_correct(self):
        m = metrics(["负面", "中性", "正面"], ["负面", "中性", "正面"])
        assert m["acc"] == 1.0
        assert m["macro_f1"] == pytest.approx(1.0)
        assert m["neg_rec"] == 1.0 and m["pos_rec"] == 1.0

    def test_recall_none_when_label_absent_in_gold(self):
        m = metrics(["中性", "中性"], ["中性", "负面"])
        assert m["neg_rec"] is None and m["pos_rec"] is None
        assert m["neu_rec"] == 0.5

    def test_neg_recall_counts_only_gold_negatives(self):
        m = metrics(["负面", "负面", "中性"], ["负面", "中性", "负面"])
        assert m["neg_rec"] == pytest.approx(0.5)
        assert m["acc"] == pytest.approx(1 / 3)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            metrics(["中性"], ["中性", "负面"])

    def test_neg_precision_penalizes_over_predicting_negative(self):
        """富集集的核心指标：只看召回会鼓励乱判负面，必须同时量精确率。"""
        # 全判负面：召回满分，精确率只有 1/3
        m = metrics(["负面", "中性", "正面"], ["负面", "负面", "负面"])
        assert m["neg_rec"] == 1.0
        assert m["neg_prec"] == pytest.approx(1 / 3)

    def test_precision_none_when_class_never_predicted(self):
        m = metrics(["负面", "中性"], ["中性", "中性"])
        assert m["neg_prec"] is None and m["neg_rec"] == 0.0

    def test_n_neg_gold_exposes_recall_resolution(self):
        # gold_v6 的 B站片只有 17 条金负，1 条 = 5.9pt——分辨力要能一眼看到
        m = metrics(["负面"] * 17 + ["中性"] * 373, ["中性"] * 390)
        assert m["n_neg_gold"] == 17


class TestSliceMetrics:
    def test_splits_by_platform_and_content_type(self):
        records = [
            {"platform": "weibo", "content_type": "评论"},
            {"platform": "weibo", "content_type": "评论"},
            {"platform": "weibo", "content_type": "博文"},
            {"platform": "bilibili", "content_type": "评论"},
        ]
        golds = ["正面", "中性", "负面", "中性"]
        preds = ["正面", "负面", "负面", "中性"]
        out = slice_metrics(records, golds, preds)
        assert out["weibo/评论"]["n"] == 2 and out["weibo/评论"]["acc"] == 0.5
        assert out["weibo/博文"]["acc"] == 1.0
        assert out["bilibili/评论"]["acc"] == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            slice_metrics([{"platform": "weibo", "content_type": "评论"}], ["中性"], [])


class TestCheckGates:
    BASE = {
        "weibo/评论": {"acc": 0.7406, "n": 505},
        "weibo/博文": {"acc": 0.8304, "n": 401},
        "bilibili/评论": {"acc": 0.9179, "n": 390},
    }

    def _cand(self, comment, post, bili):
        return {
            "weibo/评论": {"acc": comment, "n": 505, "neg_rec": 0.75},
            "weibo/博文": {"acc": post, "n": 401, "neg_rec": 0.9},
            "bilibili/评论": {"acc": bili, "n": 390, "neg_rec": 0.2},
        }

    def test_all_slices_above_baseline_passes(self):
        # v9o 实测：三片全部 >= v8
        result = check_gates(self._cand(0.7604, 0.8329, 0.9231), self.BASE)
        assert result["passed"] is True
        assert all(r["passed"] for r in result["slices"])

    def test_single_slice_below_baseline_fails(self):
        # v9j 实测：评论/博文过，B站 −1.03pt 即整体 FAIL
        result = check_gates(self._cand(0.7604, 0.8329, 0.9077), self.BASE)
        assert result["passed"] is False
        failed = [r["slice"] for r in result["slices"] if not r["passed"]]
        assert failed == ["bilibili/评论"]

    def test_tie_counts_as_pass(self):
        # 闸门是「不低于」，持平算过（v9n 博文 ±0 视为 PASS）
        result = check_gates(self._cand(0.7604, 0.8304, 0.9231), self.BASE)
        assert result["passed"] is True

    def test_delta_is_candidate_minus_baseline(self):
        result = check_gates(self._cand(0.7604, 0.8329, 0.9231), self.BASE)
        deltas = {r["slice"]: r["delta"] for r in result["slices"]}
        assert deltas["weibo/评论"] == pytest.approx(0.0198, abs=1e-4)
        assert deltas["bilibili/评论"] == pytest.approx(0.0052, abs=1e-4)

    def test_missing_candidate_slice_raises_not_passes(self):
        cand = self._cand(0.76, 0.84, 0.93)
        del cand["bilibili/评论"]
        with pytest.raises(KeyError, match="bilibili/评论"):
            check_gates(cand, self.BASE)

    def test_missing_baseline_slice_raises(self):
        base = dict(self.BASE)
        del base["weibo/博文"]
        with pytest.raises(KeyError, match="weibo/博文"):
            check_gates(self._cand(0.76, 0.84, 0.93), base)


class TestGatesV7:
    """v7 闸门（2026-09-17 拍板）：B站 acc 降为护栏，负召回（富集集）升为主判据。"""

    BASE = {
        "gold": {
            "weibo/评论": {"acc": 0.7406, "n": 505, "neg_rec": 0.7439},
            "weibo/博文": {"acc": 0.8304, "n": 401, "neg_rec": 0.8824},
            "bilibili/评论": {"acc": 0.9179, "n": 690, "neg_rec": 0.2353},
        },
        "challenge": {"bilibili/评论": {"acc": 0.66, "n": 250, "neg_rec": 0.45}},
    }

    def _cand(self, wc_acc=0.76, wp_acc=0.84, wp_neg=0.90, bili_acc=0.9179, chl_neg=0.50):
        return {
            "gold": {
                "weibo/评论": {"acc": wc_acc, "n": 505, "neg_rec": 0.75},
                "weibo/博文": {"acc": wp_acc, "n": 401, "neg_rec": wp_neg},
                "bilibili/评论": {"acc": bili_acc, "n": 690, "neg_rec": 0.30},
            },
            "challenge": {"bilibili/评论": {"acc": 0.70, "n": 250, "neg_rec": chl_neg}},
        }

    def test_passes_when_weibo_up_and_bili_neg_rec_up(self):
        assert check_gates(self._cand(), self.BASE, GATES_V7)["passed"] is True

    def test_bili_acc_is_a_guardrail_not_a_blocker(self):
        """核心改动：B站 acc 掉 0.8pt 但负召回涨，应当 PASS。

        v6 闸门下这会 FAIL——正是它把 v9b（负召回 7/17 最高）挡在门外、
        放行了 v9o（负召回 3/17 垫底）。
        """
        result = check_gates(self._cand(bili_acc=0.9099, chl_neg=0.60), self.BASE, GATES_V7)
        assert result["passed"] is True
        # 同一组数字在 v6 闸门下会被 B站 acc 挡掉
        v6_result = check_gates(self._cand(bili_acc=0.9099)["gold"], self.BASE["gold"], GATES_V6)
        assert v6_result["passed"] is False

    def test_bili_acc_guardrail_still_blocks_big_drop(self):
        # 护栏是「不低于基线 1pt」，掉 1.8pt 仍须 FAIL
        result = check_gates(self._cand(bili_acc=0.8999, chl_neg=0.60), self.BASE, GATES_V7)
        assert result["passed"] is False
        failed = [r for r in result["slices"] if not r["passed"]]
        assert failed[0]["slice"] == "bilibili/评论" and failed[0]["metric"] == "acc"

    def test_guardrail_tolerance_boundary_is_inclusive(self):
        # 恰好等于 基线−1.0pt 视为通过
        result = check_gates(self._cand(bili_acc=self.BASE["gold"]["bilibili/评论"]["acc"] - 0.01,
                                        chl_neg=0.60), self.BASE, GATES_V7)
        assert result["passed"] is True

    def test_fails_when_bili_neg_rec_regresses(self):
        """v9o 的失败模式：acc 好看但负召回退步，新闸门必须拦下。"""
        result = check_gates(self._cand(bili_acc=0.9231, chl_neg=0.40), self.BASE, GATES_V7)
        assert result["passed"] is False
        failed = [r for r in result["slices"] if not r["passed"]]
        assert failed[0]["metric"] == "neg_rec" and failed[0]["source"] == "challenge"

    def test_post_neg_recall_floor_is_absolute_not_baseline(self):
        # 博文负召回是绝对下限 0.88，基线涨了也不跟着涨
        assert check_gates(self._cand(wp_neg=0.879), self.BASE, GATES_V7)["passed"] is False
        assert check_gates(self._cand(wp_neg=0.88), self.BASE, GATES_V7)["passed"] is True
        floor_row = [r for r in check_gates(self._cand(), self.BASE, GATES_V7)["slices"]
                     if r["metric"] == "neg_rec" and r["slice"] == "weibo/博文"][0]
        assert floor_row["baseline"] is None and floor_row["threshold"] == 0.88

    def test_weibo_slices_keep_strict_acc_gate(self):
        # 微博两片平庸基线低、信号强，仍是「不得低于基线」，无容差
        assert check_gates(self._cand(wc_acc=0.7405), self.BASE, GATES_V7)["passed"] is False
        assert check_gates(self._cand(wp_acc=0.8303), self.BASE, GATES_V7)["passed"] is False

    def test_missing_challenge_source_raises(self):
        cand = self._cand()
        del cand["challenge"]
        with pytest.raises(KeyError, match="challenge"):
            check_gates(cand, self.BASE, GATES_V7)

    def test_none_metric_raises_instead_of_silently_passing(self):
        # 富集集里若没有负面支撑，neg_rec=None，必须报错而不是当作通过
        cand = self._cand()
        cand["challenge"]["bilibili/评论"]["neg_rec"] = None
        with pytest.raises(ValueError, match="neg_rec"):
            check_gates(cand, self.BASE, GATES_V7)

    def test_flat_dict_still_accepted_for_v6_gates(self):
        # 向后兼容：扁平 {slice: metrics} 等价于只有 gold 一个来源
        flat_cand = self._cand()["gold"]
        assert check_gates(flat_cand, self.BASE["gold"], GATES_V6)["passed"] is True


class TestGateSpecDefinition:
    def test_v7_bili_acc_tolerance_is_one_point(self):
        g = [x for x in GATES_V7 if x.slice == "bilibili/评论" and x.metric == "acc"][0]
        assert g.tolerance == 0.01 and g.role == "护栏"

    def test_v7_bili_neg_rec_comes_from_challenge_set(self):
        g = [x for x in GATES_V7 if x.slice == "bilibili/评论" and x.metric == "neg_rec"][0]
        assert g.source == "challenge" and g.role == "主判据" and g.tolerance == 0

    def test_v6_gates_are_three_plain_acc_gates(self):
        assert len(GATES_V6) == 3
        assert all(g.metric == "acc" and g.tolerance == 0 and g.source == "gold"
                   for g in GATES_V6)
        assert tuple(g.slice for g in GATES_V6) == GATE_SLICES

    def test_gate_is_immutable(self):
        with pytest.raises(Exception):
            GATES_V7[0].tolerance = 0.5


class TestAgainstRealV8Baseline:
    @pytest.mark.skipif(not os.path.exists(V8_BASELINE), reason="v8 基线文件不在")
    def test_v8_vs_itself_is_exactly_tie_and_passes(self):
        baseline = json.load(open(V8_BASELINE, encoding="utf-8"))["slices_ex"]
        result = check_gates(baseline, baseline, GATES_V6)
        assert result["passed"] is True
        assert all(r["delta"] == 0 for r in result["slices"])

    @pytest.mark.skipif(not os.path.exists(V8_BASELINE), reason="v8 基线文件不在")
    def test_baseline_file_has_all_three_gate_slices(self):
        baseline = json.load(open(V8_BASELINE, encoding="utf-8"))["slices_ex"]
        assert set(GATE_SLICES) <= set(baseline)


class TestFormatGateReport:
    def test_marks_pass_and_fail_per_slice(self):
        result = check_gates(
            {
                "weibo/评论": {"acc": 0.76, "n": 505, "neg_rec": 0.75},
                "weibo/博文": {"acc": 0.83, "n": 401, "neg_rec": 0.9},
                "bilibili/评论": {"acc": 0.90, "n": 390, "neg_rec": 0.2},
            },
            TestCheckGates.BASE,
        )
        text = format_gate_report(result, {"acc": 0.825, "neg_rec": 0.77}, 0.8218)
        assert "3 条闸门: FAIL" in text
        assert "B站" in text and "FAIL" in text
        assert "overall" in text

    def test_v7_report_marks_source_and_guardrail_role(self):
        cand = {
            "gold": {
                "weibo/评论": {"acc": 0.77, "n": 505, "neg_rec": 0.75},
                "weibo/博文": {"acc": 0.84, "n": 401, "neg_rec": 0.90},
                "bilibili/评论": {"acc": 0.912, "n": 690, "neg_rec": 0.30},
            },
            "challenge": {"bilibili/评论": {"acc": 0.70, "n": 250, "neg_rec": 0.55}},
        }
        base = {
            "gold": {
                "weibo/评论": {"acc": 0.7406, "n": 505, "neg_rec": 0.74},
                "weibo/博文": {"acc": 0.8304, "n": 401, "neg_rec": 0.88},
                "bilibili/评论": {"acc": 0.9179, "n": 690, "neg_rec": 0.24},
            },
            "challenge": {"bilibili/评论": {"acc": 0.65, "n": 250, "neg_rec": 0.45}},
        }
        text = format_gate_report(check_gates(cand, base, GATES_V7))
        assert "@challenge" in text          # 标明该指标取自富集集
        assert "[护栏]" in text               # B站 acc 已降级为护栏
        assert "5 条闸门: PASS" in text
