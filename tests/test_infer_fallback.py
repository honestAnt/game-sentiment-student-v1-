# -*- coding: utf-8 -*-
"""apply_conf_fallback（批量/在线共用回退纯函数）与 prefix_enabled 的口径单测。"""

from infer_batch import apply_conf_fallback, fallback_skip_mask
from platform_input import prefix_enabled


class TestApplyConfFallback:
    def test_threshold_zero_disables_fallback(self):
        labels, confs = ["正面", "负面", "中性"], [0.1, 0.1, 0.1]
        final, mask = apply_conf_fallback(labels, confs, 0)
        assert final == labels and mask == [False] * 3

    def test_negative_threshold_disables_fallback(self):
        final, mask = apply_conf_fallback(["负面"], [0.2], -1)
        assert final == ["负面"] and mask == [False]

    def test_high_conf_kept(self):
        final, mask = apply_conf_fallback(["正面", "负面"], [0.99, 0.76], 0.75)
        assert final == ["正面", "负面"] and mask == [False, False]

    def test_low_conf_falls_back_to_neutral(self):
        final, mask = apply_conf_fallback(["正面"], [0.51], 0.75)
        assert final == ["中性"] and mask == [True]

    def test_low_conf_neutral_counts_hit_but_label_unchanged(self):
        # 中性命中回退条件：mask=True（口径与 v5 旧内联实现一致），标签不变
        final, mask = apply_conf_fallback(["中性"], [0.51], 0.75)
        assert final == ["中性"] and mask == [True]

    def test_neg_exempt_keeps_uncertain_negative(self):
        final, mask = apply_conf_fallback(["负面"], [0.51], 0.75, neg_exempt=True)
        assert final == ["负面"] and mask == [False]

    def test_neg_exempt_still_flips_non_negative(self):
        final, mask = apply_conf_fallback(
            ["正面", "负面", "中性"], [0.51, 0.51, 0.51], 0.75, neg_exempt=True
        )
        assert final == ["中性", "负面", "中性"]
        assert mask == [True, False, True]

    def test_boundary_equal_not_fallback(self):
        # conf == 阈值不回退（严格小于），与部署口径一致
        final, mask = apply_conf_fallback(["正面"], [0.75], 0.75)
        assert final == ["正面"] and mask == [False]

    def test_does_not_mutate_input(self):
        labels = ["正面"]
        apply_conf_fallback(labels, [0.1], 0.75)
        assert labels == ["正面"]

    def test_skip_keeps_low_conf_non_negative(self):
        final, mask = apply_conf_fallback(
            ["正面"], [0.51], 0.75, neg_exempt=True, skip=[True]
        )
        assert final == ["正面"] and mask == [False]

    def test_skip_false_still_falls_back(self):
        final, mask = apply_conf_fallback(
            ["正面"], [0.51], 0.75, neg_exempt=True, skip=[False]
        )
        assert final == ["中性"] and mask == [True]

    def test_skip_mixed_rows(self):
        final, mask = apply_conf_fallback(
            ["正面", "正面", "负面"],
            [0.51, 0.51, 0.51],
            0.75,
            neg_exempt=True,
            skip=[True, False, False],
        )
        assert final == ["正面", "中性", "负面"]
        assert mask == [False, True, False]

    def test_skip_length_mismatch_raises(self):
        try:
            apply_conf_fallback(["正面", "中性"], [0.5, 0.5], 0.75, skip=[True])
        except ValueError as e:
            assert "skip" in str(e)
        else:
            raise AssertionError("expected ValueError")


class TestFallbackSkipMask:
    def test_off_is_all_false(self):
        mask = fallback_skip_mask(
            ["weibo", "bilibili"], ["评论", "评论"], weibo_comment_raw=False
        )
        assert mask == [False, False]

    def test_on_skips_only_weibo_comments(self):
        mask = fallback_skip_mask(
            ["weibo", "weibo", "bilibili", "weibo"],
            ["评论", "博文", "评论", None],
            weibo_comment_raw=True,
        )
        assert mask == [True, False, False, False]

    def test_length_mismatch_raises(self):
        try:
            fallback_skip_mask(["weibo"], ["评论", "博文"], weibo_comment_raw=True)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


class TestPrefixEnabled:
    def test_forced_on_off(self):
        assert prefix_enabled("models/student_v5", "on") is True
        assert prefix_enabled("models/student_v8", "off") is False

    def test_auto_v8_plus_models(self):
        for d in (
            "models/student_v8",
            "models/student_v8b",
            "models/student_v9",
            "/abs/path/to/v8",
            "v8",
            "v10",
        ):
            assert prefix_enabled(d, "auto") is True

    def test_auto_legacy_models(self):
        for d in ("models/student_v5", "models/student_v7_1", "models/student_v4b", "student"):
            assert prefix_enabled(d, "auto") is False

    def test_auto_trailing_slash_normalized(self):
        assert prefix_enabled("models/student_v8/", "auto") is True
        assert prefix_enabled("models/student_v9/", "auto") is True
        assert prefix_enabled("models/student_v5/", "auto") is False


class TestPickDevice:
    def test_prefers_cuda_then_mps_then_cpu(self, monkeypatch):
        from infer_batch import pick_device
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert pick_device() == "cuda"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert pick_device() == "mps"
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert pick_device() == "cpu"
