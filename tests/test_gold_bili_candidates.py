# -*- coding: utf-8 -*-
"""B站金标扩样候选导出的纯函数单测：富集通道、长度过滤、配额均摊、上限采样。"""

import pandas as pd
import pytest

from export_gold_bili_candidates import (
    CHANNEL_QUOTA, MAX_TEXT_LEN, MIN_COMMENT_LEN, NEG_CHANNELS, allocate_by_channel,
    cap_and_sample, channel_quotas, match_channels, valid_text,
)


class TestMatchChannels:
    def test_protest_mobilization(self):
        # v8 抽检里被高置信判中性的抗议动员文体
        assert "抗议动员" in match_channels("抵制这个游戏，退游了")
        assert "抗议动员" in match_channels("已卸载，劝退各位")

    def test_accountability(self):
        assert "问责" in match_channels("策划出来给个说法")
        assert "问责" in match_channels("吃相太难看了，恰烂钱")

    def test_explicit_criticism(self):
        assert "显式批评" in match_channels("这优化真是垃圾")

    def test_euphemistic_criticism(self):
        # 不带脏字的委婉批评：v8 点名的另一类漏判
        assert "委婉批评" in match_channels("这转盘有点敷衍了")
        assert "委婉批评" in match_channels("内容一代不如一代")

    def test_neutral_text_hits_nothing(self):
        assert match_channels("今天天气不错，来看看新角色") == []
        assert match_channels("这个 BGM 叫什么名字") == []

    @pytest.mark.parametrize("text", [
        # 42 万条真实池审计出的「伪负面」：游戏语境中性高频词，命中即稀释富集集
        "原神退坑了，什么时候回坑",          # 坑：退坑/回坑/入坑
        "音乐应该是男声女声合一起",           # 一起
        "要是碰到这种你不炸了",              # 要是
        "本来看见奶奶还能绷住的",            # 本来
        "希望鸣潮把大世界升级一些",          # 希望（期待非批评）
        "建模做的越来越好了，加油哦",        # 越来越（接"好"）
        "可惜来晚了，上不了榜",              # 可惜（惋惜非批评）
        "我手法贼烂",                       # 烂（自嘲）
        "低命龙王可以用水冰队或者垃圾队",     # 垃圾队（配队术语，仍会被"垃圾"命中，见下）
        "还不如21温迪",                     # 还不如（比较句式）
    ])
    def test_neutral_gaming_idioms_not_matched(self, text):
        hit = match_channels(text)
        # "垃圾队" 属已知残留假阳性：显式批评通道保留"垃圾"的收益大于代价，
        # 人工标注时如实标中性即可，用于量负面精确率
        if "垃圾" in text:
            assert hit == ["显式批评"]
        else:
            assert hit == [], f"{text} 不应命中，实际 {hit}"

    def test_multi_channel_hit_is_recorded(self):
        chans = match_channels("策划垃圾，抵制")
        assert len(chans) >= 2
        assert set(chans) <= set(NEG_CHANNELS)

    def test_channel_order_is_stable_for_primary(self):
        # primary_channel 取第一个命中，顺序须稳定（依赖 dict 插入序）
        assert match_channels("策划垃圾，抵制")[0] == "抗议动员"


class TestChannelQuotas:
    def test_sums_to_total(self):
        for n in (250, 200, 137, 4):
            assert sum(channel_quotas(n).values()) == n

    def test_matches_declared_weights_at_250(self):
        # 声明权重 70/70/70/40 在 n=250 时应原样落地
        assert channel_quotas(250) == CHANNEL_QUOTA

    def test_euphemistic_channel_gets_smallest_share(self):
        q = channel_quotas(250)
        assert q["委婉批评"] == min(q.values())
        assert q["抗议动员"] > q["委婉批评"]


class TestValidText:
    def test_too_short_rejected(self):
        # 1-3 字人工也判不出口径，不浪费标注量
        assert not valid_text("草")
        assert not valid_text("哈哈哈")

    def test_boundary_lengths(self):
        assert valid_text("x" * MIN_COMMENT_LEN)
        assert valid_text("x" * MAX_TEXT_LEN)
        assert not valid_text("x" * (MIN_COMMENT_LEN - 1))
        assert not valid_text("x" * (MAX_TEXT_LEN + 1))


def _df(n, channel="显式批评", domain="异环", bvid_span=1, domain_span=1, tag=""):
    doms = ["异环", "原神", "绝区零", "鸣潮", "其他"]
    return pd.DataFrame([{
        "cid": f"{tag or channel}c{i}", "bvid": f"{tag or channel}bv{i % bvid_span}",
        "text": f"{tag or channel}t{i}",
        "domain": doms[i % domain_span] if domain_span > 1 else domain,
        "ctx": "", "channels": [channel], "primary_channel": channel,
    } for i in range(n)])


class TestCapAndSample:
    def test_per_video_cap_limits_output(self):
        # 100 条全来自同 1 个视频，单视频限 2 -> 最多 2 条
        df = _df(100, bvid_span=1)
        out = cap_and_sample(df, 50, max_per_video=2, max_per_domain=50, seed=1, label="t")
        assert len(out) == 2

    def test_per_domain_cap_limits_output(self):
        df = _df(100, bvid_span=100)  # 每条独立视频，视频上限不绑定
        out = cap_and_sample(df, 80, max_per_video=2, max_per_domain=30, seed=1, label="t")
        assert len(out) == 30

    def test_returns_requested_n_when_pool_ample(self):
        df = _df(200, bvid_span=200)
        out = cap_and_sample(df, 50, max_per_video=2, max_per_domain=500, seed=1, label="t")
        assert len(out) == 50

    def test_empty_pool_returns_empty(self):
        out = cap_and_sample(pd.DataFrame(), 10, 2, 50, 1, "t")
        assert out.empty

    def test_deterministic_for_same_seed(self):
        df = _df(200, bvid_span=200)
        a = cap_and_sample(df, 40, 2, 500, seed=7, label="t")
        b = cap_and_sample(df, 40, 2, 500, seed=7, label="t")
        assert list(a["cid"]) == list(b["cid"])


def _ample_pool(per_channel=400):
    """每通道充足、跨 5 域、每视频 1 条的池子（上限均不绑定）。"""
    return pd.concat([
        _df(per_channel, channel=ch, bvid_span=per_channel, domain_span=5, tag=f"{i}")
        for i, ch in enumerate(NEG_CHANNELS)
    ]).reset_index(drop=True)


class TestAllocateByChannel:
    def test_splits_by_declared_quota(self):
        out = allocate_by_channel(_ample_pool(), 250, seed=3)
        counts = out["primary_channel"].value_counts().to_dict()
        assert len(out) == 250
        assert counts == CHANNEL_QUOTA

    def test_sparse_channel_capped_by_its_pool_not_by_quota(self):
        # 抗议动员池只有 10 条、其余充足：按权重该给 70，实际只能给 10，缺额由别的通道补
        df = pd.concat([
            _df(500, channel="显式批评", bvid_span=500, domain_span=5, tag="a"),
            _df(10, channel="抗议动员", bvid_span=10, domain_span=5, tag="b"),
            _df(500, channel="问责", bvid_span=500, domain_span=5, tag="c"),
            _df(500, channel="委婉批评", bvid_span=500, domain_span=5, tag="d"),
        ]).reset_index(drop=True)
        out = allocate_by_channel(df, 250, seed=3)
        assert (out["primary_channel"] == "抗议动员").sum() == 10
        assert len(out) == 250

    def test_dense_channel_does_not_exceed_its_quota(self):
        # 回归：问责池比抗议动员大 2 倍，不能因此多占配额
        out = allocate_by_channel(_ample_pool(400), 250, seed=3)
        assert (out["primary_channel"] == "问责").sum() == CHANNEL_QUOTA["问责"]

    def test_domain_cap_applied_inside_channel_not_globally(self):
        """回归：全局先压域上限会让域分布决定通道构成（实测抗议动员被压到 39/250）。

        构造「抗议动员」全部集中在单一域：通道内域上限应只限制该通道自身，
        其他通道照常拿满，且总量靠补齐机制补回。
        """
        df = pd.concat([
            _df(400, channel="显式批评", bvid_span=400, domain_span=5, tag="a"),
            _df(400, channel="抗议动员", bvid_span=400, domain="异环", tag="b"),
            _df(400, channel="问责", bvid_span=400, domain_span=5, tag="c"),
            _df(400, channel="委婉批评", bvid_span=400, domain_span=5, tag="d"),
        ]).reset_index(drop=True)
        out = allocate_by_channel(df, 200, seed=3)
        counts = out["primary_channel"].value_counts()
        # 单域的抗议动员受通道内域上限（62*0.4≈25）限制，但不拖累其他通道
        assert counts.get("显式批评", 0) >= 45
        assert counts.get("问责", 0) >= 45
        assert len(out) == 200

    def test_shortfall_is_backfilled_to_full_quota(self):
        df = pd.concat([
            _df(500, channel="显式批评", bvid_span=500, domain_span=5, tag="a"),
            _df(10, channel="委婉批评", bvid_span=10, domain_span=5, tag="d"),
        ]).reset_index(drop=True)
        out = allocate_by_channel(df, 200, seed=3)
        assert len(out) == 200  # 缺额由富余通道补齐

    def test_cannot_exceed_pool(self):
        out = allocate_by_channel(_df(30, bvid_span=30, domain_span=5), 200, seed=3)
        assert len(out) == 30

    def test_no_duplicate_rows(self):
        out = allocate_by_channel(_ample_pool(), 250, seed=3)
        assert out["cid"].is_unique

    def test_video_budget_is_shared_across_channels(self):
        """回归：单视频预算必须跨通道共享，否则争议视频按通道数翻倍贡献。

        同一批 bvid 在 4 个通道里都出现（真实情况：一个争议视频的评论会同时命中
        抗议动员/问责/显式批评）。8 个视频 × 上限 2 = 全集最多 16 条。
        """
        df = pd.concat([
            _df(400, channel=ch, bvid_span=8, domain_span=5, tag=f"ch{i}")
            for i, ch in enumerate(NEG_CHANNELS)
        ]).reset_index(drop=True)
        df["bvid"] = df["bvid"].str.replace(r"^ch\d+", "shared", regex=True)
        out = allocate_by_channel(df, 200, seed=3, max_per_video=2)
        assert out["bvid"].value_counts().max() <= 2
        assert len(out) <= 8 * 2

    def test_per_video_cap_enforced_within_single_channel(self):
        df = _df(400, channel="显式批评", bvid_span=3, domain_span=5)
        out = allocate_by_channel(df, 200, seed=3, max_per_video=2)
        assert out["bvid"].value_counts().max() <= 2

    def test_empty_pool_returns_empty(self):
        assert allocate_by_channel(pd.DataFrame(), 10, seed=1).empty
