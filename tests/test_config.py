import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def test_max_seq_len_is_256():
    assert config.MAX_SEQ_LEN == 256


def test_aliyun_mapping():
    assert config.ALIYUN_SENTIMENT_TO_LABEL == {1: "正面", 2: "中性", 3: "负面"}


def test_round1_math():
    assert config.ROUND1_SAMPLE_SIZE == config.ROUND1_N_SHARDS * config.ROUND1_SHARD_SIZE
    assert config.ROUND1_TARGET_COMMENTS == 1000
    assert config.ROUND1_TARGET_NEUTRALS == 800


def test_round6_math():
    assert config.ROUND6_SAMPLE_SIZE == config.ROUND6_N_SHARDS * config.ROUND6_SHARD_SIZE
    assert abs(sum(config.ROUND6_SLICE_FRAC.values()) - 1.0) < 1e-9
    assert config.ROUND6_SLICE_FRAC[("weibo", "评论")] == 0.50


def test_require_starrocks_env_rejects_missing(monkeypatch):
    for k in [
        "STARROCKS_HOST", "STARROCKS_PORT", "STARROCKS_USER",
        "STARROCKS_PASSWORD", "STARROCKS_DATABASE",
    ]:
        monkeypatch.delenv(k, raising=False)
    try:
        config.require_starrocks_env()
        assert False, "should have SystemExit"
    except SystemExit as e:
        assert "STARROCKS_HOST" in str(e)


def test_require_starrocks_env_ok(monkeypatch):
    monkeypatch.setenv("STARROCKS_HOST", "127.0.0.1")
    monkeypatch.setenv("STARROCKS_PORT", "9030")
    monkeypatch.setenv("STARROCKS_USER", "u")
    monkeypatch.setenv("STARROCKS_PASSWORD", "secret")
    monkeypatch.setenv("STARROCKS_DATABASE", "example_sentiment")
    cfg = config.require_starrocks_env()
    assert cfg["host"] == "127.0.0.1"
    assert cfg["port"] == 9030
    assert cfg["password"] == "secret"


def test_mps_16g_uses_tiny_microbatch_and_keeps_effective_64():
    assert config.TRAIN_BATCH_SIZE == 64
    assert config.MPS_TRAIN_BATCH_SIZE == 2
    assert config.MPS_EVAL_BATCH_SIZE == 4
    assert config.MPS_INFER_BATCH_SIZE == 8
    assert config.MPS_HIGH_WATERMARK_RATIO == 0.0  # unlimited；有限水位在本机训练会 OOM
    kw = config.mps_training_arg_overrides()
    assert kw["per_device_train_batch_size"] == 2
    assert kw["per_device_eval_batch_size"] == 4
    assert kw["gradient_accumulation_steps"] == 32
    assert kw["dataloader_num_workers"] == 0
    assert "use_mps_device" not in kw


def test_default_infer_batch_size_by_device():
    assert config.default_infer_batch_size("cuda") == 256
    assert config.default_infer_batch_size("mps") == 8
    assert config.default_infer_batch_size("cpu") == 16
