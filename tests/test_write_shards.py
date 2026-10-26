import json
import os
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMMENT_SOURCE, POST_SOURCE
from export_starrocks_candidates import candidate_record, write_round1_outputs
from sampling import prepare_pool


def test_candidate_record_strips_aliyun():
    row = {
        "id": "a" * 32,
        "text": "你好世界",
        "platform": "weibo",
        "source_table": POST_SOURCE,
        "content_hash": "a" * 32,
        "aliyun_final": 3,
        "positive_prob_aliyun": 0.1,
        "negative_prob_aliyun": 0.8,
    }
    rec = candidate_record(row)
    assert set(rec.keys()) == {"id", "text", "platform", "source_table", "content_hash"}
    assert "aliyun" not in json.dumps(rec)


def test_write_round1_outputs_shards_and_sidecar():
    rows = []
    for i in range(20):
        src = COMMENT_SOURCE if i < 8 else POST_SOURCE
        rows.append({
            "id": f"{i:032d}",
            "text": f"文本{i:04d}内容",
            "platform": "weibo",
            "source_table": src,
            "content_hash": f"{i:032d}",
            "aliyun_final": 1 if i % 3 else 3,
            "positive_prob_aliyun": 0.2,
            "negative_prob_aliyun": 0.7,
        })
    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as td:
        write_round1_outputs(df, td, shard_size=10)
        names = sorted(os.listdir(td))
        assert "aliyun_strata_sidecar.jsonl" in names
        shards = [n for n in names if n.startswith("shard_")]
        assert shards == ["shard_00.jsonl", "shard_01.jsonl"]
        with open(os.path.join(td, "shard_00.jsonl"), encoding="utf-8") as f:
            lines = [json.loads(x) for x in f]
        assert len(lines) == 10
        assert set(lines[0].keys()) == {"id", "text", "platform", "source_table", "content_hash"}
        with open(os.path.join(td, "aliyun_strata_sidecar.jsonl"), encoding="utf-8") as f:
            side = [json.loads(x) for x in f]
        assert len(side) == 20
        assert "aliyun_final" in side[0]
