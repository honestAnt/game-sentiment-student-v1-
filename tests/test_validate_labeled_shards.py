import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validate_labeled_shards import merge_valid_shards, validate_one_shard

CAND = {
    "id": "a" * 32,
    "text": "这游戏真好玩",
    "platform": "weibo",
    "source_table": "dwd_crawl_weibo_post_details_merged",
    "content_hash": "a" * 32,
}


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_validate_ok():
    with tempfile.TemporaryDirectory() as td:
        c = os.path.join(td, "c.jsonl")
        l = os.path.join(td, "l.jsonl")
        labeled = dict(CAND, final_label="正面", final_emotion="惊喜", reason="喜欢", teacher="glm-5.3")
        _write(c, [CAND])
        _write(l, [labeled])
        assert validate_one_shard(c, l) == []


def test_validate_rejects_rewritten_text():
    with tempfile.TemporaryDirectory() as td:
        c = os.path.join(td, "c.jsonl")
        l = os.path.join(td, "l.jsonl")
        labeled = dict(CAND, text="被改写了", final_label="正面", final_emotion="其他", reason="x", teacher="glm-5.3")
        _write(c, [CAND])
        _write(l, [labeled])
        errs = validate_one_shard(c, l)
        assert any("text" in e for e in errs)


def test_validate_rejects_bad_label():
    with tempfile.TemporaryDirectory() as td:
        c = os.path.join(td, "c.jsonl")
        l = os.path.join(td, "l.jsonl")
        labeled = dict(CAND, final_label="正向", final_emotion="其他", reason="x", teacher="glm-5.3")
        _write(c, [CAND])
        _write(l, [labeled])
        errs = validate_one_shard(c, l)
        assert any("final_label" in e for e in errs)


def test_merge_moves_failures_and_refuses_below_min():
    with tempfile.TemporaryDirectory() as td:
        cand_dir = os.path.join(td, "cand")
        lab_dir = os.path.join(td, "lab")
        review = os.path.join(td, "review")
        merged = os.path.join(td, "merged.jsonl")
        os.makedirs(cand_dir)
        os.makedirs(lab_dir)
        ok = dict(CAND, final_label="负面", final_emotion="失望", reason="贵", teacher="glm-5.3")
        _write(os.path.join(cand_dir, "shard_00.jsonl"), [CAND])
        _write(os.path.join(lab_dir, "shard_00.jsonl"), [ok])
        bad_c = dict(CAND, id="b" * 32, content_hash="b" * 32, text="另一条")
        _write(os.path.join(cand_dir, "shard_01.jsonl"), [bad_c])
        _write(os.path.join(lab_dir, "shard_01.jsonl"), [dict(bad_c, text="改了", final_label="中性", final_emotion="其他", reason="x", teacher="glm-5.3")])
        try:
            merge_valid_shards(cand_dir, lab_dir, merged, review, min_train=2000)
            assert False, "expected SystemExit"
        except SystemExit:
            pass
        assert os.path.exists(os.path.join(review, "shard_01.jsonl"))
        assert os.path.exists(os.path.join(review, "shard_01.errors.txt"))
