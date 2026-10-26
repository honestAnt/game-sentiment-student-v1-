import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_teacher_aliyun import compare_frames, load_sidecar, load_labeled


def test_compare_perfect_agreement():
    with tempfile.TemporaryDirectory() as td:
        side = os.path.join(td, "side.jsonl")
        lab = os.path.join(td, "lab.jsonl")
        with open(side, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "1", "aliyun_final": 3}) + "\n")
        with open(lab, "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "1", "final_label": "负面"}) + "\n")
        stats = compare_frames(load_sidecar(side), load_labeled(lab))
        assert stats["n"] == 1
        assert stats["agree"] == 1.0
