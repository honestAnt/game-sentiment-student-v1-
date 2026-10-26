# -*- coding: utf-8 -*-
"""把 GLM teacher 的逐条判定合并回候选分片，产出标注文件。

判定文件每行: {"id":..., "final_label":..., "final_emotion":..., "reason":...}
输出行: 候选原字段(id/text/platform/source_table/content_hash) + final_label/final_emotion/reason/teacher，
顺序与候选分片完全一致，text 原样带回。

用法:
    python apply_teacher_labels.py --shard 00 --judgments dev-task/judgments/round1/shard_00.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import EMOTION_TAGS, LABEL2ID  # noqa: E402

TEACHER = "glm-5.3"


def load_jsonl(path: str) -> list:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i} JSON 解析失败: {e}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, help="分片编号，如 00")
    ap.add_argument("--judgments", required=True, help="判定 jsonl 路径")
    ap.add_argument("--candidate-dir", default=os.path.join("data", "candidates", "round1"))
    ap.add_argument("--labeled-dir", default=os.path.join("data", "labeled", "round1"))
    args = ap.parse_args()

    name = f"shard_{args.shard}.jsonl"
    cand = load_jsonl(os.path.join(args.candidate_dir, name))
    judgs = load_jsonl(args.judgments)
    by_id = {}
    for j in judgs:
        if j["id"] in by_id:
            raise SystemExit(f"判定文件重复 id: {j['id']}")
        by_id[j["id"]] = j

    cand_ids = {c["id"] for c in cand}
    missing = [c["id"] for c in cand if c["id"] not in by_id]
    extra = sorted(set(by_id) - cand_ids)
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 条判定: {missing[:5]} ...")
    if extra:
        raise SystemExit(f"多余 {len(extra)} 条判定: {extra[:5]} ...")

    os.makedirs(args.labeled_dir, exist_ok=True)
    out_path = os.path.join(args.labeled_dir, name)
    counts = {}
    with open(out_path, "w", encoding="utf-8") as f:
        for c in cand:
            j = by_id[c["id"]]
            label = j["final_label"]
            emotion = j["final_emotion"]
            reason = j["reason"]
            if label not in LABEL2ID:
                raise SystemExit(f"{c['id']} final_label 非法: {label}")
            if emotion not in EMOTION_TAGS:
                raise SystemExit(f"{c['id']} final_emotion 非法: {emotion}")
            if len(reason) > 20:
                raise SystemExit(f"{c['id']} reason 超 20 字: {reason}")
            row = {
                "id": c["id"],
                "text": c["text"],
                "platform": c["platform"],
                "source_table": c["source_table"],
                "content_hash": c["content_hash"],
                "final_label": label,
                "final_emotion": emotion,
                "reason": reason,
                "teacher": TEACHER,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[label] = counts.get(label, 0) + 1
    print(f"[OK] {out_path} 共 {len(cand)} 行，分布 {counts}")


if __name__ == "__main__":
    main()
