# -*- coding: utf-8 -*-
import argparse
import json
import os
from collections import Counter

from config import CLASS_WARN_MIN, LABEL2ID, LABELED_DIR, MIN_TRAIN_LABELED, REVIEW_DIR


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
                raise ValueError(f"{path}:{i} JSON 无法解析: {e}")
    return rows


def validate_one_shard(candidate_path: str, labeled_path: str) -> list:
    errors = []
    if not os.path.exists(labeled_path):
        return [f"缺少标注文件 {labeled_path}"]
    try:
        cand = load_jsonl(candidate_path)
        lab = load_jsonl(labeled_path)
    except ValueError as e:
        return [str(e)]
    if len(cand) != len(lab):
        errors.append(f"行数不一致: 候选 {len(cand)} vs 标注 {len(lab)}")
    cand_ids = [r["id"] for r in cand]
    lab_ids = [r.get("id") for r in lab]
    if cand_ids != lab_ids:
        errors.append("id 顺序或集合与候选不一致")
    for i, (c, r) in enumerate(zip(cand, lab)):
        if r.get("text") != c.get("text"):
            errors.append(f"第{i}行 text 被改写 id={c['id']}")
        if r.get("final_label") not in LABEL2ID:
            errors.append(f"第{i}行 final_label 非法: {r.get('final_label')}")
    return errors


def merge_valid_shards(
    candidate_dir: str,
    labeled_dir: str,
    merged_path: str,
    review_dir: str,
    min_train: int = MIN_TRAIN_LABELED,
) -> int:
    os.makedirs(review_dir, exist_ok=True)
    os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
    shard_names = sorted(
        n for n in os.listdir(candidate_dir) if n.startswith("shard_") and n.endswith(".jsonl")
    )
    merged = []
    for name in shard_names:
        cpath = os.path.join(candidate_dir, name)
        lpath = os.path.join(labeled_dir, name)
        errs = validate_one_shard(cpath, lpath)
        if errs:
            os.makedirs(review_dir, exist_ok=True)
            if os.path.exists(lpath):
                with open(lpath, "r", encoding="utf-8") as src, open(
                    os.path.join(review_dir, name), "w", encoding="utf-8"
                ) as dst:
                    dst.write(src.read())
            with open(os.path.join(review_dir, name.replace(".jsonl", ".errors.txt")), "w", encoding="utf-8") as f:
                f.write("\n".join(errs) + "\n")
            print(f"[失败] {name}: {errs[0]}")
            continue
        merged.extend(load_jsonl(lpath))
        print(f"[通过] {name} {len(load_jsonl(lpath))} 条")
    counts = Counter(r["final_label"] for r in merged)
    print(f"[合并] {len(merged)} 条 分布 {dict(counts)}")
    for lab in LABEL2ID:
        n = counts.get(lab, 0)
        if n < CLASS_WARN_MIN:
            print(f"[警告] 类别 {lab} 仅 {n} 条 (<{CLASS_WARN_MIN})")
    if len(merged) < min_train:
        raise SystemExit(f"有效标注 {len(merged)} < {min_train}，拒绝开训")
    with open(merged_path, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[完成] 写入 {merged_path}")
    return len(merged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", default=os.path.join("data", "candidates", "round1"))
    parser.add_argument("--labeled-dir", default=os.path.join(LABELED_DIR, "round1"))
    parser.add_argument("--merged", default=os.path.join(LABELED_DIR, "round1_labeled.jsonl"))
    parser.add_argument("--review-dir", default=os.path.join(REVIEW_DIR, "round1"))
    args = parser.parse_args()
    merge_valid_shards(args.candidate_dir, args.labeled_dir, args.merged, args.review_dir)


if __name__ == "__main__":
    main()
