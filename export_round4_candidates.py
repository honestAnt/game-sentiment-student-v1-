# -*- coding: utf-8 -*-
"""round4 候选导出：多游戏微博（双盲2000未标注行）+ B站定向难例。

与 round3 的区别（v6 失败归因的直接修复）：
- 数据源不再是 LoL 单域：多游戏片来自双盲2000（与金标博文同源同分布，异环/鸣潮等），
  修复「round3_full 全 LoL 而金标多游戏」的域偏移
- B站片不再随机，按 v5 两次抽检确认的弱点定向挖：直球应援判中性（应正面）、
  黑话反讽判正面（应负面），关键词 + agent-v5 分歧分层选取
- 采样逻辑在 sampling.build_round4_multigame_sample / build_round4_bili_hard_sample
"""
import argparse
import json
import os

import pandas as pd

from config import (
    CANDIDATE_DIR,
    LABELED_DIR,
    RAW_DATA_DIR,
    ROUND4_BILI_HARD_TARGET,
    ROUND4_BILI_1600_XLSX,
    ROUND4_BLIND_XLSX,
    ROUND4_SEED,
    ROUND4_SHARD_SIZE,
)
from export_starrocks_candidates import candidate_record
from sampling import (
    build_round4_bili_hard_sample,
    build_round4_multigame_sample,
)

# 文本排除（多游戏片用）：全部已训练文本 + 全部金标/抽检文本
TEXT_EXCLUDE_FILES = (
    os.path.join(LABELED_DIR, "v6_train_mixed.jsonl"),        # v5/v6 训练全集的超集
    os.path.join(LABELED_DIR, "gold_new_train.jsonl"),
    os.path.join("data", "validation", "gold_eval_v6.jsonl"),
    os.path.join("data", "validation", "gold_eval_v7.jsonl"),
    os.path.join("data", "validation", "gold_bili_neg_challenge.jsonl"),
    os.path.join("data", "validation", "validation_gold.jsonl"),
    os.path.join("data", "validation", "v5_spotcheck_raw400.jsonl"),
    os.path.join("data", "validation", "v5_spotcheck_raw200_teacher.jsonl"),
    os.path.join("data", "validation", "v5_spotcheck_raw200.jsonl"),
)
# id 排除（B站片用）：与文本排除同源
ID_EXCLUDE_FILES = TEXT_EXCLUDE_FILES


def load_texts_and_ids(paths) -> tuple:
    texts, ids = set(), set()
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"排除文件不存在: {p}")
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                texts.add(r["text"])
                ids.add(str(r["id"]))
    return texts, ids


def load_blind_frame(raw_dir: str) -> pd.DataFrame:
    df = pd.read_excel(os.path.join(raw_dir, ROUND4_BLIND_XLSX))
    return pd.DataFrame({
        "seq": df["序号"].astype(str),
        "type": df["类型"].astype(str),
        "text": df["文本"].astype(str).str.strip(),
        "human_label": df["人工标注-杠"],
    })


def load_bili_agent_frame(raw_dir: str) -> pd.DataFrame:
    df = pd.read_excel(os.path.join(raw_dir, ROUND4_BILI_1600_XLSX))
    return pd.DataFrame({
        "id": df["id"].astype(str),
        "text": df["评论文本"].astype(str).str.strip(),
        "agent_label": df["agent标注"].astype(str),
    })


def load_bili_score_map(pool_path: str) -> dict:
    score_map = {}
    with open(pool_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("platform") == "bilibili":
                score_map[str(r["id"])] = r
    return score_map


def write_round4_outputs(sample_df: pd.DataFrame, out_dir: str, shard_size: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    n = len(sample_df)
    n_shards = (n + shard_size - 1) // shard_size
    for i in range(n_shards):
        chunk = sample_df.iloc[i * shard_size:(i + 1) * shard_size]
        with open(os.path.join(out_dir, f"shard_{i:02d}.jsonl"), "w", encoding="utf-8") as f:
            for _, row in chunk.iterrows():
                f.write(json.dumps(candidate_record(row), ensure_ascii=False) + "\n")
    print(f"[导出] {n} 条 -> {n_shards} 个分片，目录 {out_dir}")
    return n_shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=RAW_DATA_DIR)
    parser.add_argument("--pool-scored", default=os.path.join(CANDIDATE_DIR, "round3_full", "pool_scored.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(CANDIDATE_DIR, "round4"))
    parser.add_argument("--bili-target", type=int, default=ROUND4_BILI_HARD_TARGET)
    args = parser.parse_args()

    exclude_texts, exclude_ids = load_texts_and_ids(TEXT_EXCLUDE_FILES)
    print(f"[排除] 已训练/金标/抽检：文本 {len(exclude_texts)} 条，id {len(exclude_ids)} 个")

    blind = build_round4_multigame_sample(load_blind_frame(args.raw_dir), exclude_texts, seed=ROUND4_SEED)
    comp = blind.groupby("content_type").size().to_dict()
    print(f"[多游戏] 双盲2000未标注且未重合 {len(blind)} 条，构成 {comp}")

    score_map = load_bili_score_map(args.pool_scored)
    bili_src = load_bili_agent_frame(args.raw_dir)
    bili_src = bili_src[~bili_src["text"].isin(exclude_texts)]  # 跨源同文不同 id，文本也要排
    bili = build_round4_bili_hard_sample(
        bili_src, score_map, exclude_ids,
        n_target=args.bili_target, seed=ROUND4_SEED,
    )
    print(f"[B站难例] 定向 {len(bili)} 条")

    sample = pd.concat([blind, bili], ignore_index=True)
    overlap_texts = sample["text"].isin(exclude_texts).sum()
    if overlap_texts:
        raise SystemExit(f"采样后仍与排除集文本重合 {overlap_texts} 条")
    if sample["content_hash"].duplicated().any():
        raise SystemExit("采样内部存在重复文本")
    sample = sample.sample(frac=1, random_state=ROUND4_SEED).reset_index(drop=True)
    print(f"[合计] {len(sample)} 条 "
          f"({(sample['platform'] == 'weibo').sum()} weibo / {(sample['platform'] == 'bilibili').sum()} bilibili)")
    write_round4_outputs(sample, args.output_dir, ROUND4_SHARD_SIZE)


if __name__ == "__main__":
    main()
