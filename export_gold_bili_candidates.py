# -*- coding: utf-8 -*-
"""
export_gold_bili_candidates.py
导出 B站评论金标扩样候选，交人工标注。对应 docs/v10训练方案-先修尺子-2026-09-17.md 步骤 A。

为什么要扩：gold_eval_v6 的 B站片 n=390 里中性 357（91.5%）、负面仅 17。
「全判中性」平庸基线就有 0.9154，v8 只比它多对 1 条——该片 acc 闸门分辨力约等于 0，
且与「抓负面」目标反向（负召回最高的 v9b 反而 acc 最低被 FAIL 掉）。
负召回分母 17 条时 1 条 = 5.9pt，无法判定任何真实改进。

产出两份**互不混用**的集合：

  A. iid  —— 纯随机（延续 gold_v6 的 max_per_post=2 / max_per_domain=50 协议）
             用途：量分布真实下的 acc，可与 v6 的 B站片数字续接。
  B. challenge —— 抗议/问责/委婉批评文体定向富集
             用途：**只量负召回与负面精确率**，acc 不可与 iid/v6 横向比较。

富集通道**刻意不使用任何模型分数**（不用 v8 低置信、不用 v8 负概率）。
用模型不确定性挑样本会让后续模型在这个集合上天然占便宜，
尺子就不公正了；文体/词汇通道对 v8 与 v10 是同一分布。

标注口径：与既有金标一致，人工在「人工标注-杠」列填 中立/负向/正向（或 中性/负面/正面）。
challenge 集里必然混有大量「带负面词但实为玩梗/中性」的样本，这是有意的——
没有这些就无法量负面精确率，只量到召回会鼓励模型乱判负面。

用法：
    python export_gold_bili_candidates.py                    # iid 300 + challenge 250
    python export_gold_bili_candidates.py --n_iid 300 --n_challenge 250

产出：data/candidates/gold_expansion_bili/
    bili_iid_candidates_<n>.xlsx / .jsonl          （id 前缀 ge7b_）
    bili_challenge_candidates_<n>.xlsx / .jsonl    （id 前缀 ge7c_，jsonl 带 channel 字段）
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from export_gold_starrocks_weibo_comments import (
    CONTEXT_TRUNC, HUMAN_LABEL_COL, ID_COL, TEXT_COL, TYPE_COL,
    connect, load_env_file,
)
from export_starrocks_iid_spotcheck import (
    BILI_COMMENTS_SQL, assign_domain, load_excluded,
)

OUT_DIR = os.path.join("data", "candidates", "gold_expansion_bili")
CONTEXT_COL = "视频标题（评论的上下文）"
CHANNEL_COL = "召回通道"

MIN_COMMENT_LEN = 4      # 比抽检的 2 字更严：1-3 字（"草"/"哈哈"）人工也判不出，白耗标注量
MAX_TEXT_LEN = 300
IID_MAX_PER_DOMAIN = 50  # 延续 gold_v6 扩样协议
IID_MAX_PER_VIDEO = 2
CHL_MAX_PER_VIDEO = 2
CHL_DOMAIN_SHARE = 0.4   # 通道内单域上限 = 配额 × 该比例，强制每通道跨 ~3 个域
MIN_CHANNEL_DOMAIN_CAP = 10

# 富集通道：全部为文体/词汇规则，不含任何模型分数。
# 分通道记录是为了标完后能分别看「显式辱骂」与「委婉批评」的召回差异——
# v8 抽检显示前者能抓、后者漏，混在一起看不出来。
#
# 词表经过 42 万条真实池的逐 token 审计（logs/export_gold_bili_candidates.log）。
# 被剔除的高频「伪负面」词，全部是游戏语境里的中性高频表达，留着会把富集集稀释成随机集：
#   坑(2708) 退坑/回坑/入坑是中性游戏词 · 一起(2692)/联合 纯连词 · 要是(2454)/本来(1759) 话语标记
#   希望(2453) 多为期待非批评 · 越来越(886) 常接"越来越好" · 可惜(1204) 多为对剧情的惋惜
#   烂(1210) 含"手法烂"自嘲 · 屎(461) 含"红屎"BOSS 外号 · 还不如 是比较句式（"还不如21温迪"）
NEG_CHANNELS = {
    # 抗议/动员：v8 的核心盲区（抽检里抗议文体高置信判中性 9/11）
    "抗议动员": r"抵制|退游|脱坑|卸载|差评|举报|维权|劝退|别玩|不玩了|倒闭|跑路|退款|不再玩",
    # 问责：指向官方/策划的责任追究。策划(2670)/补偿(1989) 实测语境几乎全为投诉
    "问责": r"策划|补偿|给个说法|恰烂钱|骗氪|吃相|割韭菜|黑心|诈骗|客服|官方.{0,6}(管|回应|态度|解决)",
    # 显式批评/辱骂
    "显式批评": r"垃圾|辣鸡|恶心|离谱|失望|拉胯|摆烂|崩了|难玩|气死|太差|很差|差劲|烂尾|难看死",
    # 委婉批评：v8 抽检点名的另一类漏判，不带脏字。关键词天然圈不准，
    # 故只保留高精度核并给小配额——它的价值是诊断「非显式批评能否抓到」，不是凑量
    "委婉批评": (r"越来越(差|拉|水|烂|敷衍)|不如以前|不如从前|一代不如|有点失望|挺失望|"
                r"差点意思|敷衍|划水|缩水|没诚意|活不长|不太行|越做越"),
}

# 通道配额权重：前三个通道负面密度实测 55-65%，委婉批评约 30-40%，按价值分配而非均摊
CHANNEL_QUOTA = {"抗议动员": 70, "问责": 70, "显式批评": 70, "委婉批评": 40}


def match_channels(text: str) -> list[str]:
    """文本命中的富集通道（可多通道命中）。"""
    return [name for name, pat in NEG_CHANNELS.items() if re.search(pat, text)]


def valid_text(text: str) -> bool:
    return MIN_COMMENT_LEN <= len(text) <= MAX_TEXT_LEN


def cap_and_sample(df, n, max_per_video, max_per_domain, seed, label):
    """单视频上限 -> 单域上限 -> 随机取 n；少抽时明确告警而不是静默少给。"""
    if df.empty:
        print(f"  [警告] {label} 候选池为空")
        return df
    df = df.copy()
    df["_r"] = np.random.RandomState(seed).rand(len(df))
    df = df.sort_values(["bvid", "_r"]).groupby("bvid", group_keys=False).head(max_per_video)
    n_after_video = len(df)
    df = df.sort_values(["domain", "_r"]).groupby("domain", group_keys=False).head(max_per_domain)
    sampled = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    if len(sampled) < n:
        print(f"  [警告] {label} 只抽到 {len(sampled)}/{n} 条："
              f"单视频限 {max_per_video} 后 {n_after_video}，"
              f"单域限 {max_per_domain} 后 {len(df)}。需要足量请放宽上限。")
    return sampled


def take_greedy(pool, n, video_counts, max_per_video, max_per_domain, seed):
    """在**跨通道共享**的单视频预算下贪心取 n 条，同时施加单域上限。

    单视频预算必须全局共享：一个争议视频的评论会同时命中抗议动员/问责/显式批评，
    各通道各算各的上限就会让这一个视频贡献 3-4 倍配额，challenge 集退化成单事件样本。
    """
    empty = pool.head(0)
    if pool.empty or n <= 0:
        return empty
    shuffled = pool.sample(frac=1, random_state=seed)
    domain_counts, keep = {}, []
    for idx, row in shuffled.iterrows():
        if len(keep) >= n:
            break
        if video_counts.get(row["bvid"], 0) >= max_per_video:
            continue
        if domain_counts.get(row["domain"], 0) >= max_per_domain:
            continue
        keep.append(idx)
        video_counts[row["bvid"]] = video_counts.get(row["bvid"], 0) + 1
        domain_counts[row["domain"]] = domain_counts.get(row["domain"], 0) + 1
    return shuffled.loc[keep]


def channel_quotas(n: int) -> dict:
    """把总量 n 按 CHANNEL_QUOTA 权重分到各通道，余数补给权重最大的通道。"""
    total = sum(CHANNEL_QUOTA.values())
    quotas = {ch: n * w // total for ch, w in CHANNEL_QUOTA.items()}
    top = max(CHANNEL_QUOTA, key=lambda c: CHANNEL_QUOTA[c])
    quotas[top] += n - sum(quotas.values())
    return quotas


def allocate_by_channel(df, n, seed, max_per_video=CHL_MAX_PER_VIDEO):
    """按 CHANNEL_QUOTA 权重分配配额，不足的通道把余量让给其他通道。

    按权重而非按池子占比，是为了保证「抗议动员」这个稀疏但关键的通道
    不被「问责」（池子大 2 倍）淹没——它正是 v8 的漏判类型，样本太少就量不出来。

    上限在通道内部施加（单视频预算跨通道共享）：先全局压上限再分配，
    会让域分布决定通道构成（实测抗议动员被压到 39/250），与配额意图相反。
    """
    if df.empty:
        return df
    quotas = channel_quotas(n)
    channels = list(NEG_CHANNELS)
    video_counts: dict = {}
    picked, used_cid = [], set()
    remaining = n
    for ch in channels:
        i = channels.index(ch)
        quota = min(quotas[ch], remaining)
        pool = df[(df["primary_channel"] == ch) & (~df["cid"].isin(used_cid))]
        # 通道内单域上限按配额定，强制每个通道至少跨 ~3 个域，避免单个游戏的争议主导该通道
        per_domain = max(MIN_CHANNEL_DOMAIN_CAP, int(quota * CHL_DOMAIN_SHARE))
        take = take_greedy(pool, quota, video_counts, max_per_video, per_domain, seed + i)
        if len(take) < quota:
            print(f"  [警告] 通道「{ch}」只抽到 {len(take)}/{quota} 条"
                  f"（单视频限 {max_per_video}、单域限 {per_domain}），缺额由其他通道补齐")
        picked.append(take)
        used_cid |= set(take["cid"])
        remaining -= len(take)
    if remaining > 0:  # 有通道不足额，用剩余池补齐（不再卡域上限，但仍守单视频预算）
        rest = df[~df["cid"].isin(used_cid)]
        back = take_greedy(rest, remaining, video_counts, max_per_video, remaining or 1, seed)
        if len(back) < remaining:
            print(f"  [警告] 补齐仅 {len(back)}/{remaining} 条，总量将少于目标")
        picked.append(back)
    return pd.concat([p for p in picked if not p.empty]).reset_index(drop=True)


def write_set(rows, out_stem, id_prefix, with_channel):
    xlsx_out, jsonl_out = f"{out_stem}.xlsx", f"{out_stem}.jsonl"
    export = pd.DataFrame({
        ID_COL: rows["cid"],
        TYPE_COL: "评论",
        CONTEXT_COL: rows["ctx"],
        TEXT_COL: rows["text"],
    })
    if with_channel:
        export[CHANNEL_COL] = rows["primary_channel"]
    export[HUMAN_LABEL_COL] = None
    export.to_excel(xlsx_out, index=False)

    with open(jsonl_out, "w", encoding="utf-8") as f:
        for _, row in rows.iterrows():
            rec = {
                "id": f"{id_prefix}{row['cid']}", "text": row["text"],
                "platform": "bilibili", "content_type": "评论",
                "domain": row["domain"], "context": row["ctx"],
            }
            if with_channel:
                rec["channel"] = row["primary_channel"]
                rec["all_channels"] = row["channels"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return xlsx_out, jsonl_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_iid", type=int, default=300)
    parser.add_argument("--n_challenge", type=int, default=250)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--out_dir", default=OUT_DIR)
    args = parser.parse_args()

    load_env_file()
    excluded = load_excluded()
    # 本目录下已导出的候选也要排除（防重复导出给人标两次）
    for f in glob.glob(os.path.join(args.out_dir, "*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    excluded.add(json.loads(line)["text"])
    print(f"[排除集] 已用文本 {len(excluded)} 条")

    conn = connect()
    cur = conn.cursor()
    cur.execute(BILI_COMMENTS_SQL)
    raw = cur.fetchall()
    conn.close()
    print(f"[拉取] B站 tier1 有效评论 {len(raw)} 条")

    recs, seen = [], set()
    for cid, text, title, topics in raw:
        text = str(text or "").strip()
        if not valid_text(text) or text in excluded or text in seen:
            continue
        seen.add(text)
        title, topics = str(title or ""), str(topics or "")
        channels = match_channels(text)
        recs.append({
            "cid": str(cid), "bvid": f"bv_{abs(hash(title)) & 0xffffff}",
            "text": text, "domain": assign_domain(title, topics),
            "ctx": title[:CONTEXT_TRUNC],
            "channels": channels,
            "primary_channel": channels[0] if channels else "",
        })
    df = pd.DataFrame(recs)
    print(f"[过滤] 去重+排除+长度({MIN_COMMENT_LEN}-{MAX_TEXT_LEN}字)后 {len(df)} 条")
    print(f"[域构成] {df['domain'].value_counts().to_dict()}")
    hit = df[df["primary_channel"] != ""]
    print(f"[富集通道] 命中 {len(hit)} 条 ({len(hit)/len(df):.1%}): "
          f"{hit['primary_channel'].value_counts().to_dict()}")

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[A. iid 随机集] 目标 {args.n_iid} 条")
    iid = cap_and_sample(df, args.n_iid, IID_MAX_PER_VIDEO, IID_MAX_PER_DOMAIN,
                         args.seed, "iid")
    print(f"  域构成: {iid['domain'].value_counts().to_dict()}")
    iid_x, iid_j = write_set(iid, os.path.join(args.out_dir, f"bili_iid_candidates_{len(iid)}"),
                             "ge7b_", with_channel=False)

    print(f"\n[B. challenge 富集集] 目标 {args.n_challenge} 条")
    # 与 iid 集零重叠：同一条不能既算分布样本又算挑战样本
    pool = hit[~hit["cid"].isin(set(iid["cid"]))]
    chl = allocate_by_channel(pool, args.n_challenge, args.seed)
    print(f"  实得 {len(chl)} 条；通道构成: {chl['primary_channel'].value_counts().to_dict()}")
    print(f"  域构成: {chl['domain'].value_counts().to_dict()}")
    chl_x, chl_j = write_set(chl, os.path.join(args.out_dir, f"bili_challenge_candidates_{len(chl)}"),
                             "ge7c_", with_channel=True)

    overlap = set(iid["cid"]) & set(chl["cid"])
    assert not overlap, f"两集合重叠 {len(overlap)} 条"
    all_text = list(iid["text"]) + list(chl["text"])
    assert len(all_text) == len(set(all_text)), "文本重复"
    assert not (set(all_text) & excluded), "与排除集重合"

    print(f"\n[完成] 共 {len(iid) + len(chl)} 条待人工标注")
    print(f"  A iid       {len(iid):>4} 条  {iid_x}")
    print(f"  B challenge {len(chl):>4} 条  {chl_x}")
    print(f"  存档        {iid_j} / {chl_j}")
    print("\n[标注说明]")
    print("  · 在「人工标注-杠」列填：中立/中性、负向/负面、正向/正面（回收脚本会归一化）")
    print("  · 两份表**分开标、分开回收**，challenge 集的 acc 不可与 iid 集比较")
    print("  · challenge 集里带负面词但实为玩梗/中性的，请如实标中性——"
          "这些样本用于量负面精确率，缺了会鼓励模型乱判负面")


if __name__ == "__main__":
    main()
