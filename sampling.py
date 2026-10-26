# -*- coding: utf-8 -*-
import hashlib

import numpy as np
import pandas as pd

from config import (
    COMMENT_SOURCE,
    LOW_CONFIDENCE_THRESHOLD,
    POST_SOURCE,
    ROUND1_SAMPLE_SIZE,
    ROUND1_SEED,
    ROUND1_TARGET_COMMENTS,
    ROUND1_TARGET_NEUTRALS,
    ROUND2_HARD_FRAC,
    ROUND2_MAX_POST_FRAC,
    ROUND2_P_NEG_HARD,
    ROUND2_POST_SOURCE,
    ROUND2_SAMPLE_SIZE,
    ROUND2_SEED,
    ROUND3_HARD_FRAC,
    ROUND3_NEG_MAX_FRAC,
    ROUND3_SAMPLE_SIZE,
    ROUND3_SEED,
    ROUND3_SLICE_FRAC,
    ROUND4_BILI_HARD_TARGET,
    ROUND4_CHEER_PATTERN,
    ROUND4_SARCASM_PATTERN,
    ROUND4_SEED,
    ROUND5_SEED,
    ROUND5_TEXT_MIN_LEN,
    ROUND5_TOTAL_TARGET,
    ROUND6_CHEER_PATTERN,
    ROUND6_COMMENT_HARD_FRAC,
    ROUND6_PROTEST_PATTERN,
    ROUND6_SAMPLE_SIZE,
    ROUND6_SEED,
    ROUND6_SLICE_FRAC,
)
from data_prep import clean_text


def length_bucket(text: str) -> str:
    n = len(text) if isinstance(text, str) else 0
    if n <= 20:
        return "short"
    if n <= 128:
        return "mid"
    return "long"


def prepare_pool(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out["request_content_text"].map(clean_text)
    out = out[out["text"].str.len() >= 2]
    out = out.sort_values(by=["content_hash", "aliyun_final"], na_position="last")
    out = out.drop_duplicates(subset=["content_hash"], keep="first")
    out["id"] = out["content_hash"].astype(str)
    out["platform"] = "weibo"
    out["source_table"] = out["source_table"].astype(str)
    keep = [
        "id", "text", "platform", "source_table", "content_hash",
        "aliyun_final", "positive_prob_aliyun", "negative_prob_aliyun",
    ]
    return out[keep].reset_index(drop=True)


def sample_spread_length(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if len(df) <= n:
        return df.sample(frac=1, random_state=int(rng.randint(0, 1_000_000_000))).reset_index(drop=True)
    work = df.copy()
    work["_bucket"] = work["text"].map(length_bucket)
    buckets = ["short", "mid", "long"]
    present = [b for b in buckets if (work["_bucket"] == b).any()]
    base, rem = divmod(n, len(present))
    parts = []
    unused = []
    for i, b in enumerate(present):
        want = base + (1 if i < rem else 0)
        g = work[work["_bucket"] == b]
        take_n = min(want, len(g))
        seed = int(rng.randint(0, 1_000_000_000))
        parts.append(g.sample(n=take_n, random_state=seed))
        rest = g.drop(parts[-1].index)
        if not rest.empty:
            unused.append(rest)
    still = n - sum(len(p) for p in parts)
    if still > 0 and unused:
        rest_df = pd.concat(unused)
        extra_n = min(still, len(rest_df))
        seed = int(rng.randint(0, 1_000_000_000))
        parts.append(rest_df.sample(n=extra_n, random_state=int(seed)))
    result = pd.concat(parts).drop(columns=["_bucket"])
    if len(result) > n:
        result = result.sample(n=n, random_state=int(rng.randint(0, 1_000_000_000)))
    return result.reset_index(drop=True)


def build_round1_sample(
    df: pd.DataFrame,
    sample_size: int = ROUND1_SAMPLE_SIZE,
    target_comments: int = ROUND1_TARGET_COMMENTS,
    target_neutrals: int = ROUND1_TARGET_NEUTRALS,
    seed: int = ROUND1_SEED,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    work = df.copy()
    neg = work[work["aliyun_final"] == 3]
    selected_ids = set(neg["id"])
    n_neg_c = int((neg["source_table"] == COMMENT_SOURCE).sum())
    n_neg_p = int((neg["source_table"] == POST_SOURCE).sum())
    comment_need = max(0, target_comments - n_neg_c)
    post_need = max(0, sample_size - target_comments - n_neg_p)

    remain = work[~work["id"].isin(selected_ids)]
    remain_c = remain[remain["source_table"] == COMMENT_SOURCE]
    remain_p = remain[remain["source_table"] == POST_SOURCE]

    neu_c = remain_c[remain_c["aliyun_final"] == 2]
    neu_p = remain_p[remain_p["aliyun_final"] == 2]
    pos_c = remain_c[remain_c["aliyun_final"] == 1]
    pos_p = remain_p[remain_p["aliyun_final"] == 1]

    n_neu_c = min(len(neu_c), comment_need, target_neutrals)
    take_neu_c = sample_spread_length(neu_c, n_neu_c, rng)
    take_pos_c = sample_spread_length(pos_c, comment_need - len(take_neu_c), rng)

    n_neu_p = min(len(neu_p), post_need, target_neutrals - len(take_neu_c))
    take_neu_p = sample_spread_length(neu_p, n_neu_p, rng)
    take_pos_p = sample_spread_length(pos_p, post_need - len(take_neu_p), rng)

    result = pd.concat(
        [neg, take_neu_c, take_pos_c, take_neu_p, take_pos_p], ignore_index=True
    )
    if len(result) < sample_size:
        leftover = work[~work["id"].isin(set(result["id"]))]
        extra = sample_spread_length(leftover, sample_size - len(result), rng)
        result = pd.concat([result, extra], ignore_index=True)
    result = result.drop_duplicates(subset=["id"]).reset_index(drop=True)
    if len(result) > sample_size:
        neg_rows = result[result["aliyun_final"] == 3]
        non_neg = result[result["aliyun_final"] != 3]
        if len(neg_rows) >= sample_size:
            result = neg_rows.reset_index(drop=True)
        else:
            slots = sample_size - len(neg_rows)
            if len(non_neg) > slots:
                non_neg = non_neg.sample(n=slots, random_state=seed)
            result = pd.concat([neg_rows, non_neg], ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    return result


def is_hard_example(conf, pred, p_neg, conf_th=None, p_neg_th=None) -> bool:
    """低置信，或判成中性但负面概率不低（阴阳/难例）。"""
    conf_th = LOW_CONFIDENCE_THRESHOLD if conf_th is None else conf_th
    p_neg_th = ROUND2_P_NEG_HARD if p_neg_th is None else p_neg_th
    if float(conf) < conf_th:
        return True
    return pred == "中性" and float(p_neg) >= p_neg_th


def _rng_take(df: pd.DataFrame, n: int, rng: np.random.RandomState) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    seed = int(rng.randint(0, 1_000_000_000))
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def build_round2_sample(
    df: pd.DataFrame,
    sample_size: int = ROUND2_SAMPLE_SIZE,
    seed: int = ROUND2_SEED,
    max_post_frac: float = ROUND2_MAX_POST_FRAC,
    hard_frac: float = ROUND2_HARD_FRAC,
    post_source: str = ROUND2_POST_SOURCE,
) -> pd.DataFrame:
    """主动学习抽样：总量固定，官方博文有上限，优先难例。"""
    rng = np.random.RandomState(seed)
    work = df.drop_duplicates(subset=["id"]).copy()
    hard_mask = [
        is_hard_example(r["student_conf"], r["student_pred"], r["p_neg"])
        for _, r in work.iterrows()
    ]
    work = work.assign(_hard=hard_mask)
    posts = work[work["source_table"] == post_source]
    comments = work[work["source_table"] != post_source]
    n_post_max = min(int(sample_size * max_post_frac), len(posts), sample_size)
    n_hard = min(int(round(sample_size * hard_frac)), sample_size)

    hard_c = comments[comments["_hard"]]
    hard_p = posts[posts["_hard"]]
    easy_c = comments[~comments["_hard"]]
    easy_p = posts[~posts["_hard"]]

    take_hc = _rng_take(hard_c, min(n_hard, len(hard_c)), rng)
    n_hard_left = n_hard - len(take_hc)
    take_hp = _rng_take(hard_p, min(n_hard_left, n_post_max, len(hard_p)), rng)

    selected = pd.concat([take_hc, take_hp], ignore_index=True)
    posts_used = len(take_hp)
    still = sample_size - len(selected)

    take_ec = _rng_take(easy_c, min(still, len(easy_c)), rng)
    selected = pd.concat([selected, take_ec], ignore_index=True)
    still = sample_size - len(selected)
    post_room = n_post_max - posts_used
    take_ep = _rng_take(easy_p, min(still, post_room, len(easy_p)), rng)
    selected = pd.concat([selected, take_ep], ignore_index=True)

    if len(selected) < sample_size:
        leftover = work[~work["id"].isin(set(selected["id"]))]
        leftover = leftover[leftover["source_table"] != post_source]
        extra = _rng_take(leftover, sample_size - len(selected), rng)
        selected = pd.concat([selected, extra], ignore_index=True)

    selected = selected.drop_duplicates(subset=["id"])
    if "_hard" in selected.columns:
        selected = selected.drop(columns=["_hard"])
    if len(selected) > sample_size:
        selected = selected.sample(n=sample_size, random_state=int(rng.randint(0, 1_000_000_000)))
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)


def is_low_conf(conf) -> bool:
    """round3 难例判定：仅低置信。
    round2 的 is_hard_example 还含「判中性但 p_neg 高」的单向捞负面条件，
    是采出负面 44.7%（金标 21%）的直接原因之一，round3 弃用。"""
    return float(conf) < LOW_CONFIDENCE_THRESHOLD


def build_round3_sample(
    df: pd.DataFrame,
    sample_size: int = ROUND3_SAMPLE_SIZE,
    seed: int = ROUND3_SEED,
    slice_fracs: dict | None = None,
    hard_frac: float = ROUND3_HARD_FRAC,
) -> pd.DataFrame:
    """round3 抽样：(platform, content_type) 分层随机为主 + 低置信难例为辅。

    round2 的 70% 难例配额导致标签先验漂移（负面 44.7%），round3 难例降为
    hard_frac，其余从各片池随机抽取，保证训练分布贴近真实分布。
    配额必须按 (platform, content_type) 而非仅 platform：学生模型只看文本，
    片间口径相反（博文应援=中性 vs B站/评论直球=正面），缺片的口径会被其他
    片带偏（v5 试标博文仅 8 条 -> 博文仪式应援被评论口径泄漏）。
    """
    if slice_fracs is None:
        slice_fracs = ROUND3_SLICE_FRAC
    rng = np.random.RandomState(seed)
    work = df.drop_duplicates(subset=["id"]).copy()
    work["_hard"] = work["student_conf"].map(is_low_conf)

    parts = []
    for (plat, ctype), frac in slice_fracs.items():
        want = int(round(sample_size * frac))
        sub = work[(work["platform"] == plat) & (work["content_type"] == ctype)]
        hard, easy = sub[sub["_hard"]], sub[~sub["_hard"]]
        take_hard = _rng_take(hard, min(int(round(want * hard_frac)), len(hard)), rng)
        take_easy = _rng_take(easy, min(want - len(take_hard), len(easy)), rng)
        still = want - len(take_hard) - len(take_easy)
        if still > 0:  # easy 池不足时用该平台剩余 hard 补
            rest_hard = hard[~hard["id"].isin(set(take_hard["id"]))]
            take_hard = pd.concat([take_hard, _rng_take(rest_hard, still, rng)])
        parts.append(take_hard)
        parts.append(take_easy)

    selected = pd.concat(parts, ignore_index=True)
    if len(selected) < sample_size:  # 平台池整体不足时从剩余全池补
        leftover = work[~work["id"].isin(set(selected["id"]))]
        selected = pd.concat([selected, _rng_take(leftover, sample_size - len(selected), rng)],
                             ignore_index=True)
    selected = selected.drop_duplicates(subset=["id"])
    if "_hard" in selected.columns:
        selected = selected.drop(columns=["_hard"])
    if len(selected) > sample_size:
        selected = selected.sample(n=sample_size, random_state=int(rng.randint(0, 1_000_000_000)))
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)


def downsample_by_label(
    df: pd.DataFrame,
    neg_max_frac: float = ROUND3_NEG_MAX_FRAC,
    label_col: str = "final_label",
    seed: int = 42,
) -> pd.DataFrame:
    """teacher 打标后的分布硬约束：负面占比超过 neg_max_frac 时随机降采样到该值。
    v2->v3 的实证：同批数据仅把负面降到 25.5% 就 +2pt 金标 acc。"""
    n = len(df)
    n_neg = int((df[label_col] == "负面").sum())
    if n_neg / n <= neg_max_frac:
        return df
    keep_n = int(round(n * neg_max_frac))
    neg_rows = df[df[label_col] == "负面"]
    keep_neg = neg_rows.sample(n=keep_n, random_state=seed)
    out = pd.concat([df[df[label_col] != "负面"], keep_neg], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


BLIND_SOURCE_TABLE = "weibo_sentiment_doubleblind_2000"
BILI_1600_SOURCE_TABLE = "bilibili_comments_1600"


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_round4_multigame_sample(
    blind_df: pd.DataFrame,
    exclude_texts: set,
    seed: int = ROUND4_SEED,
) -> pd.DataFrame:
    """round4 多游戏片：双盲2000 无人工标注行 -> 标准候选行（weibo 博文/评论）。

    金标 401 条出自该文件的人工标注列，绝不能进训练：只取标注为空的行；
    另按文本排除与既有训练/评估重合的行（实测与 v6_train 重合 428 条）。
    这批数据与金标博文同源同分布（异环/鸣潮等多游戏），用于修复 v6 的域偏移。
    blind_df 需含列：seq(序号)、text(文本)、type(类型: 博文/评论)、human_label(人工标注)。
    """
    work = blind_df.copy()
    work = work[work["human_label"].isna()]
    work["text"] = work["text"].astype(str).str.strip()
    keep = (work["text"].str.len() >= 2) & (~work["text"].isin(exclude_texts))
    work = work[keep]
    out = pd.DataFrame({
        "id": "blind2000_" + work["seq"].astype(str),
        "text": work["text"],
        "platform": "weibo",
        "content_type": work["type"].astype(str),
        "source_table": BLIND_SOURCE_TABLE,
    })
    out["content_hash"] = out["text"].map(_md5)
    out = out.drop_duplicates(subset=["content_hash"])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_round4_bili_hard_sample(
    agent_df: pd.DataFrame,
    score_map: dict,
    exclude_ids: set,
    n_target: int = ROUND4_BILI_HARD_TARGET,
    cheer_pattern: str = ROUND4_CHEER_PATTERN,
    sarcasm_pattern: str = ROUND4_SARCASM_PATTERN,
    seed: int = ROUND4_SEED,
) -> pd.DataFrame:
    """round4 B站定向难例：优先挖 v5 抽检确认的两类弱点样本，再按 agent-v5 分歧补足。

    分层优先级（先到先得，逐层去重）：
    1. 应援关键词 + v5 判非正面（直球应援判中性，抽检确认的正面召回弱点）
    2. agent 正向 & v5 中性（同上，换一种挖法）
    3. 黑话反讽关键词（捞批捞/定制/蛆类，抽检确认判成反方向的负面弱点）
    4. agent 负向 & v5 正面/中性（agent 负面口径偏激进，只作候选，由 teacher 重判）
    5. 随机补齐到 n_target
    agent_df 需含列：id、text(评论文本)、agent_label(agent标注)；score_map: id -> {student_pred,...}。
    """
    rng = np.random.RandomState(seed)
    work = agent_df.copy()
    work["id"] = work["id"].astype(str)
    work = work[~work["id"].isin(exclude_ids)]
    work["text"] = work["text"].astype(str).str.strip()
    work = work[(work["text"].str.len() >= 2) & (~work["text"].duplicated(keep="first"))]
    work["v5_pred"] = work["id"].map(lambda i: (score_map.get(i) or {}).get("student_pred"))

    cheer = work["text"].str.contains(cheer_pattern, regex=True)
    sarc = work["text"].str.contains(sarcasm_pattern, regex=True)
    tiers = [
        work[cheer & (work["v5_pred"] != "正面")],
        work[(work["agent_label"] == "正向") & (work["v5_pred"] == "中性")],
        work[sarc],
        work[(work["agent_label"] == "负向") & (work["v5_pred"].isin(["正面", "中性"]))],
    ]

    chosen_ids: set = set()
    picked = []
    for tier in tiers:
        tier = tier[~tier["id"].isin(chosen_ids)]
        take = _rng_take(tier, n_target - len(chosen_ids), rng)
        picked.append(take)
        chosen_ids |= set(take["id"])
        if len(chosen_ids) >= n_target:
            break
    non_empty = [p for p in picked if not p.empty]
    selected = pd.concat(non_empty, ignore_index=True) if non_empty else work.iloc[0:0]
    if len(selected) < n_target:  # 随机补齐
        rest = work[~work["id"].isin(chosen_ids)]
        selected = pd.concat([selected, _rng_take(rest, n_target - len(selected), rng)], ignore_index=True)

    out = pd.DataFrame({
        "id": selected["id"],
        "text": selected["text"],
        "platform": "bilibili",
        "content_type": "评论",
        "source_table": BILI_1600_SOURCE_TABLE,
    })
    out["content_hash"] = out["text"].map(_md5)
    return out.reset_index(drop=True)


def _proportional_alloc(total: int, counts: dict) -> dict:
    """最大余数法把 total 按各组规模比例分配（结果不超过各组自身，也不超过 total）。"""
    s = sum(counts.values())
    if s <= 0 or total <= 0:
        return {k: 0 for k in counts}
    raw = {k: total * v / s for k, v in counts.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    rem = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if rem <= 0:
            break
        if alloc[k] < counts[k]:
            alloc[k] += 1
            rem -= 1
    return {k: min(v, counts[k]) for k, v in alloc.items()}


def build_round5_pool(
    frames: dict,
    exclude_texts: set,
    min_len: int = ROUND5_TEXT_MIN_LEN,
) -> pd.DataFrame:
    """round5 示例品牌候选池：4 源归一化行合并 -> 长度/截断/排除集过滤 -> 跨源文本去重。

    frames: {源名: DataFrame}，每个 DataFrame 需含列：
    id / text / platform / content_type / source_table / machine_sentiment，
    可选 truncated（'1' 表示平台截断，剔除）。相关性过滤由调用方归一化时完成。
    """
    parts = []
    for name, df in frames.items():
        work = df.copy()
        work["_source"] = name
        work["text"] = work["text"].fillna("").astype(str).str.strip()
        if "truncated" in work.columns:
            work = work[work["truncated"].fillna("0").astype(str) != "1"]
        work = work[work["text"].str.len() >= min_len]
        work = work[~work["text"].isin(exclude_texts)]
        parts.append(work)
    pool = pd.concat(parts, ignore_index=True)
    pool["content_hash"] = pool["text"].map(_md5)
    pool = pool.drop_duplicates(subset=["content_hash"], keep="first")
    return pool.reset_index(drop=True)


def build_round5_sample(
    pool: pd.DataFrame,
    total_target: int = ROUND5_TOTAL_TARGET,
    seed: int = ROUND5_SEED,
) -> pd.DataFrame:
    """round5 采样：稀缺片（_source != xhs_comments）全取，xhs评论按机器先验分层补足余量。

    新域试点采用代表性采样而非难例挖掘：跨域机器置信不可靠，先验对齐交给 teacher
    重标 + downsample_by_label（round2/round3 实证路径）。
    """
    rng = np.random.RandomState(seed)
    rest = pool[pool["_source"] != "xhs_comments"]
    xhs = pool[pool["_source"] == "xhs_comments"]
    need = max(0, total_target - len(rest))
    parts = [rest]
    if len(xhs) and need:
        counts = xhs["machine_sentiment"].fillna("未知").value_counts().to_dict()
        alloc = _proportional_alloc(min(need, len(xhs)), counts)
        for lab, n in alloc.items():
            stratum = xhs[xhs["machine_sentiment"].fillna("未知") == lab]
            parts.append(_rng_take(stratum, n, rng))
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def _round6_hard_mask(df: pd.DataFrame, cheer_pattern: str) -> pd.Series:
    cheer = df["text"].str.contains(cheer_pattern, regex=True, na=False)
    if "student_pred" not in df.columns or "student_conf" not in df.columns:
        return cheer
    low_pos = (df["student_pred"] == "正面") & (pd.to_numeric(df["student_conf"], errors="coerce") < 0.75)
    return cheer | low_pos.fillna(False)


def _take_priority_then_rest(df: pd.DataFrame, n: int, priority: pd.Series, rng: np.random.RandomState) -> pd.DataFrame:
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    pri = df[priority.reindex(df.index).fillna(False)]
    rest = df[~priority.reindex(df.index).fillna(False)]
    take_pri = _rng_take(pri, min(n, len(pri)), rng)
    still = n - len(take_pri)
    take_rest = _rng_take(rest, still, rng)
    parts = [p for p in (take_pri, take_rest) if not p.empty]
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def build_round6_sample(
    pool: pd.DataFrame,
    n_total: int = ROUND6_SAMPLE_SIZE,
    slice_fracs: dict | None = None,
    comment_hard_frac: float = ROUND6_COMMENT_HARD_FRAC,
    cheer_pattern: str = ROUND6_CHEER_PATTERN,
    protest_pattern: str = ROUND6_PROTEST_PATTERN,
    exclude_texts: set | None = None,
    seed: int = ROUND6_SEED,
    min_len: int = 2,
) -> pd.DataFrame:
    """round6：三分片配额 + 微博评论应援/低置信正面难例 + B站抗议优先。

    pool 需含 id/text/platform/content_type/source_table，微博评论难例还要
    student_pred/student_conf。exclude_texts 按全文精确匹配剔除（金标/已训练）。
    """
    if slice_fracs is None:
        slice_fracs = ROUND6_SLICE_FRAC
    rng = np.random.RandomState(seed)
    work = pool.copy()
    work["text"] = work["text"].fillna("").astype(str).str.strip()
    work = work[work["text"].str.len() >= min_len]
    if exclude_texts:
        work = work[~work["text"].isin(exclude_texts)]
    work["content_hash"] = work["text"].map(_md5)
    work = work.drop_duplicates(subset=["content_hash"], keep="first").reset_index(drop=True)

    fake_counts = {k: max(1, int(round(v * 1000))) for k, v in slice_fracs.items()}
    alloc = _proportional_alloc(n_total, fake_counts)

    picked_ids: set = set()
    parts = []
    for (plat, ctype), n_want in alloc.items():
        sl = work[(work["platform"] == plat) & (work["content_type"] == ctype)]
        sl = sl[~sl["id"].isin(picked_ids)]
        if plat == "weibo" and ctype == "评论":
            n_hard = int(round(n_want * comment_hard_frac))
            hard_mask = _round6_hard_mask(sl, cheer_pattern)
            hard_take = _rng_take(sl[hard_mask], min(n_hard, int(hard_mask.sum())), rng)
            rest_need = n_want - len(hard_take)
            rest_take = _rng_take(sl[~sl["id"].isin(set(hard_take["id"]))], rest_need, rng)
            non_empty = [p for p in (hard_take, rest_take) if not p.empty]
            take = pd.concat(non_empty, ignore_index=True) if non_empty else sl.iloc[0:0].copy()
        elif plat == "bilibili" and ctype == "评论":
            protest = sl["text"].str.contains(protest_pattern, regex=True, na=False)
            take = _take_priority_then_rest(sl, n_want, protest, rng)
        else:
            take = _rng_take(sl, n_want, rng)
        if not take.empty:
            parts.append(take)
            picked_ids |= set(take["id"])

    selected = pd.concat(parts, ignore_index=True) if parts else work.iloc[0:0]
    still = n_total - len(selected)
    if still > 0:
        rest = work[~work["id"].isin(picked_ids)]
        selected = pd.concat([selected, _rng_take(rest, still, rng)], ignore_index=True)
    if len(selected) > n_total:
        selected = selected.sample(n=n_total, random_state=seed)
    keep = ["id", "text", "platform", "content_type", "source_table", "content_hash"]
    extra = [c for c in ("student_pred", "student_conf", "domain", "ctx") if c in selected.columns]
    out = selected[keep + extra].copy()
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
