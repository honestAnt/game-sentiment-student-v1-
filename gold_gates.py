# -*- coding: utf-8 -*-
"""晋级闸门判定的纯函数（无 torch）：分片切分、指标、三片闸门比对。

背景：v9 系列消融（v9…v9o）每轮都靠临时脚本算 gold_eval_v6 三片，口径散落在一次性
代码里且已经踩过一次 API 漂移（model_input_text 签名改过，临时脚本白跑一轮）。
这里把「分片口径 + 闸门比对」收成唯一真源，evaluate_gold_v6_gates.py 只管跑模型。

闸门定义（docs/v9训练方案-gold_eval_v6-2026-09-15.md 第 2 节）：
gold_v6 三片（微博评论 / 微博博文 / B站评论）在同一工作点下均不低于现役基线。
"""

from __future__ import annotations

from dataclasses import dataclass

from platform_input import derive_content_type

LABELS = ["负面", "中性", "正面"]
RECALL_KEYS = {"负面": "neg_rec", "中性": "neu_rec", "正面": "pos_rec"}
PRECISION_KEYS = {"负面": "neg_prec", "中性": "neu_prec", "正面": "pos_prec"}

# 三片闸门的 slice key，与 gold_eval_v6_*_baseline.json 的 slices_ex 同名
GATE_SLICES = ("weibo/评论", "weibo/博文", "bilibili/评论")
SLICE_CN = {"weibo/评论": "评论", "weibo/博文": "博文", "bilibili/评论": "B站"}


def slice_key(record: dict) -> str:
    """分片键 = platform/content_type，content_type 缺失时按 source_table 推导。"""
    return f"{record.get('platform')}/{derive_content_type(record)}"


def _accuracy(y: list[str], p: list[str]) -> float:
    if not y:
        return 0.0
    return sum(1 for yi, pi in zip(y, p) if yi == pi) / len(y)


def _recall(y: list[str], p: list[str], label: str) -> float | None:
    """该类别的召回；金标里没有这个类别时返回 None（不记 0，避免拉低 macro 解读）。"""
    support = [i for i, yi in enumerate(y) if yi == label]
    if not support:
        return None
    return sum(1 for i in support if p[i] == label) / len(support)


def _precision(y: list[str], p: list[str], label: str) -> float | None:
    """该类别的精确率；一条都没判成该类时返回 None。

    负面精确率是富集集的关键指标：只看召回会鼓励模型乱判负面，
    challenge 集里特意混了「带负面词但实为玩梗」的样本就是为了量它。
    """
    predicted = [i for i, pi in enumerate(p) if pi == label]
    if not predicted:
        return None
    return sum(1 for i in predicted if y[i] == label) / len(predicted)


def _macro_f1(y: list[str], p: list[str]) -> float:
    """三类 macro F1；无支撑且无预测的类别按 0 计入，与 sklearn zero_division=0 一致。"""
    f1s = []
    for label in LABELS:
        tp = sum(1 for yi, pi in zip(y, p) if yi == label and pi == label)
        fp = sum(1 for yi, pi in zip(y, p) if yi != label and pi == label)
        fn = sum(1 for yi, pi in zip(y, p) if yi == label and pi != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(f1s) / len(f1s)


def metrics(y: list[str], p: list[str]) -> dict:
    """单片/整体指标，字段与既有 baseline json 对齐。"""
    if len(y) != len(p):
        raise ValueError(f"gold/pred 长度不一致: {len(y)} != {len(p)}")
    out = {"n": len(y), "acc": _accuracy(y, p), "macro_f1": _macro_f1(y, p)}
    for label, key in RECALL_KEYS.items():
        out[key] = _recall(y, p, label)
    for label, key in PRECISION_KEYS.items():
        out[key] = _precision(y, p, label)
    out["n_neg_gold"] = sum(1 for yi in y if yi == "负面")
    return out


def slice_metrics(records: list[dict], golds: list[str], preds: list[str]) -> dict:
    """按 platform/content_type 切片算指标。"""
    if not (len(records) == len(golds) == len(preds)):
        raise ValueError(
            f"长度不一致: records {len(records)} / golds {len(golds)} / preds {len(preds)}"
        )
    buckets: dict[str, tuple[list[str], list[str]]] = {}
    for record, gold, pred in zip(records, golds, preds):
        y, p = buckets.setdefault(slice_key(record), ([], []))
        y.append(gold)
        p.append(pred)
    return {key: metrics(y, p) for key, (y, p) in buckets.items()}


@dataclass(frozen=True)
class Gate:
    """单条闸门。

    slice/metric  量哪一片的哪个指标
    source        指标取自哪个集合（"gold"=分布真实集；"challenge"=负面富集集）
    tolerance     允许低于基线的幅度；0=不得低于，0.01=护栏式「不低于基线 1pt」
    floor         绝对下限；设了就比绝对值而不比基线
    role          仅影响报告可读性（主判据 / 护栏）
    """
    slice: str
    metric: str = "acc"
    source: str = "gold"
    tolerance: float = 0.0
    floor: float | None = None
    role: str = "主判据"


# v6 口径（历史记录，勿改）：三片 acc 均不得低于基线。
# 已证伪——B站片 91.5% 是中性，「全判中性」即 0.9154，v8 只比它多对 1 条，
# 该片 acc 闸门分辨力约等于 0 且与「抓负面」反向，见 docs/v10训练方案-先修尺子-2026-09-17.md。
GATES_V6 = tuple(Gate(slice=s) for s in GATE_SLICES)

# v7 口径（2026-09-17 拍板）：
# 微博两片平庸基线只有 0.55/0.67，acc 有 15-20pt 有效信号，闸门照旧；
# B站片 acc 降级为护栏（不低于基线 1pt），主判据换成富集集上的负召回。
GATES_V7 = (
    Gate(slice="weibo/评论", metric="acc"),
    Gate(slice="weibo/博文", metric="acc"),
    Gate(slice="weibo/博文", metric="neg_rec", floor=0.88),
    Gate(slice="bilibili/评论", metric="acc", tolerance=0.01, role="护栏"),
    Gate(slice="bilibili/评论", metric="neg_rec", source="challenge"),
)


def _as_sourced(metrics: dict) -> dict:
    """允许传扁平的 {slice: metrics}，等价于只有 gold 一个来源。"""
    if metrics and all(isinstance(v, dict) and "acc" in v for v in metrics.values()):
        return {"gold": metrics}
    return metrics


def check_gates(candidate, baseline, gates=GATES_V6) -> dict:
    """逐条闸门比对。

    candidate/baseline 可以是扁平的 {slice: metrics}，
    也可以是多来源的 {source: {slice: metrics}}（v7 需要 challenge 集）。

    缺片/缺指标一律报错而不是当作通过——v9 系列里最容易出的错就是 slice key 拼错后
    「闸门 PASS」实际少算了一片。
    """
    cand, base = _as_sourced(candidate), _as_sourced(baseline)
    rows = []
    for gate in gates:
        for tag, data in (("候选", cand), ("基线", base)):
            if gate.source not in data:
                raise KeyError(f"{tag}缺来源 {gate.source}，实际有 {sorted(data)}")
            if gate.slice not in data[gate.source]:
                raise KeyError(f"{tag}[{gate.source}] 缺分片 {gate.slice}，"
                               f"实际有 {sorted(data[gate.source])}")
        c = cand[gate.source][gate.slice]
        b = base[gate.source][gate.slice]
        if gate.metric not in c:
            raise KeyError(f"候选[{gate.source}][{gate.slice}] 缺指标 {gate.metric}")
        value = c[gate.metric]
        if value is None:
            raise ValueError(f"候选[{gate.source}][{gate.slice}] 的 {gate.metric} 为 None"
                             f"（该类别在集合里无支撑，闸门无法判定）")
        if gate.floor is not None:
            threshold, ref = gate.floor, None
        else:
            ref = b[gate.metric]
            if ref is None:
                raise ValueError(f"基线[{gate.source}][{gate.slice}] 的 {gate.metric} 为 None")
            threshold = ref - gate.tolerance
        rows.append({
            "slice": gate.slice,
            "name": SLICE_CN.get(gate.slice, gate.slice),
            "source": gate.source,
            "metric": gate.metric,
            "role": gate.role,
            "n": c["n"],
            "value": value,
            "baseline": ref,
            "threshold": threshold,
            "tolerance": gate.tolerance,
            "delta": None if ref is None else value - ref,
            "passed": value >= threshold,
            # 兼容既有产物字段（gold_eval_v6_*_baseline.json 的 gates.slices[].acc/neg_rec）
            "acc": c["acc"],
            "baseline_acc": b["acc"],
            "neg_rec": c.get("neg_rec"),
        })
    return {"slices": rows, "passed": all(r["passed"] for r in rows)}


def format_gate_report(gate_result: dict, overall: dict | None = None,
                       baseline_overall_acc: float | None = None) -> str:
    """闸门结果的终端表格，供 CLI 与日志共用。"""
    lines = []
    if overall is not None:
        line = f"  overall: acc={overall['acc']:.4f}"
        if baseline_overall_acc is not None:
            line += f" Δ基线={overall['acc'] - baseline_overall_acc:+.4f}"
        neg_rec = overall.get("neg_rec")
        if neg_rec is not None:
            line += f" neg_rec={neg_rec:.4f}"
        lines.append(line)
    for row in gate_result["slices"]:
        src = "" if row["source"] == "gold" else f"@{row['source']}"
        role = "" if row["role"] == "主判据" else f"[{row['role']}]"
        if row["baseline"] is None:
            cmp_txt = f"下限={row['threshold']:.4f}"
        else:
            tol = f"−{row['tolerance']:.2f}" if row["tolerance"] else ""
            cmp_txt = f"Δ基线={row['delta']:+.4f}（阈值 基线{tol}={row['threshold']:.4f}）"
        lines.append(
            f"  {row['name']}{src} {row['metric']}{role}: {row['value']:.4f} "
            f"n={row['n']} {cmp_txt} {'PASS' if row['passed'] else 'FAIL'}"
        )
    n_gates = len(gate_result["slices"])
    lines.append(f"\n{n_gates} 条闸门: {'PASS' if gate_result['passed'] else 'FAIL'}")
    return "\n".join(lines)
