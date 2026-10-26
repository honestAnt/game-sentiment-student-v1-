# -*- coding: utf-8 -*-
"""
teacher_labeling.py
用两个大模型（Qwen + DeepSeek）分别对候选样本做情感标注，
只保留两者判断一致的样本作为高置信度训练数据，不一致的样本转人工复核。

依赖：pip install openai pandas tqdm

用法：
    export DASHSCOPE_API_KEY=xxx
    export DEEPSEEK_API_KEY=xxx
    python teacher_labeling.py --input data/candidates/round1_candidates.jsonl \
        --output data/labeled/round1_labeled.jsonl \
        --review_output data/needs_review/round1_review.jsonl
"""

import argparse
import concurrent.futures
import json
import time

from openai import OpenAI
from tqdm import tqdm

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL_NAME,
    QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL_NAME,
    LLM_CONCURRENCY, LLM_MAX_RETRIES, LABEL2ID, PLATFORM_CONTEXT, EMOTION_TAGS,
)

qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

VALID_LABELS = list(LABEL2ID.keys())
VALID_EMOTIONS = EMOTION_TAGS

SYSTEM_PROMPT = """你是一个专业的游戏行业社交媒体舆情分析助手。你的任务是判断用户评论对游戏内容/游戏本身的情感倾向。

请严格按照以下 JSON 格式输出，不要输出任何其他文字、解释或 markdown 代码块标记：
{"sentiment": "正面/负面/中性之一", "emotion": "从给定情绪标签中选一个最贴切的", "reason": "一句话理由，不超过20字"}

情感判定规则：
- 正面：认可、喜爱、推荐、期待、满意
- 负面：批评、失望、愤怒、抱怨、不推荐
- 中性：单纯陈述事实、提问、与情感无关的攻略类内容、灌水评论（如"沙发""关注了"）

注意事项：
- 游戏圈的阴阳怪气、反讽表达要识别真实情感，不要被字面的褒义词欺骗（例如"这波运营真是绝了"在骂人的语境下是负面）
- 数值讨论、机制讨论如果不带明显情绪色彩，判定为中性
- 请勿被无关话题（如引流、广告）干扰，这类内容按中性处理

可选情绪标签（emotion 字段从中选择）：{emotion_options}
"""


def build_prompt(text: str, platform: str) -> list:
    platform_ctx = PLATFORM_CONTEXT.get(platform, "内容来自中文游戏相关社交媒体评论。")
    system = SYSTEM_PROMPT.format(emotion_options="、".join(VALID_EMOTIONS))
    user = f"平台背景：{platform_ctx}\n\n待判断文本：{text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_llm(client: OpenAI, model_name: str, text: str, platform: str) -> dict:
    """调用单个 LLM，带重试和 JSON 解析容错"""
    messages = build_prompt(text, platform)
    last_err = None
    for attempt in range(LLM_MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,        # 标注任务要稳定输出，不需要多样性
                max_tokens=150,
                response_format={"type": "json_object"},   # 强制 JSON 输出，减少解析失败
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)
            sentiment = parsed.get("sentiment", "").strip()
            emotion = parsed.get("emotion", "其他").strip()
            reason = parsed.get("reason", "")
            if sentiment not in VALID_LABELS:
                # 模型偶尔会输出同义表达，做一次简单归一化兜底
                sentiment = normalize_label(sentiment)
            if emotion not in VALID_EMOTIONS:
                emotion = "其他"
            return {"sentiment": sentiment, "emotion": emotion, "reason": reason}
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    print(f"[警告] 调用 {model_name} 失败，已重试{LLM_MAX_RETRIES}次: {last_err}")
    return {"sentiment": None, "emotion": None, "reason": f"ERROR: {last_err}"}


def normalize_label(raw: str) -> str:
    mapping = {
        "positive": "正面", "negative": "负面", "neutral": "中性",
        "积极": "正面", "消极": "负面",
    }
    return mapping.get(raw.lower() if isinstance(raw, str) else raw, None)


def label_one_sample(record: dict) -> dict:
    text, platform = record["text"], record["platform"]
    qwen_result = call_llm(qwen_client, QWEN_MODEL_NAME, text, platform)
    deepseek_result = call_llm(deepseek_client, DEEPSEEK_MODEL_NAME, text, platform)

    record["qwen_sentiment"] = qwen_result["sentiment"]
    record["qwen_emotion"] = qwen_result["emotion"]
    record["deepseek_sentiment"] = deepseek_result["sentiment"]
    record["deepseek_emotion"] = deepseek_result["emotion"]

    agree = (
        qwen_result["sentiment"] is not None
        and qwen_result["sentiment"] == deepseek_result["sentiment"]
    )
    record["agree"] = agree
    if agree:
        record["final_label"] = qwen_result["sentiment"]
        # 情绪标签取 Qwen 结果作为主展示字段（辅助分析用，不影响训练主任务）
        record["final_emotion"] = qwen_result["emotion"]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="双模型一致的高置信度标注结果")
    parser.add_argument("--review_output", required=True, help="双模型不一致的样本，需人工复核")
    args = parser.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as executor:
        futures = [executor.submit(label_one_sample, r) for r in records]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM 标注中"):
            results.append(future.result())

    agreed = [r for r in results if r.get("agree")]
    disagreed = [r for r in results if not r.get("agree")]

    with open(args.output, "w", encoding="utf-8") as f:
        for r in agreed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.review_output, "w", encoding="utf-8") as f:
        for r in disagreed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    print(f"\n[完成] 总样本 {total} 条")
    print(f"  一致（写入训练集）: {len(agreed)} 条 ({len(agreed)/total:.1%})")
    print(f"  不一致（转人工复核）: {len(disagreed)} 条 ({len(disagreed)/total:.1%})")


if __name__ == "__main__":
    main()
