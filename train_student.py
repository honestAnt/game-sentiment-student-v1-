# -*- coding: utf-8 -*-
"""
train_student.py
用 teacher_labeling.py 产出的高置信度蒸馏数据，微调生产环境用的 BERT 分类模型。

依赖：pip install transformers datasets scikit-learn torch accelerate

用法：
    python train_student.py --train_file data/labeled/round1_labeled.jsonl \
        --output_dir models/student_v1 --val_ratio 0.1

    # 后续迭代（有旧模型基础上继续训练新增数据）：
    python train_student.py --train_file data/labeled/round2_labeled.jsonl \
        --output_dir models/student_v2 --init_from models/student_v1
"""

import argparse
import json
import os
from collections import Counter

from config import (
    ID2LABEL, LABEL2ID, MAX_SEQ_LEN, NUM_LABELS, STUDENT_BASE_MODEL,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    MPS_HIGH_WATERMARK_RATIO, mps_training_arg_overrides,
)
from platform_input import derive_content_type, model_input_text

# TOKENIZERS 必须在 import tokenizers/torch 之前关掉并行，避免 fork 警告。
# MPS 内存上限：只用 torch.mps.set_per_process_memory_fraction，不要再设
# PYTORCH_MPS_HIGH_WATERMARK_RATIO——torch 2.2.2 上 env 与 API 同时生效会报
#「invalid low watermark ratio 1.4」，训练直接起不来。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 若外部环境误留了该变量，清掉，避免与下方 API 冲突
os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from infer_batch import pick_device


def load_jsonl_as_dataset(path: str, platform_only: bool = False) -> Dataset:
    texts, labels, slice_keys = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            label = r.get("final_label")
            if label not in LABEL2ID:
                continue   # 跳过没有有效标签的脏数据
            texts.append(
                model_input_text(r["text"], r.get("platform"), derive_content_type(r),
                                 platform_only=platform_only)
            )
            labels.append(LABEL2ID[label])
            slice_keys.append(f"{r.get('platform')}|{derive_content_type(r) or '?'}")
    return Dataset.from_dict({"text": texts, "label": labels, "slice_key": slice_keys})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0
    )
    metrics = {"accuracy": acc, "macro_f1": macro_f1}
    for i, label_name in ID2LABEL.items():
        metrics[f"f1_{label_name}"] = f1_per_class[i]
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="teacher_labeling.py 产出的 jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--init_from", default=None,
                         help="从上一轮训练好的学生模型继续训练（增量迭代场景），不填则从 STUDENT_BASE_MODEL 开始")
    parser.add_argument("--resume_from", default=None, help="从本次 output_dir 里的 checkpoint 接着训")
    parser.add_argument("--prefix_granularity", choices=["platform_ct", "platform"], default="platform_ct",
                        help="平台前缀粒度：platform_ct=(platform,content_type) 组合前缀（v8），"
                             "platform=仅平台前缀（v8b 消融，验证 content_type 粒度是否引入噪声）")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="覆盖 config.LEARNING_RATE（续训消融用，默认不改）")
    parser.add_argument("--num_train_epochs", type=float, default=None,
                        help="覆盖 config.NUM_EPOCHS（续训消融用，默认不改）")
    args = parser.parse_args()

    device = pick_device()
    print(f"[训练设备] {device}")
    if device == "mps" and hasattr(torch.mps, "set_per_process_memory_fraction"):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        torch.mps.set_per_process_memory_fraction(float(MPS_HIGH_WATERMARK_RATIO))
        print(f"[MPS] per-process memory fraction={MPS_HIGH_WATERMARK_RATIO}")

    base_model = args.init_from or STUDENT_BASE_MODEL
    print(f"[加载底座模型] {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,   # 分类头类别数可能和原 checkpoint 不同（比如从二分类换成三分类）
        attn_implementation="eager",
    )

    full_dataset = load_jsonl_as_dataset(args.train_file, platform_only=args.prefix_granularity == "platform")
    print(f"[数据量] 共 {len(full_dataset)} 条有效标注样本")

    df = full_dataset.to_pandas()
    # 前缀模型按 (platform, content_type, label) 分层，保证早停验证信号覆盖每个切片；
    # 任一层样本数 < 2 时逐级回退：label 分层 -> 不分层，避免 split 报错
    slice_label = [f"{sk}|{ID2LABEL[l]}" for sk, l in zip(df["slice_key"], df["label"])]
    label_list = list(df["label"])
    n_slice, n_label = Counter(slice_label), Counter(label_list)
    if n_slice and min(n_slice.values()) >= 2:
        stratify = slice_label
    elif min(n_label.values()) >= 2:
        stratify = label_list
        print(f"[切分] 存在样本数<2 的切片层，回退 label 分层：{[k for k, v in n_slice.items() if v < 2]}")
    else:
        stratify = None
        print("[切分] 有单例标签，放弃分层（样本过少，仅供冒烟）")
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=42, stratify=stratify
    )
    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    lr = LEARNING_RATE if args.learning_rate is None else args.learning_rate
    epochs = NUM_EPOCHS if args.num_train_epochs is None else args.num_train_epochs
    print(f"[超参] lr={lr} epochs={epochs} (config默认 lr={LEARNING_RATE} epochs={NUM_EPOCHS})")
    training_kw = dict(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=5,
        save_total_limit=4,
        report_to="none",
        dataloader_pin_memory=False,
    )
    if device == "cpu":
        training_kw["use_cpu"] = True
    elif device == "mps":
        training_kw.update(mps_training_arg_overrides())
        print(
            f"[MPS] train_bs={training_kw['per_device_train_batch_size']} "
            f"accum={training_kw['gradient_accumulation_steps']} "
            f"eval_bs={training_kw['per_device_eval_batch_size']} "
            f"watermark={MPS_HIGH_WATERMARK_RATIO} "
            f"（MPS micro-batch + set_per_process_memory_fraction）"
        )
    training_args = TrainingArguments(**training_kw)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)

    final_metrics = trainer.evaluate()
    print("\n[最终验证集指标]")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[完成] 模型已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()
