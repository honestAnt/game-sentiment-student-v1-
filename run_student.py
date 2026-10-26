# -*- coding: utf-8 -*-
"""
run_student.py
学生模型推理入口。可对解压后的本地目录或 Hugging Face 仓库 ID 使用，
不依赖本仓库其他文件。

    python run_student.py --model . --text "这游戏太坑了" --local-files-only
    python run_student.py --model models/student_v1 --input comments.txt --output pred.jsonl
    python run_student.py --model 用户名/仓库名 --text "还行吧"
"""

import argparse
import json
import os
import sys

MAX_SEQ_LEN = 256


def pick_device(explicit: str) -> str:
    import torch

    if explicit and explicit != "auto":
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_batch_size(device: str, requested: int) -> int:
    if requested > 0:
        return requested
    return 16 if device == "mps" else 32


def load_model(model_id: str, local_files_only: bool, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    kwargs = {"local_files_only": local_files_only}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict(texts, model, tokenizer, device, batch_size: int):
    import torch

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels, confidences = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
        labels.extend(id2label[int(p)] for p in pred.cpu().tolist())
        confidences.extend(float(c) for c in conf.cpu().tolist())
    return labels, confidences


def read_texts(args) -> list:
    if args.text:
        return list(args.text)
    if args.input:
        path = args.input
        if path.endswith(".jsonl"):
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    rows.append(obj.get("text") or obj.get("评论文本") or "")
            return rows
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f if ln.strip()]
    if not sys.stdin.isatty():
        return [ln.rstrip("\n") for ln in sys.stdin if ln.strip()]
    raise SystemExit("请用 --text、--input 或管道提供文本")


def main():
    parser = argparse.ArgumentParser(description="学生情感模型推理（本地目录或 Hugging Face）")
    parser.add_argument("--model", required=True, help="本地目录或 Hugging Face 仓库 ID")
    parser.add_argument("--text", action="append", default=[], help="可重复传入多条")
    parser.add_argument("--input", default="", help="txt（一行一条）或 jsonl（text 字段）")
    parser.add_argument("--output", default="", help="jsonl；不填则打印到终端")
    parser.add_argument("--local-files-only", action="store_true", help="离线，禁止访问 Hugging Face")
    parser.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    parser.add_argument("--batch-size", type=int, default=0, help="0 表示按设备自动选")
    args = parser.parse_args()

    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    texts = read_texts(args)
    device = pick_device(args.device)
    batch_size = infer_batch_size(device, args.batch_size)
    print(f"[device] {device}  n={len(texts)}  batch={batch_size}", file=sys.stderr)

    model, tokenizer = load_model(args.model, args.local_files_only, device)
    labels, confidences = predict(texts, model, tokenizer, device, batch_size)

    records = [
        {"text": t, "label": lab, "confidence": round(conf, 6)}
        for t, lab, conf in zip(texts, labels, confidences)
    ]
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[wrote] {args.output}", file=sys.stderr)
    else:
        for rec in records:
            print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
