# -*- coding: utf-8 -*-
import argparse
import json

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from config import ALIYUN_SENTIMENT_TO_LABEL, LABEL2ID


def load_sidecar(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def load_labeled(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def compare_frames(side: pd.DataFrame, lab: pd.DataFrame) -> dict:
    m = side.merge(lab[["id", "final_label"]], on="id", how="inner")
    m["aliyun_label"] = m["aliyun_final"].map(ALIYUN_SENTIMENT_TO_LABEL)
    m = m[m["aliyun_label"].notna() & m["final_label"].notna()]
    labels = list(LABEL2ID.keys())
    agree = float((m["aliyun_label"] == m["final_label"]).mean()) if len(m) else 0.0
    print(classification_report(m["aliyun_label"], m["final_label"], labels=labels, zero_division=0, digits=3))
    print("混淆矩阵 行=阿里云 列=GLM")
    print(confusion_matrix(m["aliyun_label"], m["final_label"], labels=labels))
    print(f"一致率 {agree:.1%} n={len(m)}")
    return {"n": len(m), "agree": agree}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", default="data/candidates/round1/aliyun_strata_sidecar.jsonl")
    parser.add_argument("--labeled", default="data/labeled/round1_labeled.jsonl")
    args = parser.parse_args()
    compare_frames(load_sidecar(args.sidecar), load_labeled(args.labeled))


if __name__ == "__main__":
    main()
