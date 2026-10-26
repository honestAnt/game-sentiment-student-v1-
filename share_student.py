# -*- coding: utf-8 -*-
"""
share_student.py
把训练好的学生模型打成可发给别人的离线包，也可上传到 Hugging Face Hub。

离线包里是标准 Transformers 目录 + 独立推理脚本，对方不需要本仓库其余代码。

用法：
    # 打包 zip（默认写出 dist/student_v1/ 和 dist/student_v1.zip）
    python share_student.py --model_dir models/student_v1

    # 同时传到 Hugging Face（需 huggingface-cli login 或环境变量 HF_TOKEN）
    python share_student.py --model_dir models/student_v1 \\
        --hf-repo 你的用户名/game-sentiment-student-v1
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

REQUIRED_CONFIG = "config.json"

WEIGHT_FILES = {
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
}
CONFIG_FILES = {REQUIRED_CONFIG, "generation_config.json"}
SKIP_FILES = {
    "training_args.bin",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
}

INFER_REQUIREMENTS = """torch>=2.1.0
transformers>=4.40.0
"""


def _is_weight_name(name: str) -> bool:
    if name in WEIGHT_FILES:
        return True
    if name.startswith("model-") and name.endswith(".safetensors"):
        return True
    if name.startswith("pytorch_model-") and name.endswith(".bin"):
        return True
    return False


def _has_weight(names) -> bool:
    skip_index = {"model.safetensors.index.json", "pytorch_model.bin.index.json"}
    return any(_is_weight_name(n) and n not in skip_index for n in names)


def collect_export_files(model_dir) -> list:
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"缺少模型目录: {model_dir}")

    files = []
    for path in model_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if name.startswith(".") or name == ".DS_Store" or name in SKIP_FILES:
            continue
        if name in CONFIG_FILES or name in TOKENIZER_FILES or _is_weight_name(name):
            files.append(path)

    names = {p.name for p in files}
    missing = []
    if REQUIRED_CONFIG not in names:
        missing.append(REQUIRED_CONFIG)
    if not _has_weight(names):
        missing.append("model.safetensors 或 pytorch_model.bin")
    if missing:
        raise SystemExit("缺少" + "、".join(missing))
    return sorted(files, key=lambda p: p.name)


def build_model_card(model_name: str) -> str:
    return f"""# {model_name}

游戏相关中文评论 / 博文情感三分类（负面 / 中性 / 正面）。
底座：IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment，序列长度 256。

标签与 `config.json` 中 `id2label` 一致：0 负面、1 中性、2 正面。

## 离线使用

解压 zip 后，在该目录安装依赖并推理（`--local-files-only` 禁止访问网络）：

```bash
pip install -r requirements.txt
python run_student.py --model . --text "这游戏太坑了" --local-files-only
python run_student.py --model . --input comments.txt --output pred.jsonl --local-files-only
```

Python：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

path = "."  # 解压后的目录
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
model.eval()
```

## Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
tok = AutoTokenizer.from_pretrained("填写仓库ID")
model = AutoModelForSequenceClassification.from_pretrained("填写仓库ID")
```

或：

```bash
python run_student.py --model 填写仓库ID --text "这游戏太坑了"
```

已缓存到本地后可加 `--local-files-only`。
"""


def write_bundle(model_dir, bundle_dir) -> Path:
    model_dir = Path(model_dir)
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    for src in collect_export_files(model_dir):
        shutil.copy2(src, bundle_dir / src.name)

    (bundle_dir / "README.md").write_text(
        build_model_card(model_dir.name), encoding="utf-8"
    )
    (bundle_dir / "requirements.txt").write_text(INFER_REQUIREMENTS, encoding="utf-8")

    runner = Path(__file__).resolve().parent / "run_student.py"
    if not runner.is_file():
        raise SystemExit(f"缺少推理脚本: {runner}")
    shutil.copy2(runner, bundle_dir / "run_student.py")
    return bundle_dir


def make_zip(bundle_dir, zip_path) -> Path:
    bundle_dir = Path(bundle_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    root_name = bundle_dir.name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(bundle_dir.rglob("*")):
            if not path.is_file():
                continue
            arc = f"{root_name}/{path.relative_to(bundle_dir).as_posix()}"
            compress = (
                zipfile.ZIP_STORED
                if path.suffix in {".safetensors", ".bin"}
                else zipfile.ZIP_DEFLATED
            )
            zf.write(path, arcname=arc, compress_type=compress)
    return zip_path


def push_to_hub(bundle_dir, repo_id: str, token: str | None, private: bool) -> str:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise SystemExit("上传 Hugging Face 需要: pip install huggingface_hub") from exc

    create_repo(repo_id, exist_ok=True, private=private, token=token, repo_type="model")
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    return f"https://huggingface.co/{repo_id}"


def main():
    parser = argparse.ArgumentParser(description="打包学生模型：离线 zip 或上传 Hugging Face")
    parser.add_argument("--model_dir", default="models/student_v1")
    parser.add_argument("--output_dir", default="dist")
    parser.add_argument("--no-zip", action="store_true", help="只写目录，不打 zip")
    parser.add_argument("--hf-repo", default="", help="Hugging Face 仓库，如 username/game-sentiment-student-v1")
    parser.add_argument("--public", action="store_true", help="HF 仓库公开；默认私有")
    parser.add_argument("--hf-token", default="", help="不填则读环境变量 HF_TOKEN")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_root = Path(args.output_dir)
    bundle_dir = out_root / model_dir.name
    write_bundle(model_dir, bundle_dir)
    print(f"[离线目录] {bundle_dir.resolve()}")

    if not args.no_zip:
        zpath = out_root / f"{model_dir.name}.zip"
        make_zip(bundle_dir, zpath)
        size_mb = zpath.stat().st_size / (1024 * 1024)
        print(f"[离线 zip] {zpath.resolve()}  ({size_mb:.1f} MB)")
        print("发给别人：解压后 pip install -r requirements.txt，再 python run_student.py --model . --text '...' --local-files-only")

    if args.hf_repo:
        token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        url = push_to_hub(bundle_dir, args.hf_repo, token or None, private=not args.public)
        print(f"[Hugging Face] {url}")
        vis = "公开" if args.public else "私有"
        print(f"仓库为{vis}。对方: python run_student.py --model {args.hf_repo} --text '...'")


if __name__ == "__main__":
    main()
