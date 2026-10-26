import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from share_student import (
    REQUIRED_CONFIG,
    build_model_card,
    collect_export_files,
    make_zip,
    write_bundle,
)


def _touch(path: Path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_model_dir(root: Path) -> Path:
    model = root / "student_v1"
    _touch(model / "config.json", '{"id2label": {"0": "负面", "1": "中性", "2": "正面"}}')
    _touch(model / "model.safetensors")
    _touch(model / "tokenizer.json")
    _touch(model / "tokenizer_config.json")
    _touch(model / "special_tokens_map.json")
    _touch(model / "vocab.txt")
    _touch(model / "training_args.bin")
    _touch(model / "checkpoint-168" / "model.safetensors")
    _touch(model / "checkpoint-168" / "trainer_state.json")
    return model


def test_collect_export_files_keeps_hf_weights_skips_trainer_and_checkpoints(tmp_path):
    model = _fake_model_dir(tmp_path)
    names = {p.name for p in collect_export_files(model)}
    assert "config.json" in names
    assert "model.safetensors" in names
    assert "vocab.txt" in names
    assert "tokenizer.json" in names
    assert "training_args.bin" not in names
    assert "trainer_state.json" not in names
    rels = {str(p.relative_to(model)) for p in collect_export_files(model)}
    assert not any(r.startswith("checkpoint-") for r in rels)


def test_collect_export_files_requires_config_and_weights(tmp_path):
    model = tmp_path / "empty"
    model.mkdir()
    _touch(model / "vocab.txt")
    with pytest.raises(SystemExit, match="缺少"):
        collect_export_files(model)


def test_write_bundle_and_zip_are_self_contained(tmp_path):
    model = _fake_model_dir(tmp_path)
    bundle = tmp_path / "bundle"
    write_bundle(model, bundle)
    assert (bundle / "config.json").is_file()
    assert (bundle / "model.safetensors").is_file()
    assert (bundle / "run_student.py").is_file()
    assert (bundle / "requirements.txt").is_file()
    readme = (bundle / "README.md").read_text(encoding="utf-8")
    assert "负面" in readme
    assert "local_files_only" in readme
    assert "from_pretrained" in readme
    assert not (bundle / "training_args.bin").exists()
    assert not (bundle / "checkpoint-168").exists()

    zpath = tmp_path / "student_v1.zip"
    make_zip(bundle, zpath)
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    assert any(n.endswith("run_student.py") for n in names)
    assert any(n.endswith("model.safetensors") for n in names)
    assert all("checkpoint-" not in n for n in names)


def test_model_card_mentions_offline_and_hub():
    card = build_model_card("student_v1")
    assert "huggingface" in card.lower() or "Hugging Face" in card
    assert "--local-files-only" in card
    assert REQUIRED_CONFIG in card or "config.json" in card
    assert "256" in card
