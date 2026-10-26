# -*- coding: utf-8 -*-
"""B站金标回收（merge_gold_bili）的单测：标签归一化、存档回查、去重、覆盖率闸。"""

import json

import pandas as pd
import pytest

from merge_gold_bili import (
    append_to_gold, archive_path_for, labeled_rows, load_archive, report_issues,
    to_gold_record,
)


def _write_archive(tmp_path, recs, name="cand.jsonl"):
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(p)


def _write_xlsx(tmp_path, rows, name="cand.xlsx"):
    p = tmp_path / name
    pd.DataFrame(rows).to_excel(p, index=False)
    return str(p)


ARCHIVE = [
    {"id": "ge7c_101", "text": "策划给个说法", "platform": "bilibili",
     "content_type": "评论", "domain": "异环", "channel": "问责"},
    {"id": "ge7c_102", "text": "这优化真垃圾", "platform": "bilibili",
     "content_type": "评论", "domain": "原神", "channel": "显式批评"},
    {"id": "ge7c_103", "text": "低命龙王可以用垃圾队", "platform": "bilibili",
     "content_type": "评论", "domain": "原神", "channel": "显式批评"},
]


def _xlsx_rows(labels):
    return [
        {"序号": a["id"].split("_", 1)[1], "类型": "评论",
         "视频标题（评论的上下文）": "", "文本": a["text"],
         "召回通道": a["channel"], "人工标注-杠": lab}
        for a, lab in zip(ARCHIVE, labels)
    ]


class TestLoadArchive:
    def test_keys_on_raw_cid_not_prefixed_id(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        assert set(arc) == {"101", "102", "103"}
        assert arc["101"]["id"] == "ge7c_101"

    def test_archive_path_derived_from_xlsx(self):
        assert archive_path_for("a/b/c.xlsx") == "a/b/c.jsonl"


class TestLabeledRows:
    def test_normalizes_label_synonyms(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        path = _write_xlsx(tmp_path, _xlsx_rows(["负向", "负面", "中立"]))
        rows, issues = labeled_rows(path, arc)
        assert [r["gold_label"] for r in rows] == ["负面", "负面", "中性"]
        assert issues["bad_label"] == []

    def test_blank_labels_skipped_not_failed(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        path = _write_xlsx(tmp_path, _xlsx_rows(["负面", None, None]))
        rows, issues = labeled_rows(path, arc)
        assert len(rows) == 1 and issues["n_unlabeled"] == 2

    def test_unmappable_label_reported_not_silently_dropped(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        path = _write_xlsx(tmp_path, _xlsx_rows(["负面", "说不准", "中性"]))
        rows, issues = labeled_rows(path, arc)
        assert len(rows) == 2
        assert issues["bad_label"] == [("102", "说不准")]

    def test_structure_fields_come_from_archive_not_xlsx(self, tmp_path):
        """xlsx 经人手编辑，不能当结构真源：platform/channel 一律回查存档。"""
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        rows_in = _xlsx_rows(["负面", "负面", "中性"])
        rows_in[0]["召回通道"] = "被手改乱的值"
        rows_in[0]["类型"] = "博文"
        path = _write_xlsx(tmp_path, rows_in)
        rows, _ = labeled_rows(path, arc)
        assert rows[0]["channel"] == "问责"
        assert rows[0]["content_type"] == "评论"
        assert rows[0]["platform"] == "bilibili"

    def test_text_drift_reported_and_archive_wins(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        rows_in = _xlsx_rows(["负面", "负面", "中性"])
        rows_in[1]["文本"] = "被 Excel 改过的文本"
        path = _write_xlsx(tmp_path, rows_in)
        rows, issues = labeled_rows(path, arc)
        assert issues["text_drift"] == ["102"]
        assert rows[1]["text"] == "这优化真垃圾"

    def test_unknown_id_reported(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE[:2]))
        path = _write_xlsx(tmp_path, _xlsx_rows(["负面", "负面", "中性"]))
        rows, issues = labeled_rows(path, arc)
        assert len(rows) == 2 and issues["missing_in_archive"] == ["103"]

    def test_missing_column_raises_with_actual_columns(self, tmp_path):
        arc = load_archive(_write_archive(tmp_path, ARCHIVE))
        path = _write_xlsx(tmp_path, [{"序号": "101", "文本": "x"}])
        with pytest.raises(ValueError, match="人工标注-杠"):
            labeled_rows(path, arc)


class TestReportIssues:
    def _issues(self, n_total, n_unlabeled):
        return {"n_total": n_total, "n_unlabeled": n_unlabeled, "bad_label": [],
                "missing_in_archive": [], "text_drift": []}

    def test_aborts_when_coverage_below_threshold(self, capsys):
        with pytest.raises(SystemExit, match="覆盖率"):
            report_issues("x.xlsx", self._issues(300, 100), force=False)

    def test_force_overrides_coverage_gate(self):
        report_issues("x.xlsx", self._issues(300, 100), force=True)

    def test_full_coverage_passes(self):
        report_issues("x.xlsx", self._issues(300, 0), force=False)

    def test_90_percent_coverage_is_the_boundary(self):
        report_issues("x.xlsx", self._issues(300, 30), force=False)   # 恰好 90%
        with pytest.raises(SystemExit):
            report_issues("x.xlsx", self._issues(300, 31), force=False)


class TestAppendToGold:
    GOLD = [
        {"id": "g1", "text": "老行", "platform": "weibo",
         "content_type": "评论", "gold_label": "中性"},
    ]

    def _incoming(self, texts):
        return [{"id": f"ge7b_{i}", "text": t, "platform": "bilibili",
                 "content_type": "评论", "gold_label": "负面", "domain": "异环"}
                for i, t in enumerate(texts)]

    def test_appends_and_keeps_original_rows_untouched(self):
        merged, dedup = append_to_gold(list(self.GOLD), self._incoming(["新1", "新2"]))
        assert merged[0] == self.GOLD[0]
        assert len(merged) == 3
        assert dedup == {"dup_text": 0, "dup_id": 0}

    def test_duplicate_text_against_existing_gold_skipped(self):
        merged, dedup = append_to_gold(list(self.GOLD), self._incoming(["老行", "新2"]))
        assert len(merged) == 2 and dedup["dup_text"] == 1

    def test_duplicate_text_within_incoming_skipped(self):
        merged, dedup = append_to_gold(list(self.GOLD), self._incoming(["新1", "新1"]))
        assert len(merged) == 2 and dedup["dup_text"] == 1

    def test_gold_record_drops_working_fields(self):
        # gold 行只保留评估需要的字段，domain/channel 不进 gold
        rec = to_gold_record(self._incoming(["新1"])[0])
        assert set(rec) == {"id", "text", "platform", "content_type", "gold_label"}
