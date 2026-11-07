import os
import re

import pytest

from cvo.artifacts import arm_dir, load_json, page_key, run_metadata, save_json


def test_page_key_is_filename_safe():
    assert re.fullmatch(r"[0-9a-f]{40}", page_key("a/b:c.png"))


def test_atomic_write_leaves_no_tmp(tmp_path):
    p = tmp_path / "x.json"
    save_json(p, {"é": 1})
    assert load_json(p) == {"é": 1} and not list(tmp_path.glob("*.tmp"))


def test_crash_before_replace_leaves_no_target(tmp_path, monkeypatch):
    def boom(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(KeyboardInterrupt):
        save_json(tmp_path / "x.json", {"a": 1})
    assert not (tmp_path / "x.json").exists()


def test_arm_dir_sanitizes_model_id(tmp_path):
    d = arm_dir(tmp_path, "tatdqa", "colqwen", "vidore/colqwen2-v1.0")
    assert d == tmp_path / "tatdqa" / "colqwen" / "vidore__colqwen2-v1.0" and d.is_dir()


def test_run_metadata_keys():
    assert {"timestamp", "python", "versions", "gpu", "limit"} <= run_metadata(50).keys()
