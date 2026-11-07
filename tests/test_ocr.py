import logging

from fixtures import make_pages
from pytest import approx

from cvo.ocr import run_ocr


class CountingEngine:
    def __init__(self):
        self.calls = 0

    def read(self, image):
        self.calls += 1
        return f"text {self.calls}"


class FailingEngine:
    """Raises on the call with index `fail_on` (0-based call counter)."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def read(self, image):
        n = self.calls
        self.calls += 1
        if n == self.fail_on:
            raise RuntimeError("boom")
        return f"text {n}"


def test_ocr_caches_and_reuses(tmp_path):
    eng = CountingEngine()
    pages = make_pages(3)
    first = run_ocr(pages, eng, tmp_path)
    second = run_ocr(pages, eng, tmp_path)
    assert eng.calls == 3 and first.texts == second.texts
    assert second.seconds == approx(first.seconds)  # recorded time, not cache-load time


def test_ocr_failure_recorded_not_dropped(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        res = run_ocr(make_pages(2), FailingEngine(fail_on=1), tmp_path)
    assert set(res.texts) == {"p0", "p1"} and res.texts["p1"] == "" and res.pages_failed == 1
    assert "p1" in caplog.text
