"""The borrower app renders, and only calls the shared scorer (no decision logic of its own)."""
from __future__ import annotations

import pytest

from src import config as C

APP = C.ROOT / "app" / "streamlit_app.py"


def test_app_renders_without_errors() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert any("Quick cash" in m.value for m in at.markdown)


def test_app_uses_the_scorer_and_shows_borrower_reasons() -> None:
    src = APP.read_text(encoding="utf-8")
    assert "score_application(" in src and 'r["borrower_reasons"]' in src
    for decision in ("approve", "limit", "starter", "decline"):
        assert f'"{decision}"' in src
