"""The README status generator: closed issues vanish, open PRs mark issues in review, counts are right."""
from __future__ import annotations

import pytest
import yaml

from scripts import update_readme as U

README = """<!-- status:badge -->x<!-- /status:badge -->
## Progress
<!-- status:progress -->
old
<!-- /status:progress -->
## What's left, by person
<!-- status:people -->
old
<!-- /status:people -->
"""

CARD = {"what": "w", "why": "y", "already": "a", "where": "f", "run": "r", "done_when": "d", "steps": ["one", "two"]}
CARDS = {
    "people": [
        {"name": "Ada Lovelace", "github": "ada", "label": "P1", "role": "Data",
         "issues": [{"issue": 2, "status": "todo", **CARD}, {"issue": 3, "status": "partly", **CARD}]},
    ],
    "everyone": [{"issue": 4, "status": "urgent", **CARD}],
}


def issue(n, state="OPEN", labels=("part-A", "P1")):
    return {"number": n, "title": f"Issue {n}", "state": state, "labels": [{"name": lab} for lab in labels], "assignees": []}


ISSUES = {1: issue(1, labels=("roadmap",)), 2: issue(2, "CLOSED"), 3: issue(3), 4: issue(4, labels=("deliverable", "all")), 5: issue(5)}
PRS = [{"number": 9, "title": "Fix things", "body": "Closes #3\nRefs #99"}]


def build():
    return U.build(README, CARDS, ISSUES, PRS, "17 Nov 2026")


def test_closed_issues_disappear_and_open_ones_get_cards() -> None:
    text, _ = build()
    assert "#2 · Issue 2" not in text
    assert "#3 · Issue 3" in text and "#4 · Issue 4" in text


def test_issue_referenced_by_an_open_pr_shows_in_review() -> None:
    text, _ = build()
    card3 = text.split("#3 · Issue 3")[0].rsplit("<summary>", 1)[1]
    assert "in_review" in card3
    assert "[#9](https://github.com/" in text  # the PR is linked from the card


def test_only_linking_keywords_count_as_in_review() -> None:
    refs = U.referenced_issues([{"number": 7, "title": "x", "body": "Closes #12, fixes #13\nRefs #14\nSee #15 and PR #47 for context"}])
    assert set(refs) == {12, 13, 14}


def test_badge_and_progress_counts() -> None:
    text, _ = build()
    assert "issues-1_of_4_closed" in text  # roadmap #1 is not counted
    assert "1%2F3-closed" in text  # part-A: #2 closed of #2, #3, #5
    assert "17 Nov 2026" in text


def test_open_issue_without_a_card_is_reported() -> None:
    text, uncarded = build()
    assert uncarded == [5]
    assert "without a card" in text


def test_markers_are_required() -> None:
    with pytest.raises(ValueError, match="markers"):
        U.replace_block("no markers here", "people", "x")


def test_rebuild_is_idempotent() -> None:
    once, _ = build()
    twice, _ = U.build(once, CARDS, ISSUES, PRS, "17 Nov 2026")
    assert once == twice


def test_real_cards_file_is_valid_and_readme_has_markers() -> None:
    cards = yaml.safe_load(U.CARDS.read_text(encoding="utf-8"))
    every = [c for p in cards["people"] for c in p["issues"]] + cards["everyone"]
    for c in every:
        assert set(CARD) | {"issue", "status"} <= set(c), c.get("issue")
        assert c["status"] in U.STATUS and c["status"] != "review", f"#{c['issue']}: 'review' is set automatically"
        assert c["steps"], f"#{c['issue']} needs steps"
    readme = U.README.read_text(encoding="utf-8")
    for name in ("badge", "progress", "people"):
        assert f"<!-- status:{name} -->" in readme and f"<!-- /status:{name} -->" in readme
