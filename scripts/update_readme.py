"""Rebuild the README's status sections from live GitHub issues and pull requests.

    python -m scripts.update_readme            # rewrite README.md in place
    python -m scripts.update_readme --check    # exit 1 if README.md is out of date (no changes made)

Runs automatically in the "README status" GitHub Action whenever an issue or pull request changes.
Only the text between the markers is replaced:
    <!-- status:badge -->     issues-closed badge in the header
    <!-- status:progress -->  progress per rubric part
    <!-- status:people -->    "What's left, by person": one card per open issue (text from docs/issue_cards.yaml)

Rules: a closed issue disappears from the cards; an open issue linked by an open pull request
(`Closes #12`, `Fixes #12`, `Resolves #12` or `Refs #12` in its title or description) shows "in review"; otherwise the card's
`status` from docs/issue_cards.yaml is used. Needs `gh` (authenticated) and PyYAML.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CARDS = ROOT / "docs" / "issue_cards.yaml"
ACCEPTANCE = ROOT / "tests" / "acceptance" / "test_issues.py"
REPO = "ASU-MICS-2028/mle-group-2-prosit-2"
URL = f"https://github.com/{REPO}"

STATUS = {  # key: (badge label, colour, meaning)
    "review": ("in_review", "1f4e79", "a pull request for it is open"),
    "merged": ("merged,_verify_%26_close", "2e7d32", "work is on main; owner verifies and closes"),
    "partly": ("partly_done", "d4a72c", "evidence exists, owner must finish"),
    "trained": ("trained,_review", "d4a72c", "models already ran, owner must review"),
    "todo": ("to_do", "9aa0a6", "not started"),
    "urgent": ("start_now", "b03a2e", "start today"),
}
PARTS = [  # (label, name, rubric)
    ("setup", "Setup", "—"),
    ("part-A", "A · Frame the problem", "10 pts"),
    ("part-B", "B · Build & evaluate", "30 + 20 pts"),
    ("part-C", "C · Cross-market transfer", "in B, D"),
    ("part-D", "D · Ethics & regulation", "20 pts"),
    ("part-E", "E · Deployment", "15 pts"),
    ("deliverable", "Deliverables", "5 pts + all"),
]


# --------------------------------------------------------------------------- GitHub
def gh_json(*args: str) -> list[dict]:
    out = subprocess.run(["gh", *args, "-R", REPO], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def fetch() -> tuple[dict[int, dict], list[dict]]:
    issues = gh_json("issue", "list", "--state", "all", "--limit", "500", "--json", "number,title,state,labels,assignees")
    prs = gh_json("pr", "list", "--state", "open", "--limit", "100", "--json", "number,title,body")
    return {i["number"]: i for i in issues}, prs


# --------------------------------------------------------------------------- pure helpers
ACCEPTANCE_TEST = re.compile(r'@issue\((\d+)\)\s*\ndef \w+\([^)]*\):\s*\n\s*"""(.+?)(?:\n|""")')


def acceptance_checks(source: str) -> dict[int, str]:
    """issue number -> first docstring line of its acceptance test in tests/acceptance/test_issues.py."""
    return {int(n): doc.strip() for n, doc in ACCEPTANCE_TEST.findall(source)}


LINK = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?|refs?)\s+#(\d+)", re.I)


def referenced_issues(prs: list[dict]) -> dict[int, list[int]]:
    """issue number -> open PRs that link it with Closes/Fixes/Resolves/Refs (a passing mention does not count)."""
    refs: dict[int, list[int]] = {}
    for pr in prs:
        for n in {int(m) for m in LINK.findall(f"{pr.get('title', '')}\n{pr.get('body') or ''}")}:
            if n != pr["number"]:
                refs.setdefault(n, []).append(pr["number"])
    return refs


def badge(status: str) -> str:
    label, colour, meaning = STATUS[status]
    return f'<img alt="{meaning}" src="https://img.shields.io/badge/-{label}-{colour}?style=flat-square">'


def labels_of(issue: dict) -> set[str]:
    return {lab["name"] for lab in issue.get("labels", [])}


def effective_status(card: dict, issue: dict, refs: dict[int, list[int]]) -> str:
    return "review" if card["issue"] in refs else card["status"]


def render_badge(issues: dict[int, dict]) -> str:
    tracked = [i for n, i in issues.items() if "roadmap" not in labels_of(i)]
    closed = sum(i["state"] == "CLOSED" for i in tracked)
    colour = "2e7d32" if closed == len(tracked) else "d4a72c"
    return f'<img alt="Issues" src="https://img.shields.io/badge/issues-{closed}_of_{len(tracked)}_closed-{colour}?style=flat-square">'


def render_progress(issues: dict[int, dict], refs: dict[int, list[int]], today: str) -> str:
    rows = [
        f"Generated from GitHub on **{today}** by `scripts/update_readme.py` (the *README status* Action keeps it current).\n\n",
        "| Part | Rubric | Closed | Still open |\n| --- | :---: | :---: | --- |\n",
    ]
    for label, name, rubric in PARTS:
        part = sorted((n, i) for n, i in issues.items() if label in labels_of(i))
        if not part:
            continue
        closed = sum(i["state"] == "CLOSED" for _, i in part)
        colour = "2e7d32" if closed == len(part) else ("9aa0a6" if closed == 0 else "d4a72c")
        open_links = " ".join(
            f"[#{n}]({URL}/issues/{n}){' ⏳' if n in refs else ''}" for n, i in part if i["state"] == "OPEN"
        ) or "—"
        rows.append(
            f"| **{name}** | {rubric} | ![{closed}/{len(part)}](https://img.shields.io/badge/{closed}%2F{len(part)}-closed-{colour}?style=flat-square) | {open_links} |\n"
        )
    rows.append("\n<sub>⏳ = a pull request for it is open.</sub>\n")
    return "".join(rows)


def render_card(
    card: dict, issue: dict, status: str, refs: dict[int, list[int]], checks: dict[int, str] | None = None
) -> str:
    n = card["issue"]
    check = (checks or {}).get(n)
    check_row = (
        f"| **Automated check** | {check} Run `make acceptance ISSUE={n}`; it also runs on every pull request that "
        f"says `Closes #{n}` and when the issue is closed (a failing check reopens it). |\n"
        if check
        else ""
    )
    steps = "".join(f"{k}. {s}\n" for k, s in enumerate(card["steps"], 1))
    review = (
        f"| **Pull request** | {', '.join(f'[#{p}]({URL}/pull/{p})' for p in refs[n])} |\n" if n in refs else ""
    )
    return (
        f"<details>\n<summary>{badge(status)} <b>#{n} · {issue['title']}</b></summary>\n\n"
        "| | |\n| --- | --- |\n"
        f"| **What it is** | {card['what']} |\n"
        f"| **Why it matters** | {card['why']} |\n"
        f"| **Already done** | {card['already']} |\n"
        f"| **Where to work** | {card['where']} |\n"
        f"| **How to run** | {card['run']} |\n"
        f"| **Done when** | {card['done_when']} |\n"
        f"{check_row}"
        f"{review}\n"
        f"**Steps**\n\n{steps}\n"
        f"[Open issue #{n} on GitHub →]({URL}/issues/{n})\n\n</details>\n\n"
    )


def render_people(
    cards: dict, issues: dict[int, dict], refs: dict[int, list[int]], checks: dict[int, str] | None = None
) -> tuple[str, list[int]]:
    """The whole "What's left, by person" section, plus open issues that have no card (to warn about)."""
    legend = " &nbsp; ".join(f"{badge(k)} {v[2]}" for k, v in STATUS.items())

    def live(cs):
        return [(c, issues[c["issue"]]) for c in cs if c["issue"] in issues and issues[c["issue"]]["state"] == "OPEN"]

    def counts(cs):
        sts = [effective_status(c, i, refs) for c, i in live(cs)]
        return len(sts), sts.count("review") + sts.count("merged"), sum(s in ("todo", "urgent") for s in sts)

    out = [
        "Find your name, open each issue card and follow the steps. Every card says what the task is, why it matters "
        "for the grade, what already exists, where to work, how to run it, and when it is done. Close an issue with a "
        "pull request whose description says `Closes #<number>`. Every open issue is **assigned on GitHub** to its "
        f"owner; signed in, [see the issues assigned to you]({URL}/issues?q=is%3Aopen+assignee%3A%40me).\n\n",
        f"<sub>{legend}</sub>\n\n",
        "| | Person | Open | In review or merged | Not started | Jump to |\n| :---: | --- | :---: | :---: | :---: | --- |\n",
    ]
    for p in cards["people"]:
        o, r, t = counts(p["issues"])
        first = p["name"].split()[0]
        out.append(
            f'| <img src="https://github.com/{p["github"]}.png?size=60" width="28"> | **{p["name"]}** · {p["role"]} '
            f"| {o} | {r} | {t} | [{first}'s issues](#{p['github'].lower()}) |\n"
        )
    o, r, t = counts(cards["everyone"])
    out.append(f"| 👥 | **Everyone** | {o} | {r} | {t} | [Shared issues](#everyone) |\n")
    out.append(
        "\n### How to pick up an issue\n\n```bash\n"
        "git switch main && git pull                      # start from the latest main\n"
        "git switch -c p4/54-fix-notebook-04              # <role>/<issue>-<short-name>\n"
        "# ... do the steps on the card, run `make test` ...\n"
        'git add -A && git commit -m "Fix notebook 04 calibration (#54)"\n'
        "git push -u origin p4/54-fix-notebook-04\n"
        "gh pr create --fill                               # description must say: Closes #54\n"
        "```\n\nThen add a line to `docs/WORKLOG.md` and `docs/PROGRESS.md` (a check enforces it) and ask the next "
        "person in the review rotation to approve. The cards below update themselves when issues and pull requests change.\n"
    )
    for p in cards["people"]:
        o, r, t = counts(p["issues"])
        out.append(
            f'\n<h3 id="{p["github"].lower()}"><img src="https://github.com/{p["github"]}.png?size=60" width="32" align="center"> {p["name"]}</h3>\n\n'
            f"{p['role']} · [{p['github']}](https://github.com/{p['github']}) · **{o} open** · "
            f"[{p['name'].split()[0]}'s assigned issues on GitHub]({URL}/issues?q=is%3Aopen+assignee%3A{p['github']})\n\n"
        )
        if not o:
            out.append("All of this person's issues are closed. ✅\n\n")
        out.extend(render_card(c, i, effective_status(c, i, refs), refs, checks) for c, i in live(p["issues"]))
    out.append('\n<h3 id="everyone">👥 Everyone</h3>\n\n')
    out.extend(render_card(c, i, effective_status(c, i, refs), refs, checks) for c, i in live(cards["everyone"]))

    carded = {c["issue"] for p in cards["people"] for c in p["issues"]} | {c["issue"] for c in cards["everyone"]}
    uncarded = sorted(
        n for n, i in issues.items() if i["state"] == "OPEN" and n not in carded and "roadmap" not in labels_of(i)
    )
    if uncarded:
        out.append(
            "\n**Open issues without a card yet** (add them to `docs/issue_cards.yaml`): "
            + " · ".join(f"[#{n} {issues[n]['title']}]({URL}/issues/{n})" for n in uncarded)
            + "\n"
        )
    return "".join(out), uncarded


def replace_block(text: str, name: str, body: str) -> str:
    start, end = f"<!-- status:{name} -->", f"<!-- /status:{name} -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        raise ValueError(f"README is missing the markers {start} ... {end}")
    sep = "" if name == "badge" else "\n"
    return pattern.sub(lambda _: f"{start}{sep}{body}{sep}{end}", text)


def build(
    readme: str,
    cards: dict,
    issues: dict[int, dict],
    prs: list[dict],
    today: str,
    checks: dict[int, str] | None = None,
) -> tuple[str, list[int]]:
    refs = referenced_issues(prs)
    people, uncarded = render_people(cards, issues, refs, checks)
    text = replace_block(readme, "badge", render_badge(issues))
    text = replace_block(text, "progress", render_progress(issues, refs, today))
    text = replace_block(text, "people", people)
    return text, uncarded


# --------------------------------------------------------------------------- entry point
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if README.md would change")
    args = parser.parse_args()

    cards = yaml.safe_load(CARDS.read_text(encoding="utf-8"))
    issues, prs = fetch()
    current = README.read_text(encoding="utf-8")
    today = dt.date.today().strftime("%d %b %Y")
    checks = acceptance_checks(ACCEPTANCE.read_text(encoding="utf-8")) if ACCEPTANCE.exists() else {}
    new, uncarded = build(current, cards, issues, prs, today, checks)

    def strip_date(s: str) -> str:
        return re.sub(r"Generated from GitHub on \*\*.*?\*\*", "", s)

    changed = strip_date(new) != strip_date(current)
    if uncarded:
        print(f"note: open issues without a card: {uncarded}")
    if args.check:
        print("README status is up to date" if not changed else "README status is out of date: run python -m scripts.update_readme")
        sys.exit(1 if changed else 0)
    if changed:
        README.write_text(new, encoding="utf-8")
    closed = sum(i["state"] == "CLOSED" for i in issues.values() if "roadmap" not in labels_of(i))
    print(f"README status {'updated' if changed else 'unchanged'}: {closed}/{len(issues) - 1} issues closed, {len(prs)} open PR(s)")


if __name__ == "__main__":
    main()
