"""The orphan audit has to survive being written about.

This repo keeps producing capabilities nothing calls, and the audit exists to
find them. Its first version counted occurrences of the name with a regex over
the raw text, which had a failure that inverted the whole tool: recording *why*
an orphan is deliberate -- "the release half of acquire_browser_work_slot" --
put a second occurrence of the name in the file, so the audit counted two hits
and stopped reporting it. Explaining a finding made the finding disappear, and
the two functions the note was about vanished from the report the moment they
were documented.

It also read the marker only within three lines above the def, which quietly
rewarded a terse `# orphan-ok: intentional` over a real explanation: a
six-line justification pushed the marker out of range and the orphan came back.

Everything here is hermetic -- find_orphans takes the sources it should read.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import audit_orphans as a  # noqa: E402

SRC = a.REPO / "src" / "fake_module.py"
OTHER = a.REPO / "src" / "other_module.py"


def _names(sources):
    return [name for _loc, name in a.find_orphans(sources)]


# ── the basic question ──────────────────────────────────────────────────────

def test_a_function_nothing_calls_is_reported():
    assert _names({SRC: "def widget():\n    return 1\n"}) == ["widget"]


def test_a_function_called_from_another_module_is_not_reported():
    assert _names({
        SRC: "def widget():\n    return 1\n",
        OTHER: "import fake_module\n\nfake_module.widget()\n",
    }) == []


def test_a_function_used_only_inside_its_own_module_is_not_reported():
    """geo_priority.match_keys has no external caller and never needed one --
    rank and partition both use it. Reporting those would train people to
    ignore the report."""
    assert _names({SRC: "def helper():\n    return 1\n\n\ndef top():\n    return helper()\n"}) == ["top"]


def test_private_functions_are_not_reported():
    assert _names({SRC: "def _internal():\n    return 1\n"}) == []


# ── the regression that made the tool lie ───────────────────────────────────

def test_a_comment_mentioning_the_name_does_not_count_as_a_caller():
    """The bug this tool shipped with. Writing "widget is unused" in a comment
    made widget stop being reported as unused."""
    assert _names({
        SRC: "def widget():\n    return 1\n",
        OTHER: "# widget is unwired on purpose, see the note there\n",
    }) == ["widget"]


def test_a_docstring_mentioning_the_name_does_not_count_as_a_caller():
    assert _names({
        SRC: '"""This module owns widget."""\n\n\ndef widget():\n    return 1\n',
    }) == ["widget"]


def test_the_orphans_own_explanation_does_not_silence_the_next_one():
    """Two orphans, one documented and one not. The documented one's note names
    the other -- which is exactly how the real pair read -- and that must not
    excuse the undocumented one."""
    source = (
        "# orphan-ok: kept beside release_slot, which nothing calls either\n"
        "def acquire_slot():\n    return 1\n\n\n"
        "def release_slot():\n    return 2\n"
    )
    assert _names({SRC: source}) == ["release_slot"]


def test_a_name_dispatched_by_string_counts_as_used():
    """getattr(module, "handler") is a real caller. Reporting it as uncalled
    would be a false accusation, and one deletion away from an outage."""
    assert _names({
        SRC: "def handler():\n    return 1\n",
        OTHER: 'import fake_module\n\ngetattr(fake_module, "handler")()\n',
    }) == []


# ── the marker ──────────────────────────────────────────────────────────────

def test_a_marker_on_the_line_above_excuses_it():
    assert _names({SRC: "# orphan-ok: kept for the CLI\ndef widget():\n    return 1\n"}) == []


def test_a_marker_at_the_top_of_a_long_comment_block_still_excuses_it():
    """A reason worth recording runs to several lines. Scanning a fixed three
    made the audit reward "# orphan-ok: intentional" over an explanation."""
    block = "".join(f"# padding line {i}\n" for i in range(8))
    source = "# orphan-ok: the last caller was deliberately removed\n" + block + "def widget():\n    return 1\n"
    assert _names({SRC: source}) == []


def test_a_marker_above_a_decorator_still_excuses_it():
    source = ("# orphan-ok: registered by the framework\n"
              "@register\ndef widget():\n    return 1\n")
    assert _names({SRC: source}) == []


def test_a_comment_block_without_the_marker_does_not_excuse_it():
    source = "# this one is fine honestly\n# nothing to see\ndef widget():\n    return 1\n"
    assert _names({SRC: source}) == ["widget"]


def test_a_marker_further_up_the_file_does_not_excuse_an_unrelated_function():
    """The block has to be contiguous with the def, or one marker anywhere
    silences everything below it."""
    source = ("# orphan-ok: this is about the other one\n"
              "def excused():\n    return 1\n\n\n"
              "def widget():\n    return 2\n")
    assert _names({SRC: source}) == ["widget"]


def test_the_reason_is_required_not_just_the_word():
    """`# orphan-ok` on its own records nothing, and a finding silenced without
    a reason is the state this script was written to end."""
    assert _names({SRC: "# orphan-ok\ndef widget():\n    return 1\n"}) == ["widget"]
    assert _names({SRC: "# orphan-ok:\ndef widget():\n    return 1\n"}) == ["widget"]


def test_the_colon_is_optional_but_the_reason_is_not():
    """Someone writing the marker from memory should not be silently ignored;
    what is being enforced is the explanation, not the punctuation."""
    source = "# orphan-ok kept for the CLI\ndef widget():\n    return 1\n"
    assert _names({SRC: source}) == []


def test_a_word_that_merely_starts_with_the_marker_is_not_one():
    source = "# orphan-okay whatever\ndef widget():\n    return 1\n"
    assert _names({SRC: source}) == ["widget"]


# ── robustness ──────────────────────────────────────────────────────────────

def test_a_file_that_does_not_parse_does_not_stop_the_audit():
    assert _names({
        a.REPO / "src" / "broken.py": "def (((:\n",
        SRC: "def widget():\n    return 1\n",
    }) == ["widget"]


def test_findings_are_sorted_so_the_report_is_diffable():
    sources = {
        OTHER: "def zeta():\n    return 1\n",
        SRC: "def alpha():\n    return 1\n",
    }
    assert a.find_orphans(sources) == sorted(a.find_orphans(sources))


# ── the exit code, which is what enforces anything ───────────────────

# There is deliberately no test asserting the tree itself is orphan-free.
# Several modules here carry local work that is not committed, and the callers
# of half a dozen public functions live in exactly those files -- so the same
# audit reports eleven orphans against a clean checkout of HEAD and none
# against this working tree, and neither number is wrong. A test pinned to
# either would fail for whoever is holding the other, which is how a check
# stops being read. The script is the enforcement; these tests are what make
# the script trustworthy enough to act on.

def test_findings_make_the_run_fail(monkeypatch, capsys):
    """Exit 1, so a CI step or a pre-commit hook can act on it. A reporter that
    always exits 0 is a reporter nobody notices."""
    monkeypatch.setattr(a, "find_orphans", lambda: [("src/x.py", "widget")])
    monkeypatch.setattr(sys, "argv", ["audit_orphans.py"])
    assert a.main() == 1
    assert "widget" in capsys.readouterr().out


def test_list_reports_the_same_findings_without_failing(monkeypatch, capsys):
    """The loop reads this every few minutes; it should not have to treat a
    known, explained backlog as a broken run."""
    monkeypatch.setattr(a, "find_orphans", lambda: [("src/x.py", "widget")])
    monkeypatch.setattr(sys, "argv", ["audit_orphans.py", "--list"])
    assert a.main() == 0
    assert "widget" in capsys.readouterr().out


def test_a_clean_tree_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(a, "find_orphans", lambda: [])
    monkeypatch.setattr(sys, "argv", ["audit_orphans.py"])
    assert a.main() == 0
    assert "none" in capsys.readouterr().out


def test_the_report_says_where_each_one_is(monkeypatch, capsys):
    """A name with no path is not actionable; there are several `refresh`es."""
    monkeypatch.setattr(a, "find_orphans", lambda: [("src/services/jba/geo_priority.py", "refresh")])
    monkeypatch.setattr(sys, "argv", ["audit_orphans.py", "--list"])
    a.main()
    assert "src/services/jba/geo_priority.py:refresh" in capsys.readouterr().out


def test_the_report_says_what_to_do_about_them(monkeypatch, capsys):
    """Including the marker, or the next person silences the finding by
    deleting the function that was deliberate."""
    monkeypatch.setattr(a, "find_orphans", lambda: [("src/x.py", "widget")])
    monkeypatch.setattr(sys, "argv", ["audit_orphans.py", "--list"])
    a.main()
    assert "orphan-ok" in capsys.readouterr().out
