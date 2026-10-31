"""Unified diffs, hunk application and the undo journal behind ``effgen code``.

These are the net-new edit layer: generating a diff from two strings, applying a
diff hunk-by-hunk against content that may have changed underneath (so the hunks
that still match apply and the rest are reported, never a whole-file clobber),
and the per-workspace undo journal that reverses an applied write. No model calls
and no CLI: the pieces are driven directly on strings and a temp workspace.
"""

from __future__ import annotations

from pathlib import Path

from effgen.cli.code.diffs import (
    apply_hunks,
    diff_stat,
    render_diff,
    split_hunks,
    unified_diff_text,
)
from effgen.cli.code.edits import EditJournal, ProposedEdit

# -- diff generation --------------------------------------------------------

def test_unified_diff_and_stat():
    old = "def f():\n    return 1\n"
    new = "def f():\n    return 2\n"
    text = unified_diff_text(old, new, "f.py")
    assert "a/f.py" in text and "b/f.py" in text
    assert "-    return 1" in text
    assert "+    return 2" in text
    stat = diff_stat(old, new)
    assert (stat.added, stat.removed) == (1, 1)
    assert str(stat) == "+1/-1"


def test_new_file_diff_is_all_additions():
    text = unified_diff_text("", "a\nb\n", "new.py")
    assert "+a" in text and "+b" in text
    stat = diff_stat("", "a\nb\n")
    assert (stat.added, stat.removed) == (2, 0)


def test_identical_content_has_empty_diff():
    assert unified_diff_text("same\n", "same\n", "x.py") == ""


def test_missing_final_newline_annotated():
    text = unified_diff_text("a\n", "a\nb", "x.py")
    assert "No newline at end of file" in text


# -- hunk application -------------------------------------------------------

def test_apply_reproduces_new_content_when_unchanged():
    old = "".join(f"line{i}\n" for i in range(1, 6))
    new = old.replace("line3\n", "CHANGED\n")
    hunks = split_hunks(old, new)
    result = apply_hunks(old, hunks, list(range(len(hunks))))
    assert result.clean
    assert result.text == new


def test_rejecting_a_hunk_leaves_that_region():
    old = "".join(f"line{i}\n" for i in range(1, 21))
    new = old.replace("line2\n", "A2\n").replace("line18\n", "B18\n")
    hunks = split_hunks(old, new)
    assert len(hunks) == 2
    # Accept only the second hunk.
    result = apply_hunks(old, hunks, [1])
    assert "B18" in result.text
    assert "A2" not in result.text
    assert "line2" in result.text


def test_changed_underneath_applies_the_rest_and_reports_the_failure():
    old = "".join(f"line{i}\n" for i in range(1, 21))
    new = old.replace("line2\n", "A2\n").replace("line18\n", "B18\n")
    hunks = split_hunks(old, new)
    # The file changed near the first hunk since the diff was prepared.
    current = old.replace("line2\n", "ZZZ\n")
    result = apply_hunks(current, hunks, list(range(len(hunks))))
    assert result.failed == [0]
    assert result.applied == [1]
    assert "B18" in result.text     # the still-matching hunk applied
    assert "ZZZ" in result.text     # the user's underneath change was kept
    assert "A2" not in result.text  # the failed hunk did not force its change


def test_all_hunks_fail_leaves_content_untouched():
    old = "a\nb\nc\n"
    new = "a\nX\nc\n"
    hunks = split_hunks(old, new)
    current = "completely\ndifferent\nfile\ncontents\n"
    result = apply_hunks(current, hunks, list(range(len(hunks))))
    assert result.failed == [0]
    assert result.text == current  # no corruption


# -- rendering --------------------------------------------------------------

def test_render_diff_styles_and_escapes():
    text = unified_diff_text("x = old\n", "x = arr[0]\n", "x.py")
    plain, markup = render_diff(text)
    assert plain == "\n".join(text.splitlines())
    assert "effgen.success" in markup   # an added line is styled
    assert "effgen.error" in markup     # a removed line is styled
    assert "effgen.info" in markup      # the @@ header is styled
    assert r"\[0]" in markup            # a bracket in content is escaped


# -- proposed edit ----------------------------------------------------------

def test_proposed_edit_resolve_against_current():
    edit = ProposedEdit(
        rel_path="f.py",
        old_content="a\nb\nc\n",
        new_content="a\nB\nc\n",
        is_new=False,
    )
    # Unchanged baseline reproduces the proposal exactly.
    text, applied, failed = edit.resolve_against_current("a\nb\nc\n")
    assert text == "a\nB\nc\n"
    assert failed == []
    # A changed baseline where the hunk no longer matches reports the failure.
    text2, _applied2, failed2 = edit.resolve_against_current("totally\nother\n")
    assert failed2  # reported, not forced
    assert text2 == "totally\nother\n"


# -- undo journal -----------------------------------------------------------

def test_journal_records_and_undoes_a_restore(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    root = tmp_path / "hist"
    (ws / "a.py").write_text("v1\n")

    journal = EditJournal(ws, root=root)
    journal.record_write("a.py", before="v1\n", after="v2\n")
    (ws / "a.py").write_text("v2\n")
    assert len(journal) == 1

    outcome = journal.undo()
    assert outcome is not None and outcome.action == "restored"
    assert (ws / "a.py").read_text() == "v1\n"
    assert len(journal) == 0
    assert journal.undo() is None  # empty stack


def test_journal_undo_removes_a_created_file(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    root = tmp_path / "hist"
    (ws / "new.py").write_text("created\n")

    journal = EditJournal(ws, root=root)
    journal.record_write("new.py", before=None, after="created\n")
    outcome = journal.undo()
    assert outcome is not None and outcome.action == "removed"
    assert not (ws / "new.py").exists()


def test_journal_walks_back_multiple_edits(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    root = tmp_path / "hist"
    f = ws / "a.py"
    f.write_text("v3\n")

    journal = EditJournal(ws, root=root)
    journal.record_write("a.py", before=None, after="v1\n")
    journal.record_write("a.py", before="v1\n", after="v2\n")
    journal.record_write("a.py", before="v2\n", after="v3\n")

    assert journal.undo().action == "restored"
    assert f.read_text() == "v2\n"
    assert journal.undo().action == "restored"
    assert f.read_text() == "v1\n"
    assert journal.undo().action == "removed"
    assert not f.exists()


def test_journal_is_bounded(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    journal = EditJournal(ws, root=tmp_path / "hist")
    for i in range(150):
        journal.record_write("a.py", before=str(i), after=str(i + 1))
    # Bounded to the most recent edits, not the full history.
    assert 0 < len(journal) <= 100
