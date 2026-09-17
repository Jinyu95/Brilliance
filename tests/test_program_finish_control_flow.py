"""A failed terminal solve must not fall through and overwrite its result."""
import ast

from brilliance.research_program import _normal_finish_states, inspect_research_program


def check(source):
    issues = []
    states = _normal_finish_states(ast.parse(source).body, issues=issues)
    return states, issues


def test_failed_branch_fallthrough_is_detected():
    states, issues = check('''
if not matched:
    run.finish(status="infeasible")
construct_ring()
run.finish(summary="Matched ring")
''')
    assert states == {True}
    assert len(issues) == 1
    assert "overwrite an earlier result" in issues[0]


def test_explicit_early_exit_is_safe():
    states, issues = check('''
if not matched:
    run.finish(status="infeasible")
    raise SystemExit(0)
construct_ring()
run.finish(summary="Matched ring")
''')
    assert states == {True}
    assert issues == []


def test_mutually_exclusive_finishes_are_safe():
    states, issues = check('''
if matched:
    run.finish(summary="Matched ring")
else:
    run.finish(status="infeasible")
''')
    assert states == {True}
    assert issues == []


def test_progress_is_not_a_final_commit():
    assert check('run.progress("seed")\nrun.progress("matching")\nrun.finish(summary="done")')[1] == []


def test_unreachable_code_does_not_restore_a_finished_path():
    assert check('raise SystemExit(0)\nrun.finish()\nrun.finish()') == (set(), [])
    assert _normal_finish_states(ast.parse('run.finish()').body, set()) == set()


def test_execution_preflight_rejects_overwrite_before_starting_a_worker(tmp_path):
    script = tmp_path / "program.py"
    script.write_text('''
from brilliance.research_api import open_research_run
run = open_research_run()
if not run.arguments.get("success"):
    run.finish(summary="match failed", status="infeasible")
run.finish(summary="matched ring")
''', encoding="utf-8")
    inspection = inspect_research_program(tmp_path, "program.py")
    assert inspection["passed"] is False
    assert any("overwrite an earlier result" in issue for issue in inspection["issues"])
