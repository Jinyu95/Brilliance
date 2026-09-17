"""ResearchRun call contracts are checked without opening a run or Julia."""
import ast

import pytest

from brilliance.public_api_lint import lint_public_api_calls


@pytest.mark.parametrize("setup", [
    "from brilliance.research_api import open_research_run\nrun = open_research_run()",
    "from brilliance.research_api import open_research_run as start\nrun = start()",
    "from brilliance import research_api as research\nrun = research.open_research_run()",
    "import brilliance.research_api as research\nrun = research.open_research_run()",
    "import brilliance.research_api\nrun = brilliance.research_api.open_research_run()",
    "import brilliance\nrun = brilliance.research_api.open_research_run()",
    "from brilliance.research_api import open_research_run\nfirst = open_research_run()\nrun = first",
])
def test_proven_factory_handles_reject_missing_methods(setup):
    issues = lint_public_api_calls(ast.parse(setup + "\nrun.check_cancelled()"))
    assert len(issues) == 1
    assert "ResearchRun" in issues[0] and "check_cancelled" in issues[0]


def test_real_signatures_check_keywords_and_required_arguments(monkeypatch):
    from brilliance import research_api

    monkeypatch.setattr(research_api, "open_research_run", lambda: pytest.fail("Preflight opened a run"))
    source = "from brilliance.research_api import open_research_run\nrun = open_research_run()\n"
    issues = lint_public_api_calls(ast.parse(source + "run.raise_if_cancelled(force=True)\nrun.artifact_path()"))
    assert len(issues) == 2
    assert "unexpected keyword argument 'force'" in issues[0]
    assert "missing a required argument: 'relative'" in issues[1]
    assert lint_public_api_calls(ast.parse(source + '''
run.raise_if_cancelled()
run.artifact_path("optics.png")
run.optimization_options("unit-search")
run.progress("Computing optics", stage_id="linear", candidate_id="candidate-1")
''')) == []


def test_local_factory_and_unshadowed_callback_handle_are_checked():
    issues = lint_public_api_calls(ast.parse('''
from brilliance.research_api import open_research_run as start
run = start()
def evaluate(point):
    run.check_cancelled()
def main():
    local_run = start()
    local_run.progress("Optics", imaginary_keyword=True)
'''))
    assert len(issues) == 2
    assert any("check_cancelled" in item for item in issues)
    assert any("imaginary_keyword" in item for item in issues)


@pytest.mark.parametrize("source", [
    "run = unrelated_factory()\nrun.check_cancelled()",
    "def evaluate(run):\n    run.check_cancelled()",
    '''from brilliance.research_api import open_research_run
run = open_research_run()
run = unrelated_factory()
run.check_cancelled()
''',
    '''from brilliance.research_api import open_research_run
run = open_research_run()
def evaluate(run):
    run.check_cancelled()
''',
    '''from brilliance.research_api import open_research_run
run = open_research_run()
def evaluate():
    run = unrelated_factory()
    run.check_cancelled()
''',
    '''from brilliance.research_api import open_research_run
open_research_run = unrelated_factory
run = open_research_run()
run.check_cancelled()
''',
    '''from brilliance import research_api
def evaluate(research_api):
    run = research_api.open_research_run()
    run.check_cancelled()
''',
    '''from brilliance.research_api import open_research_run
run = open_research_run()
def replace():
    global run
    run = unrelated_factory()
replace()
run.check_cancelled()
''',
])
def test_unproven_rebound_and_shadowed_objects_are_not_guessed(source):
    assert lint_public_api_calls(ast.parse(source)) == []


def test_inline_factory_method_is_checked():
    issues = lint_public_api_calls(ast.parse('''
from brilliance.research_api import open_research_run as start
start().raise_if_cancelled(force=True)
'''))
    assert len(issues) == 1 and "unexpected keyword argument 'force'" in issues[0]


def test_execution_boundary_rejects_wrong_run_method_before_worker_start(tmp_path, monkeypatch):
    import subprocess

    from brilliance import research_api
    from brilliance.research_program import inspect_research_program

    monkeypatch.setattr(research_api, "open_research_run", lambda: pytest.fail("Preflight opened a run"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Preflight launched a process"))
    script = tmp_path / "program.py"
    script.write_text('''
from brilliance.research_api import open_research_run
run = open_research_run()
run.check_cancelled()
run.finish(summary="finished")
''', encoding="utf-8")
    result = inspect_research_program(tmp_path, "program.py")
    assert result["passed"] is False
    assert any("ResearchRun" in issue and "check_cancelled" in issue for issue in result["issues"])


@pytest.mark.parametrize("required", [True, False])
def test_runner_does_not_prescribe_population_algorithm_or_inherit_requirement(tmp_path, monkeypatch, required):
    import sys
    from types import SimpleNamespace

    from brilliance import research_program

    script = tmp_path / "local_probe.py"
    script.write_text('''
from scipy.optimize import least_squares
from brilliance.research_api import open_research_run
run = open_research_run()
if run.arguments.get("run_search", False):
    result = least_squares(lambda x: [x[0] - 1.0], [0.0])
run.finish(summary="Probe completed", status="partial", request_satisfied=False)
''', encoding="utf-8")
    context = SimpleNamespace(
        workspace=tmp_path,
        project_id="probe",
        project=SimpleNamespace(active_lattice_id=None),
        compute_job_id=None,
        active_lattice=lambda: None,
        artifact_path=lambda name: tmp_path / "artifacts" / name,
        environment=SimpleNamespace(
            python_executable=sys.executable,
            project_process_environment=lambda *a, **k: {"BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE": "1"},
        ),
    )
    captured = {}

    class LaunchReached(Exception):
        pass

    def capture_launch(*args, **kwargs):
        captured.update(kwargs["env"])
        raise LaunchReached  # No interpreter, Julia or numerical calculation starts.

    monkeypatch.setattr(research_program.subprocess, "Popen", capture_launch)
    with pytest.raises(LaunchReached):
        research_program.run_research_program(context, {
            "script": script.name,
            "arguments": {"run_search": False},
            "require_optimizer_evidence": required,
        })
    assert captured["BRILLIANCE_REQUIRE_OPTIMIZER_EVIDENCE"] == ("1" if required else "0")
