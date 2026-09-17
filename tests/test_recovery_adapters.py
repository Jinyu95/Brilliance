"""Unknown exceptions survive both transports; rejected archives are not designs."""
import json

import pytest

from brilliance.codex_runtime import _command_operational_failure, _failure_summary
from brilliance.native_research_agent import _brief_error
from brilliance.research_api import ResearchRun

TRACEBACK = '''research program exited with code 1: Traceback (most recent call last):
  File "program.py", line 207, in <module>
    centre = next(row for row in sample["twiss_profile"] if row["requested_sample"])
StopIteration
BRILLIANCE_FAILURE_LOCALS={"sample":"<dict length=8>"}
Exact archived output directory: workspace/artifacts/programs/trial
'''


def test_actual_stopiteration_is_visible_in_both_error_adapters():
    assert _failure_summary(TRACEBACK) == "StopIteration"
    summary = _brief_error(json.dumps({"error": TRACEBACK}))
    assert summary.splitlines()[0] == "StopIteration"
    assert 'BRILLIANCE_FAILURE_LOCALS={"sample":"<dict length=8>"}' in summary
    assert "Traceback" not in summary


def test_large_locals_never_displace_the_actual_exception():
    summary = _brief_error(TRACEBACK.replace('<dict length=8>', 'x' * 6500))
    assert summary == "StopIteration"


@pytest.mark.parametrize("encoded", [False, True])
def test_error_adapter_preserves_windows_filenames_and_json_locals(encoded):
    missing = r"C:\research\programs\new_run\numerical_checkpoints\next_candidate.json"
    terminal = f"FileNotFoundError: [Errno 2] No such file: {missing!r}"
    local_values = {"selected_file": missing, "label": 'A "quoted" label'}
    diagnostic = ('Traceback (most recent call last):\n  File "program.py", line 3, in <module>\n'
                  + terminal + '\nBRILLIANCE_FAILURE_LOCALS=' + json.dumps(local_values))
    summary = _brief_error(json.dumps({"error": diagnostic}) if encoded else diagnostic)
    headline, locals_line = summary.splitlines()
    assert headline == terminal
    assert json.loads(locals_line.removeprefix("BRILLIANCE_FAILURE_LOCALS=")) == local_values


@pytest.mark.parametrize("tool", ["accelerator.run_research_program", "accelerator.run_long_research_program"])
def test_command_transport_allows_diagnosis_of_unknown_saved_program_errors(tool):
    event = {"type": "commandExecution", "status": "failed", "exitCode": 1,
             "command": f"configured-python -m brilliance run-expert-tool {tool}",
             "aggregatedOutput": TRACEBACK}
    failure = _command_operational_failure(event)
    assert failure["repairable"] is True
    assert failure["fatal"] is False
    assert failure["detail"] == "StopIteration"


@pytest.mark.parametrize("feasible,front", [(0, 60), (12, 4)])
def test_progress_distinguishes_infeasible_archive_from_feasible_front(tmp_path, feasible, front):
    workspace = tmp_path / "workspace"
    output = workspace / "artifacts" / "trial"
    output.mkdir(parents=True)
    run = ResearchRun(workspace=workspace, output_directory=output,
                      arguments_file=output / "arguments.json", result_file=output / "result.json",
                      progress_file=workspace / "progress.json", cancellation_file=None,
                      working_lattice_file=None, latest_candidate_lattice_file=None, compute_job_id=None)
    # Exercise reporting only: no optimization or scientific computation is run.
    callback = run.optimization_options("stage-a")["progress_callback"]
    callback({"phase": "generation", "generation": 30, "total_generations": 30,
              "feasible_count": feasible, "front_size": front})
    report = json.loads(run.progress_file.read_text(encoding="utf-8"))
    assert report["metrics"]["feasible_count"] == feasible
    assert report["metrics"]["front_size"] == front
    if not feasible:
        assert "No feasible candidate" in report["detail"]
        assert "60 non-dominated infeasible archive entries" in report["detail"]
        assert "not usable designs" in report["detail"]
    else:
        assert "feasible Pareto front: 4" in report["detail"]
    assert not run.result_file.exists()
