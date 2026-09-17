"""A useful Python failure must survive noisy native-library process shutdown."""
import sys
from types import SimpleNamespace

import pytest

from brilliance import research_program
from brilliance.research_recovery import coding_failure_summary


def traceback(terminal="KeyError: 'tune_x'"):
    return ('Traceback (most recent call last):\n'
            '  File "program.py", line 321, in <module>\n'
            '    value = calibration["tune_x"]\n'
            '            ~~~~~~~~~~~^^^^^^^^^^\n' + terminal + '\n')


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_python_exception_survives_trailing_julia_warnings(newline):
    locals_line = 'BRILLIANCE_FAILURE_LOCALS={"calibration": "<dict length=4>"}'
    raw = (traceback() + locals_line + '\n' + 'Invalid mode for EdwardsTengTwiss.\n' * 500)
    selected = research_program._research_failure_context(raw.replace('\n', newline))
    assert coding_failure_summary(selected) == "KeyError: 'tune_x'"
    assert 'program.py' in selected and 'line 321' in selected
    assert locals_line in selected
    assert 'Invalid mode' not in selected
    assert len(selected) <= 4000


def test_chained_error_selects_the_final_python_cause():
    raw = traceback("ValueError: inner callback") + '\nDuring handling of the above exception, another exception occurred:\n\n'
    raw += traceback("TypeError: outer callback") + 'native shutdown\n' * 1000
    selected = research_program._research_failure_context(raw)
    assert coding_failure_summary(selected) == 'TypeError: outer callback'
    assert 'ValueError' not in selected


def test_runner_locals_delimit_program_error_before_shutdown_tracebacks():
    raw = traceback() + 'BRILLIANCE_FAILURE_LOCALS={}\n'
    raw += traceback('RuntimeError: atexit cleanup failed')
    assert coding_failure_summary(research_program._research_failure_context(raw)) == "KeyError: 'tune_x'"


def test_long_traceback_and_message_retain_type_and_nearest_frame():
    raw = 'Traceback (most recent call last):\n' + '  File "library.py", line 10, in helper\n' * 300
    raw += '  File "program.py", line 321, in <module>\n    use_value()\n'
    raw += 'AttributeError: missing field ' + 'details ' * 2000 + '\n'
    raw += 'BRILLIANCE_FAILURE_LOCALS=' + 'x' * 5000 + '\n' + 'native warning\n' * 500
    selected = research_program._research_failure_context(raw)
    assert len(selected) <= 4000
    assert selected.startswith('Traceback (most recent call last):')
    assert 'program.py' in selected and 'line 321' in selected
    assert coding_failure_summary(selected).startswith('AttributeError: missing field')
    assert 'native warning' not in selected
    assert 'BRILLIANCE_FAILURE_LOCALS' not in selected  # Never emit a truncated JSON record.


@pytest.mark.parametrize("terminal", ["example.CustomProblem: diagnostic", "KeyboardInterrupt", "SystemExit: 3"])
def test_general_exception_names_are_not_limited_to_coding_error_whitelist(terminal):
    selected = research_program._research_failure_context(traceback(terminal) + 'shutdown warning\n' * 500)
    assert selected.endswith(terminal)
    assert coding_failure_summary(selected) is None


def test_native_only_failure_keeps_both_ends_within_budget():
    raw = 'fatal native error: signal 11\n' + 'native backtrace\n' * 1000 + 'last native diagnostic'
    selected = research_program._research_failure_context(raw)
    assert selected.startswith('fatal native error: signal 11')
    assert selected.endswith('last native diagnostic')
    assert len(selected) <= 4000
    assert coding_failure_summary(selected) is None


def test_short_native_and_empty_stderr_have_clear_fallbacks():
    assert research_program._research_failure_context(' native abort\n') == 'native abort'
    assert 'No stderr output' in research_program._research_failure_context('')


def test_failed_runner_preserves_raw_logs_and_reports_repairable_cause(tmp_path, monkeypatch):
    source = tmp_path / 'probe.py'
    source.write_text('from brilliance.research_api import open_research_run\n'
                      'run = open_research_run()\nrun.finish(summary="probe")\n', encoding='utf-8')
    raw_stderr = traceback() + 'BRILLIANCE_FAILURE_LOCALS={}\n' + 'Invalid mode for EdwardsTengTwiss.\n' * 500
    raw_stdout = 'Numerical progress was preserved.\n'
    context = SimpleNamespace(
        workspace=tmp_path, project_id='error-excerpt-test', compute_job_id=None,
        project=SimpleNamespace(active_lattice_id=None), active_lattice=lambda: None,
        artifact_path=lambda name: tmp_path / 'artifacts' / name,
        environment=SimpleNamespace(
            python_executable=sys.executable, project_process_environment=lambda *a, **k: {},
        ),
    )

    def failed_process(*args, **kwargs):
        kwargs['stderr'].write(raw_stderr.encode('utf-8'))
        kwargs['stdout'].write(raw_stdout.encode('utf-8'))
        return SimpleNamespace(pid=123, returncode=1, poll=lambda: 1)

    monkeypatch.setattr(research_program.subprocess, 'Popen', failed_process)
    with pytest.raises(RuntimeError) as captured:
        research_program.run_research_program(context, {'script': source.name})
    assert coding_failure_summary(str(captured.value)) == "KeyError: 'tune_x'"
    assert 'Exact archived output directory' in str(captured.value)
    assert 'Invalid mode' not in str(captured.value)
    output = next((tmp_path / 'artifacts' / 'programs').iterdir())
    assert (output / 'stderr.txt').read_text(encoding='utf-8') == raw_stderr
    assert (output / 'stdout.txt').read_text(encoding='utf-8') == raw_stdout
