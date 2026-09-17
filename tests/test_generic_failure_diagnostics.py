import pytest

from brilliance.failure_diagnostics import execution_failure_summary


def test_bare_stopiteration_survives_native_warnings_and_archived_locals():
    text = '''RuntimeError: research program exited with code 1: Traceback (most recent call last):
  File "C:\\research\\program.py", line 152, in <module>
    best = next(p for p in result["front"] if p["feasible"])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
StopIteration
BRILLIANCE_FAILURE_LOCALS={"result": "<dict length=7>"}
WARNING: Julia finalizer could not acquire lock
Fatal error: shutdown warning rather than the calculation failure
Exact archived output directory: C:\\research\\artifacts\\program-123
'''
    assert execution_failure_summary(text) == "StopIteration"


@pytest.mark.parametrize("terminal", [
    "AssertionError", "KeyboardInterrupt", "SystemExit: 2",
    "solver.NoFeasibleSeeds", "solver.NoFeasibleSeeds: no accepted seeds",
    "mypackage.StrangeFailure: unavailable", "lowercase_exception: custom type",
])
def test_traceback_terminal_type_does_not_require_error_suffix(terminal):
    assert execution_failure_summary(
        f'{_traceback_prefix()}\n{terminal}\nWARNING: finalization warning'
    ) == terminal


def _traceback_prefix():
    return '''Traceback (most recent call last):
  File "program.py", line 3, in <module>
    perform_operation()
    ^^^^^^^^^^^^^^^^^^^'''


def test_last_chained_exception_is_reported():
    text = f'''{_traceback_prefix()}
KeyError: 'missing'

The above exception was the direct cause of the following exception:

{_traceback_prefix()}
project.InvalidInput: could not read inputs
'''
    assert execution_failure_summary(text) == "project.InvalidInput: could not read inputs"


def test_python_exception_precedes_unrelated_shutdown_output():
    text = f'''WARNING: precompilation warning
{_traceback_prefix()}
ValueError: no viable candidate
ERROR: native worker shutdown failed
READY
'''
    assert execution_failure_summary(text) == "ValueError: no viable candidate"


def test_indented_source_that_looks_like_exception_is_not_terminal():
    text = '''Traceback (most recent call last):
  File "program.py", line 3, in <module>
    StopIteration
  File "other.py", line 7, in operation
    return value
NameError: name 'value' is not defined
'''
    assert execution_failure_summary(text) == "NameError: name 'value' is not defined"


@pytest.mark.parametrize("text,expected", [
    ("StopIteration", "StopIteration"),
    ("AssertionError", "AssertionError"),
    ("RuntimeError: worker did not start", "RuntimeError: worker did not start"),
    ("solver.NoFeasibleSeeds", "solver.NoFeasibleSeeds"),
    ("Compiling cached modules...\nREADY\ncompleted", "Compiling cached modules..."),
    ("Preparing interpreter\nsignal (11): Segmentation fault\nshutdown finished",
     "signal (11): Segmentation fault"),
    ("Preparing interpreter\nFatal Python error: init_fs_encoding failed\nThread 123",
     "Fatal Python error: init_fs_encoding failed"),
    ("research program exited with code 3221225477", "research program exited with code 3221225477"),
    ("", "Calculation could not complete"),
    ("  \n  ", "Calculation could not complete"),
])
def test_short_errors_and_native_fallback(text, expected):
    assert execution_failure_summary(text) == expected


def test_incomplete_traceback_is_not_shown_as_a_traceback_header():
    assert execution_failure_summary(_traceback_prefix()) == (
        "Python traceback is incomplete; see the archived diagnostic"
    )


def test_incomplete_nested_traceback_preserves_exit_information():
    text = "RuntimeError: research program exited with code 1: " + _traceback_prefix()
    assert execution_failure_summary(text) == (
        "RuntimeError: research program exited with code 1 "
        "(incomplete traceback; see archived diagnostic)"
    )


def test_summary_is_bounded_and_strips_terminal_colours():
    text = f'{_traceback_prefix()}\n\x1b[31mValueError: {"x" * 1000}\x1b[0m'
    summary = execution_failure_summary(text)
    assert summary.startswith("ValueError: x")
    assert len(summary) == 800
    assert "\x1b" not in summary
