"""Small result-shape hints, never raw model credentials or numerical dumps."""
import json
from types import SimpleNamespace

from brilliance.program_runner import _failure_locals, _safe_local


def test_failed_mapping_exposes_available_keys_without_values():
    result = _safe_local("limits", {"max_abs_k1_m2": 3.4261995199999995,
                                  "minimum_bend_radius_m": 8.3,
                                  "api_key": "MUST_NOT_LEAK"})
    assert result == {"type": "dict", "length": 3,
                      "keys": ["max_abs_k1_m2", "minimum_bend_radius_m"], "keys_truncated": False}
    assert _safe_local("credentials", {"model": "anything"}) == "<redacted>"


def test_shape_hints_are_bounded_and_do_not_call_custom_mapping_methods():
    value = {f"field_{i}": "large saved evidence" * 1000 for i in range(100)}
    result = _safe_local("result", value)
    assert len(result["keys"]) == 24 and result["keys_truncated"] is True
    assert "large saved evidence" not in str(result)


def test_error_expression_shape_survives_many_unrelated_locals(tmp_path):
    source = tmp_path / "experiment.py"
    source.write_text("value = limits['invented_key']\n", encoding="utf-8")
    values = {f"a_{i}": {f"field_{j}": None for j in range(30)} for i in range(50)}
    values["limits"] = {"minimum_bend_radius_m": 8.3, "max_abs_k1_m2": 3.4}
    frame = SimpleNamespace(tb_lineno=1, tb_frame=SimpleNamespace(
        f_code=SimpleNamespace(co_filename=str(source)), f_locals=values))
    result = _failure_locals(frame)
    assert "minimum_bend_radius_m" in result["limits"]["keys"]
    assert len(json.dumps(result, ensure_ascii=True)) <= 2400
