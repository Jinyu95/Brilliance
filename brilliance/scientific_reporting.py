"""Evidence-driven presentation of calculation outcomes, independent of topic."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_scientific_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    """Prevent failed checks from being presented as an affirmative conclusion.

    This does not judge prose or search for success words. Only explicit current
    result flags, declared final checks and assessed acceptance determine the
    outcome. Pending whole-request acceptance does not invalidate a useful
    intermediate result. Original author prose survives as diagnostic data.
    """
    result = dict(result)
    data = dict(result.get("data") or {})
    stage = dict(result.get("scientific_stage") or {})
    acceptance = result.get("scientific_acceptance") or {}
    match = data.get("match")
    failed = []
    for source, checks in (("gates", data.get("gates")), ("checks", data.get("checks")),
                           ("acceptance", acceptance.get("checks"))):
        if not isinstance(checks, Mapping):
            continue
        for name, check in checks.items():
            report = check if isinstance(check, Mapping) else {"passed": check}
            if (report.get("required") is not False and report.get("status") != "pending"
                    and (report.get("passed") is False or report.get("status") == "failed")):
                failed.append({"check": str(name), "source": source,
                               "issues": report.get("issues", [])})
    failed_boundary = (isinstance(match, Mapping) and match.get("scope") == "open_line_boundary_match"
                       and match.get("match_success") is False)
    if failed_boundary:
        boundary_checks = match.get("checks")
        names = ([str(name) for name, passed in boundary_checks.items() if passed is False]
                 if isinstance(boundary_checks, Mapping) else [])
        failed.extend({"check": name, "source": "boundary_match", "issues": []}
                      for name in names or ["boundary_targets"])
    failed_calculation = (result.get("scientific_passed") is False
                          or any(item["source"] not in {"acceptance", "boundary_match"} for item in failed))
    failed_acceptance = acceptance.get("status") == "failed" or any(
        item["source"] == "acceptance" for item in failed)
    infeasible_attempt = result.get("status") == "infeasible"
    failed_outcome = failed_calculation or failed_acceptance or failed_boundary or infeasible_attempt
    if failed_outcome:
        data.setdefault("submitted_outcome_presentation", {
            "summary": result.get("summary"), "scientific_stage": dict(stage),
            "scientific_passed": result.get("scientific_passed"),
            "request_satisfied": result.get("request_satisfied"),
        })
        heading = ("Candidate failed its numerical checks" if failed_calculation
                   else "No feasible candidate in this attempt" if infeasible_attempt
                   else "Boundary matching did not meet its tolerances" if failed_boundary
                   else "Calculation completed; candidate acceptance checks failed")
        names = list(dict.fromkeys(item["check"] for item in failed))
        summary = heading + "."
        if names:
            summary += " Failed checks: " + "; ".join(names) + "."
        summary += " Measured results and diagnostics are preserved; this attempt does not complete the request."
        result.update(summary=summary, request_satisfied=False, use_as_working_lattice=False)
        if isinstance(result.get("candidate_handoff"), Mapping):
            result["candidate_handoff"] = {**result["candidate_handoff"], "disposition":
                                           "archived_branch" if result.get("candidate_lattice") else "no_candidate"}
        if failed_calculation:
            result["scientific_passed"] = False
        if result.get("status") == "completed":
            result["status"] = "partial"
        stage.update(outcome="checks_failed" if not infeasible_attempt else "infeasible_attempt",
                     interpretation=summary,
                     next_decision=("Review the failed checks and preserved numerical evidence before advancing "
                                    "to a dependent stage. Correct the unmet conditions, or ask the scientist "
                                    "to approve any requirement changes. Preserve existing candidates and checkpoints."))
    else:
        heading = ("Research checkpoint — work remains" if result.get("status") == "partial"
                   else "Validated working candidate submitted" if (
                       (result.get("candidate_handoff") or {}).get("disposition") == "working_selection_requested"
                       and result.get("scientific_passed") is True and acceptance.get("passed") is True)
                   else "Candidate archived for comparison" if result.get("candidate_lattice")
                   else "Calculation result" if result.get("scientific_passed") is True
                   else "Unverified research checkpoint")
    calculation_status = ("failed" if failed_calculation else "passed" if result.get("scientific_passed") is True
                          else "not_verified")
    request_status = ("failed_for_this_candidate" if failed_outcome
                      else "passed" if (result.get("request_satisfied") is True
                                           and result.get("scientific_passed") is True
                                           and acceptance.get("passed") is True)
                      else "unverified" if result.get("request_satisfied") is True else "pending")
    rows = [{"Check": "Scientific checks for this result", "Outcome": calculation_status},
            {"Check": "Complete request acceptance", "Outcome": request_status}]
    if isinstance(match, Mapping) and match.get("scope") == "open_line_boundary_match":
        def verdict(value):
            return "passed" if value is True else "failed" if value is False else "not_evaluated"
        rows[:0] = [
            {"Check": "Numerical solver terminated", "Outcome": verdict(match.get("solver_terminated"))},
            {"Check": "Numerical solver convergence", "Outcome": verdict(match.get("solver_success"))},
            {"Check": "Boundary targets matched", "Outcome": verdict(match.get("match_success"))},
            {"Check": "Complete-cell acceptance", "Outcome":
             (match.get("complete_cell_acceptance") or {}).get("status", "not_evaluated")},
        ]
    rows.extend({"Check": item["check"], "Outcome": "failed", "Source": item["source"]} for item in failed)
    evidence_status = ("reporting_failed" if data.get("reporting_failure") or stage.get("outcome") == "reporting_pending"
        else "numerically_rejected" if failed_outcome else "accepted" if request_status == "passed"
        else "evaluated_not_accepted" if calculation_status == "passed" else "not_evaluated")
    result["outcome_presentation"] = {"heading": heading, "calculation_status": calculation_status,
                                       "evidence_status": evidence_status,
                                       "request_acceptance_status": request_status, "failed_checks": failed}
    if failed_outcome or (isinstance(match, Mapping) and match.get("scope") == "open_line_boundary_match"):
        result["tables"] = [*[table for table in result.get("tables", [])
                              if not isinstance(table, Mapping) or table.get("kind") != "runtime_scientific_outcome"],
                            {"title": "Calculation outcome", "kind": "runtime_scientific_outcome", "rows": rows}]
    result.update(data=data, scientific_stage=stage)
    return result
