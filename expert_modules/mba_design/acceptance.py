"""MBA-specific request policy; the application only hosts registered validators.

This optional guard recognizes shape-making requests. It does not select a seed,
prescribe an optimization campaign, or change the conventional design pipeline.
"""

from __future__ import annotations

import ast
import re

HYBRID_SHAPE = "hybrid_mba_shape"

HYBRID_INSTRUCTIONS = (
    "For conventional-to-hybrid conversion, read laboratory.mba_design references/hybrid_continuation_api.md "
    "and references/outer_dba_api.md. "
    "Replace the two old outer sections with newly designed DBA-like bump sections around the "
    "middle M-4-bend core. Keep the working conventional seed-generation pipeline unchanged. "
    "Use HybridMBAOuterReplacement with explicit core boundary names, not guessed unit labels. "
    "For 7BA B1/B2 and B6/B7 are outer pairs, U002:B/U003:B/U004:B are B3/B4/B5 core. "
    "Design one core-outward two-bend line; match its ID achromat and bump with independent "
    "quadrupoles, then mirror across the complete cell. Do not start with global optimization of the old cell. "
    "Use core_prefix=replacement.right_core_half() and center_optics_bounds to solve NEW centre "
    "betax/betay/dx alongside ALL globally mirror-paired quadrupoles, INCLUDING core quadrupoles. "
    "Use explore_hybrid_mba for checkpointed local trials: stalled continuation releases declared "
    "drift allocation, then outer bend-angle split, then grouped magnet lengths. Supply physical "
    "bounds, shape thresholds and trial budgets; do not generate repeated ad-hoc matcher/debug scripts. "
    "For custom line exploration explicitly pass rematch_core_quadrupoles=True and export via "
    "replacement.core_from_half(fit['core_prefix']) into assemble(core_elements=...). "
    "All K1 are inner-owned; structural variables must not overwrite them. No within-TME-unit ties. "
    "Show compact progress/failed checks, a checkpoint result table and one candidate optics plot. "
    "Source dispersion and beta preservation is "
    "NOT required by default. Measured source optics are diagnostics/initial guesses, not targets. "
    "Only an explicit scientist preservation request uses fixed core-exit Twiss and "
    "replacement.evaluate(preserve_core_optics=True) with measure_reference(parent). "
    "No seed-tune target. Preserve exact length, angle, magnet limits and ID straight. "
    "Stage 1 uses replacement.evaluate(stage='bump_feasibility'): two extended high-dispersion "
    "DRIFT regions, absolute/core-relative dispersion, usable lengths, slopes, beta envelope, "
    "achromat, stability and positive damping. No baseline H/I5/emittance ceiling is implicit. A stationary "
    "drift plateau is valid; a peak in a quadrupole alone is not usable sextupole space. "
    "Emittance is diagnostic, not the primary first-stage objective. Save this milestone as intermediate "
    "data with request_satisfied=False for an unfinished hybrid-design request. "
    "Then separately recover core quality/emittance while retaining bumps, and only later correct "
    "chromaticity. Final hybrid claims require the default full_hybrid gate on the exact candidate "
    "with explicit core-quality thresholds, published as "
    "run.finish(acceptance={'hybrid_mba_shape': report}, candidate_lattice=candidate). "
    "A stable low-emittance lattice or a line-matched stationary minimum is not hybrid-shape evidence."
)


def is_hybrid_shape_request(request: str) -> bool:
    """Recognize explicit shape-making requests, not project names or inspections."""
    text = str(request).casefold()
    hybrid = bool(re.search(r"\bh\s*[-–]?\s*(?:\d+|m)\s*ba\b|\bhybrid(?:ize|ise|ization|isation)?\b", text))
    bumps = bool(re.search(r"(?:two|2|double|paired)\s+(?:outer\s+)?dispersion\s+bumps", text))
    action = bool(re.search(
        r"\b(?:build|create|make|form|shape|convert|transform|design|produce|generate|hybridize|hybridise)\b",
        text,
    ))
    conceptual = bool(re.match(r"\s*(?:what\b|why\b|explain\b|describe\b|how\s+(?:does|do|is|are|can|to)\b)", text))
    negated = bool(re.search(r"\b(?:do\s+not|don't|never|without)\s+(?:\w+\s+){0,2}(?:build|create|make|form|design|hybridize|hybridise)\b", text))
    chinese_action = any(word in text for word in ("设计", "构建", "构造", "改成", "形成", "生成"))
    return bool((hybrid or bumps) and (action or chinese_action) and not conceptual and not negated)


def hybrid_optimization_warnings(source: str) -> list[str]:
    """Warn about visible objective mismatches without guessing callback behavior."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # The application's generic source preflight owns syntax errors.
    emittance_names = {"emittance", "emittance_pm", "emittance_pm_rad", "emittance_m_rad",
                       "natural_emittance_pm", "epsilon_x", "eps_x", "epsx", "eps_pm"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if name not in {"particle_swarm_optimize", "optimize_lattice_parameters"}:
            continue
        for keyword in node.keywords:
            if keyword.arg != "objective_names":
                continue
            try:
                names = ast.literal_eval(keyword.value)
            except (TypeError, ValueError, SyntaxError):
                continue
            if isinstance(names, (list, tuple)) and names and all(isinstance(item, str) and item.casefold() in emittance_names for item in names):
                return [
                    "The current request is H-MBA dispersion shaping, but the saved optimizer declares only "
                    "emittance objectives. Establish outer DBA replacement and two interior bumps first; "
                    "usable drift space, stable achromat and resulting bump/core separation are first-stage constraints; "
                    "source beta/dispersion preservation is optional, not a default. "
                    "Emittance-only search is appropriate only for a separate recovery stage after a validated "
                    "shape exists, with its shape gates retained as hard callback constraints. This is an "
                    "advisory: static source inspection cannot infer the callback's physics constraints."
                ]
    return []
