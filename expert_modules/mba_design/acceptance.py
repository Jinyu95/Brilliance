"""MBA-specific request policy; the application only hosts registered validators.

This optional guard recognizes shape-making requests. It does not select a seed,
prescribe an optimization campaign, or change the conventional design pipeline.
"""

from __future__ import annotations

import ast
import re

HYBRID_SHAPE = "hybrid_mba_shape"

HYBRID_INSTRUCTIONS = (
    "For conversion of an accepted periodic MBA, read laboratory.mba_design "
    "references/periodic_hybrid_api.md before choosing a numerical route. "
    "The new match_periodic_hybrid method is an EXPERIMENTAL radiation-aware, fixed-topology "
    "continuation of the exact parent. It is not yet a numerically demonstrated H7BA design. "
    "The outer-replacement route in EXPERT.md H3 and references/outer_dba_api.md is an "
    "alternative reconstruction hypothesis, not a mandatory conversion or proven remedy. "
    "Do not change the working TME/conventional seed-generation pipeline or restart seed design. "
    "First reproduce the exact accepted parent, including independent periodic optics and radiation. "
    "Reserve finite SF_L/SD_L/SF_R/SD_R drift slots by physical names and global symmetry. "
    "Use complete periodic-cell measurements, not independent guessed input Twiss. "
    "For the periodic pilot pass end_fraction=0 first, then continue declared slot dispersion "
    "and core Dx/H targets with ALL global mirror-paired KQUAD strengths available, including core. "
    "No parent-tune equality, frozen-core-optics requirement, duplicate matcher ownership, or "
    "universal dipole-to-dipole -I condition is imposed. "
    "Supply scientist-approved absolute emittance, energy-spread, beta and engineering limits; "
    "do not invent them or compound a relative degradation allowance at each continuation step. "
    "Minimizing emittance WITH hard dispersion/slot/core constraints is appropriate during shaping; "
    "minimizing emittance alone without those constraints does not request a hybrid shape. "
    "On a failed step restore the last accepted candidate and reduce the target step. "
    "Never carry a shape-only multi-nanometre solution forward as an acceptable low-emittance seed. "
    "A stationary field-free drift plateau is valid; a narrow magnet peak is not installation space. "
    "Preserve exact cell length, bend angle and ID length unless explicitly authorized to change them. "
    "A stalled bounded fixed-topology pilot is not proof of physical impossibility. Any later "
    "geometry release or outer replacement must be a separately declared hypothesis with the same caps. "
    "Report progress, named failed constraints, exact candidate and immutable parent identity. "
    "Keep request_satisfied=False for an unfinished hybrid-design request. "
    "Final hybrid claims still require the unchanged full_hybrid gate on the exact candidate, "
    "sampling-convergence and full-ring checks, followed by chromatic-strength/phase and nonlinear "
    "validation as required by the scientist. Publish shape evidence as "
    "run.finish(acceptance={'hybrid_mba_shape': report}, candidate_lattice=candidate). "
    "Passing a software test or an intermediate continuation endpoint is not successful H7BA physics."
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
                    "The H-MBA request needs explicit slot/dispersion/core and machine constraints. "
                    "An emittance objective is appropriate when those constraints are active throughout "
                    "the solve; unconstrained emittance minimization does not request hybridization. "
                    "Inspect the callback contract and exact candidate evidence. This is advisory: "
                    "static objective-name inspection cannot establish whether constraints are present."
                ]
    return []
