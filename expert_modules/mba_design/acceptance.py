"""MBA request scope and outcome evidence, without method-selected routes."""
from __future__ import annotations

import re

CONVENTIONAL_LINEAR = "conventional_mba_linear"
HYBRID_LINEAR = "hybrid_mba_linear"
MBA_OPTIMIZATION = "mba_emittance_optimization"

CONVENTIONAL_LINEAR_INSTRUCTIONS = (
    "For this fresh conventional-MBA prototype, implement the construction and matching in the "
    "project's program using general tools and references/conventional_api.md. No stored design "
    "or complete-design controller is available. Preserve the original machine contract. Publish "
    "conventional_mba_linear evidence from verify_conventional_candidate: a real measured "
    "modified-TME unit, its exact transfer into the assembled cell, and the exact final full-cell "
    "JuTrack result. Analytical TME scalars, a toy seven-bend ring or a passed flag alone are not "
    "that evidence. Show the unit optics, complete-cell optics, measured radiation and transfer "
    "comparison. Partial evidence is useful; do not claim completion before verification. "
    "This is a linear prototype, not proof of unrequested chromatic or nonlinear performance."
)

HYBRID_LINEAR_INSTRUCTIONS = (
    "The requested outcome is a physically verified hybrid-MBA linear candidate, not a named "
    "construction route. Use expert knowledge and general tools to write project-specific "
    "construction, matching and objective code. Read references/hmba_outcome_api.md for final "
    "measurement/evidence requirements. Preserve original limits, distinguish proposed targets, "
    "and independently verify the exact selected rows. Use HybridMeasurement's shared physical-region "
    "observables for search/checking and declare an approved radiation budget before production. "
    "The solver, phase hypothesis and sequence remain choices. Publish measured outcome evidence under "
    "hybrid_mba_linear. If chromatic correction or DA/MA was also requested, obtain its separate "
    "computed evidence before completing that compound request; this gate certifies only linear shape."
)

MBA_OPTIMIZATION_INSTRUCTIONS = (
    "Emittance optimization also needs mba_emittance_optimization evidence. Use any suitable "
    "search, including local optimization or a handwritten objective with inner matching. Preserve "
    "actual evaluations and independently measured baseline/selection as described in "
    "references/hmba_outcome_api.md. No PSO generation schema is required. A single feasibility "
    "evaluation is not a campaign; an unsuccessful campaign remains valid partial research."
)

_MBA = r"\b(?:\d+|m)\s*[-–]?\s*ba\b|\b(?:multi|\d+)\s*[-– ]?\s*bend\s+achromat\b"
_HYBRID = r"\bh\s*[-–]?\s*(?:\d+|m)\s*ba\b|\bhybrid(?:ize|ise|ization|isation)?\b"
_BUMPS = r"\b(?:two|2|double|paired)\s+(?:outer\s+)?dispersion\s+bumps\b"
_NEGATION = r"\b(?:do\s+not|don't|not|never|without|no\s+need\s+to)\s+(?:(?:a|an|the)\s+)?$"


def _positive_subject(pattern, text):
    return next((match for match in re.finditer(pattern, text)
                 if not re.search(_NEGATION, text[:match.start()])), None)


def _design_intent(text, subject, *, allow_conversion=False):
    if subject is None:
        return False
    prefix = text[:subject.start()]
    if re.match(r"\s*(?:please\s+)?(?:what|why|explain|describe|how)\b", text):
        return False
    if re.search(r"\b(?:plot|show|list|inspect|report|calculate|compute|give|tell|measure|explain|describe|learn|understand)\b", prefix):
        return False
    if re.search(r"\b(?:plots?|figures?|tables?|parameters?|chromaticit(?:y|ies)|tunes?|survey|inventory|status|instructions?|tutorial)\b", prefix):
        return False
    verbs = r"design|build|construct|create|make|generate|produce"
    if allow_conversion:
        verbs += r"|form|shape|convert|transform|hybridize|hybridise|optimize|optimise"
    actions = list(re.finditer(rf"\b(?:{verbs})\b", text))
    if any(re.search(_NEGATION, text[:match.start()]) for match in actions):
        return False
    desired = bool(re.search(r"\b(?:need|want|require|would\s+like|would\s+prefer)\b", prefix))
    if not allow_conversion and re.search(r"\b(?:existing|current|accepted|saved)\b", prefix):
        desired = False
    chinese = any(word in text for word in ("设计", "构建", "构造", "生成"))
    return bool(actions or desired or chinese)


def is_hybrid_shape_request(request: str) -> bool:
    text = str(request).casefold()
    subject = _positive_subject(_HYBRID, text) or _positive_subject(_BUMPS, text)
    return _design_intent(text, subject, allow_conversion=True)


def is_hybrid_linear_request(request: str) -> bool:
    return is_hybrid_shape_request(request)


def is_conventional_mba_design_request(request: str) -> bool:
    text = str(request).casefold()
    return (_positive_subject(_HYBRID, text) is None
            and _design_intent(text, _positive_subject(_MBA, text)))


def is_mba_emittance_optimization_request(request: str) -> bool:
    text = str(request).casefold()
    return is_hybrid_linear_request(text) and bool(re.search(
        r"\boptimi[sz](?:e|ed|ation|ing)\b.{0,70}\bemittance\b|"
        r"\bemittance\b.{0,70}\boptimi[sz](?:e|ed|ation|ing)\b|"
        r"\bminimi[sz](?:e|ed|ation|ing)\b.{0,70}\bemittance\b", text)
        or ("优化" in text and "发射度" in text))
