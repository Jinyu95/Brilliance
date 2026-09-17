# Brilliance expert authoring guide

Experts supply domain knowledge, reusable numerical tools, and optional scientific evidence checks.
The normal contribution is one readable file and does not require familiarity with Brilliance
internals. Add capabilities inside your module, not by editing the runtime or a central registry.

## The normal case: one file

```text
expert_modules/my_topic/
  EXPERT.md
```

Use front matter to make the material retrievable:

```markdown
---
id: laboratory.orbit_correction
title: Closed-orbit response and correction
summary: Practical response-matrix measurement, diagnosis, and correction methods.
domains: [orbit correction, response matrix, beam diagnostics]
authors: [Responsible expert]
tags: [orbit, svd, correctors, bpm]
version: 0.1.0
status: draft
---
```

Keep the entry point short: explain when the module applies, which method to read,
and the few scientific boundaries that govern every route. Use the following
topics as a checklist, not seven mandatory sections to fill:

- `Applicability`: systems, regimes, approximations, and exclusions.
- `Required inputs`: quantities, units, defaults that are acceptable, and choices that must come
  from the scientist.
- `Method`: equations, useful approaches, alternatives, and evidence-driven reasons to choose or
  abandon an approach.
- `Gates and tolerances`: hard validity checks, objectives, aspirations, and numerical tolerances.
- `Failure and recovery`: how to distinguish bad data, infeasible physics, numerical failure,
  runtime failure, and an exhausted hypothesis.
- `Required evidence`: code, inputs, plots, tables, residuals, provenance, and limitations needed
  to support a claim.
- `Sources`: primary sources and a clear distinction between sourced facts and expert judgment.

Describe physical reasoning, not an ordered software pipeline. For example, explain what an orbit
response singular spectrum means and when regularization is justified; do not require the agent to
call `measure -> invert -> correct -> plot` in one fixed sequence. The scientist's newest request
and computed evidence determine what happens next.

Keep one canonical reference per supported method under `references/`, containing
its practical physics, exact tool contract and evidence requirements. Link to that
reference instead of repeating the recipe in `EXPERT.md`, acceptance instructions
and several troubleshooting notes. Every file under `references/` is discoverable
by the agent: remove superseded guides rather than leave contradictory versions
there. Put developer validation/history in `docs/`, not active expert knowledge.
Do not replace essential construction details with vague principles just to shorten
the text. No extra manifest or manually maintained reference index is needed.

Run the local check:

```powershell
python -m brilliance.expert_cli doctor expert_modules\my_topic
```

Draft knowledge can guide provisional work but cannot by itself establish a validated scientific
claim. Promotion to `reviewed` should record a named expert review; use `validated` only after
independent numerical or experimental comparisons and regression cases exist.

## Research code uses the public API

The Codex agent writes task-specific programs using:

```python
from brilliance.accelerator_api import jutrack
from brilliance.research_api import open_research_run
```

The stable accelerator API includes:

- configured JuTrack loading through `jutrack()` and `build_jutrack_lattice()`;
- `LatticeIR` constructors, loading, saving, repetition, and one-period extraction;
- drift, quadrupole, sextupole, sector-bend, reverse-bend, and marker constructors;
- lattice insertion/removal and cached element-index mutation;
- JuTrack optics, radiation, chromaticity, and general mutable-lattice metrics;
- canonical combined optics/lattice and per-bend contribution plotting;
- arbitrary parameterization, response matching, feasible-basin generation; and
- callback-based PSO/MOPSO with batch evaluation and nondominated selection.

Those are primitives. The saved program defines its own variables, symmetry, objective, constraints,
matching, convergence checks, tables, plots, and claim boundary. When a JuTrack operation is not
wrapped, use the configured module returned by `jutrack()` and document the exact call. Never search
for an interpreter, Julia installation, or pyJuTrack source path.

All physical elements need unique names. Use SI units: metres, radians, GeV, `m^-2` quadrupole
strength, `m^-3` sextupole strength, and `m^-4` octupole strength. Preserve the complete ordered
lattice rather than only optimized variables or a plot.

## Optional tools: choose the appropriate interface

Add one `tools.py` alongside `EXPERT.md` when programs need a reusable expert implementation.
There are two interfaces, with different purposes:

| Interface | Use it for | Called through |
| --- | --- | --- |
| `@python_tool` | Ordinary Python functions, numerical arrays, callbacks, reusable topic-specific calculations | `api.expert_module("laboratory.my_topic").function(...)` |
| `@tool` with `@validator` | Bounded agent-callable operations with typed JSON inputs and independently checked outputs | The discovered expert-tool catalog |

Both are discovered from `tools.py`; no manifest, hand-written schema, core API edit, or adapter is
required. Neither decorator decides which research route the agent must follow.

Use `@python_tool(durable=True)` for a potentially long numerical operation.
Its calling program is automatically scheduled as a managed background job;
this does not impose PSO, a population budget, or a topic-specific core controller.

Keep builders, residual choices and experiment loops in the model-authored project.
Promote code to a tool only when its numerical operation is reusable independently
of one lattice or research route. A complete design controller is not a necessary
contribution. Remove obsolete implementations and their exports rather than keeping
hidden fallback procedures that continue to shape new work. Saved project code and
evidence remain history; removed APIs require explicit source review before replay,
not an implicit migration or a substituted calculation.

### Ordinary Python tools and scientific acceptance

For example, an orbit-correction module can provide an ordinary residual calculation and a
separate check of the evidence. This small example checks numerical residuals only; a real
experimental module must also validate calibration and measurement provenance.
For this example, set `id: laboratory.orbit_correction` in the module's `EXPERT.md` front matter;
the namespace below uses that ID, not the directory name.

```python
# expert_modules/my_topic/tools.py
import math

from brilliance.expert_sdk import acceptance, python_tool


@python_tool()
def orbit_residual(positions_m: list[float], tolerance_m: float) -> dict:
    """Summarize measured residual orbit under the declared contract tolerance."""
    if not positions_m or not all(math.isfinite(x) for x in positions_m):
        raise ValueError("Provide a nonempty list of finite measured positions")
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("The agreed RMS tolerance must be positive and finite")
    return {
        "positions_m": positions_m,
        "rms_m": math.sqrt(sum(x * x for x in positions_m) / len(positions_m)),
        "tolerance_m": tolerance_m,
    }


def requests_orbit_correction(request: str) -> bool:
    # Keep automatic claim checks narrow; this selector is not a research planner.
    return "correct orbit" in request.lower()


@acceptance(
    id="laboratory.orbit_correction.residual",
    applies_to=requests_orbit_correction,
    instructions="Archive measured residual positions in metres and the agreed RMS tolerance.",
    requires_lattice=False,
)
def validate_orbit_residual(report, lattice) -> list[str]:
    try:
        values = [float(value) for value in report["positions_m"]]
        tolerance = float(report["tolerance_m"])
        stated_rms = float(report["rms_m"])
    except (KeyError, TypeError, ValueError):
        return ["Missing or malformed residual-orbit evidence"]
    if not values or not all(math.isfinite(value) for value in values):
        return ["Residual orbit must contain finite measured positions"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        return ["The declared RMS tolerance must be positive and finite"]
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if not math.isfinite(stated_rms) or not math.isclose(stated_rms, rms, abs_tol=1e-12):
        return ["Reported RMS does not match the archived measurements"]
    return [] if rms <= tolerance else ["Residual RMS exceeds the declared tolerance"]
```

A saved research program uses the installed namespace and archives the report:

```python
import brilliance.accelerator_api as api

orbit = api.expert_module("laboratory.orbit_correction")
# These inputs come from the measurement and agreed machine contract, not a template.
report = orbit.orbit_residual(measured_positions_m, agreed_tolerance_m)
run.finish(
    summary="Measured the corrected residual orbit.",
    data={"residual_orbit": report},
    acceptance={"laboratory.orbit_correction.residual": report},
    scientific_passed=True,
    request_satisfied=True,
)
```

Here `run` is the program's existing `ResearchRun`. Its result boundary independently calls the
registered validator before accepting the claim. A report's own `passed=True` does not establish
acceptance. A missing required report, an unknown/disabled validator, or a failed check cannot
satisfy the request. Valid intermediate calculations can still be archived without promoting an
unsupported scientific claim.

The optional `@acceptance` arguments have these responsibilities:

- `id`: a stable, unique report key; a module-qualified name avoids collisions.
- `applies_to(request)`: identifies requests that require this scientific evidence, not a fixed
  sequence of tool calls. Keep read-only and conceptual questions out of a construction gate.
- Check measured outcomes independently of the solver. Do not require one class name,
  optimizer's generation schema, or unrequested later stage as proof of a different claim.
- `instructions`: explains the evidence and constraints to the agent when this check applies.
- `preflight(source)`: optionally returns a list of advisory strings about a saved program. This
  is guidance, not permission to replace actual evidence validation or silently rewrite code.
- `requires_lattice`: defaults to `True`; then the validator receives the exact candidate lattice
  and must check that the report belongs to it. Set `False` for topics without a candidate lattice.

The decorated validator always accepts `(report, lattice)` and returns a list of failure strings;
an empty list means acceptance. It should independently recompute checks from archived evidence
and run quickly, without starting simulation jobs, changing files, or making network calls.

### Typed agent-callable tools

Use `@tool` when all of these are true:

1. the operation is small and has a stable meaning;
2. research programs in the relevant topic will reuse it unchanged;
3. its input and output can be expressed with a closed typed schema; and
4. an independent validator can check its contract.

Good examples are a laboratory database lookup, a calibrated device conversion, or a reviewed
file-format parser. A complete MBA optimizer, tracking study, matching campaign, or publication
workflow is not a good registered tool: its variables and physics must remain visible in the saved
research program.

If a true reusable primitive is needed, use the optional SDK decorators:

```python
from typing import Annotated
from brilliance.expert_sdk import Field, tool, validator


@tool(authority="deterministic", output="Magnetic rigidity in T m")
def magnetic_rigidity(
    energy_gev: Annotated[float, Field(unit="GeV", minimum=0.001)]
) -> dict:
    return {"brho_tm": energy_gev / 0.299792458}


@validator(for_tool="magnetic_rigidity")
def validate_magnetic_rigidity(arguments, result):
    expected = arguments["energy_gev"] / 0.299792458
    passed = abs(result["brho_tm"] - expected) < 1e-12
    return {
        "passed": passed,
        "validator": "magnetic-rigidity-v1",
        "issues": [] if passed else ["rigidity identity failed"],
    }
```

Supported annotations include `float`, `int`, `str`, `bool`, `Literal`, `Optional`, lists,
dictionaries, and nested `TypedDict`. A context-aware tool may accept `context` first and use
`context.active_lattice()` or `context.artifact_path(...)`. Do not perform calculations, network
access, or file mutation at import time.

## Optional files

Only add what the knowledge genuinely needs:

```text
references/       API recipes, small licensed source data, or claim matrices
tools.py          optional decorated Python tools, JSON tools, and acceptance checks
helpers.py        optional local implementation; import with `from .helpers import ...`
examples/*.json  contract examples for those tools
```

Files in `references/` and `assets/` are automatically indexed for read-only access. Link them
from `EXPERT.md`; no second manifest edit is needed. The agent can call
`read_expert(module_id="laboratory.my_topic", asset_id="references/my_api.md")` and follow
`next_offset` through long documents. The reference inherits the module's review status, not
an independent validation claim. Reference changes update the registry fingerprint.
`api_help` also includes matching module-owned Markdown API recipes when an installed expert
Python function is requested. Put inputs, output keys, numerical rejection handling and one
minimal callable example there, so agents need not run calculations merely to discover schemas.

Do not add design-basis JSON, campaign state, optimizer schemas, adapters, or route selectors just
to make an expert module executable. If a reviewed seed lattice is scientifically useful, store it
as a clearly sourced reference artifact and describe its transfer limits in `EXPERT.md`; it remains
a hypothesis for a new machine, not a mandatory template.

Module Python files load as an isolated package. Relative imports work, and two experts can both
have a `helpers.py` without colliding. Discovery includes helper source changes in the package
identity and registry fingerprint, so subsequent discovery uses edited code; an already running
calculation retains the version it started with. Start a new program/turn to use the new version.
Do not mutate packages underneath running calculations as a way to repair their state.

Set `enabled: false` in `EXPERT.md` to disable a module. Its Python source is not imported during
discovery and its tools/checks are unavailable. Removing a module has the same capability effect;
existing archived evidence is retained, but replaying its scientific validation requires the
module. Duplicate acceptance IDs and legacy aliases are configuration errors, not priority rules.

New knowledge and code use the module namespace. SDK aliases, when deliberately
declared by a module author, resolve only an existing exported primitive; they do
not restore removed source or promise compatibility for old design controllers.
The removed MBA controllers have no callable compatibility layer. Brilliance never
rewrites archived programs automatically when an expert API changes.

## Packaging and review

```powershell
python -m brilliance.expert_cli new expert_modules\my_topic --title "Topic" --author "Name"
python -m brilliance.expert_cli doctor expert_modules\my_topic
python -m brilliance.expert_cli test expert_modules\my_topic
python -m brilliance.expert_cli pack expert_modules\my_topic
python -m brilliance.expert_cli install my_topic.zip --root expert_modules
```

The standalone test validates JSON tool schemas/examples when those tools exist. Direct Python
tools and acceptance checks need ordinary unit tests and numerical regression cases in the module;
decorating a function is not a scientific validation. Knowledge-only modules need no fake example
or empty `tools.py`. Search results, polished prose, and plausible AI output are not scientific review.
