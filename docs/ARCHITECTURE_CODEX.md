# Codex research architecture

Brilliance uses the official Codex SDK as a coding-agent harness. It does not implement another
LLM planner, multi-agent debate, phrase router, or accelerator-design state machine on top of it.
OpenAI, DeepSeek, Ollama, LM Studio, or a compatible Responses service can provide model inference;
the harness and scientific runtime remain the same.

## Ownership

| Concern | Owner |
|---|---|
| conversation, reasoning, code editing, shell use, context compaction, and resume | Codex |
| inference | configured model backend |
| beam-dynamics calculation | JuTrack |
| stable programmatic helpers | `brilliance.accelerator_api` |
| topic method and claim boundaries | retrieved expert knowledge |
| immutable code/arguments/results and durable execution | saved-program runner |
| canonical lattice identity and project history | Brilliance archive |
| user interaction, live activity, tables, and plots | FastAPI chat client |

## Why the old design failed

Using Codex infrastructure does not automatically create Codex-like behavior. The former Brilliance
surface exposed many large operations such as `optimize_conventional_mba`. Each operation embedded
a topology, data schema, variable ownership, objective, recovery route, and presentation policy.
The model could only choose between those operations. An unanticipated request either matched the
wrong template or required edits throughout the framework.

The rebuilt boundary gives Codex the same essential freedom it has in a software repository: read
the relevant knowledge, write a normal program, run it, inspect the result, and repair that program.
Brilliance constrains runtime and evidence, not the scientific idea.

## Minimal tool surface

The model-visible operations are intentionally small:

1. inspect archived datasets;
2. query or plot archived datasets;
3. run a short saved research program; and
4. submit a durable saved research program.

The saved program imports `brilliance.accelerator_api`. That module supplies configured JuTrack
loading, lattice construction and I/O, element editing, optics/radiation helpers, plotting, and a
general callback-based PSO/MOPSO kernel. It also provides a configured raw `jutrack()` escape hatch
for methods not yet wrapped by a stable helper.

No tool owns “MBA,” “7BA,” “tracking,” or another problem class. A model-authored program owns:

- its scientific data model;
- vector-to-system or vector-to-lattice mapping;
- matching and variable ownership;
- objectives and constraints;
- convergence logic;
- JuTrack calls;
- result tables and plots; and
- the candidate or negative evidence it reports.

## Turn lifecycle

1. The chat app appends the scientist message and starts or resumes one durable `codex_turn` job.
2. Brilliance validates the configured Python/Julia/JuTrack environment before the model runs.
3. Codex receives compact project state, the public API contract, the two evidence tools, the two
   saved-program tools, and knowledge retrieved for the latest request.
4. Codex writes a named program under the project `workspace/`; inline transient scientific code is
   discouraged because source is part of the evidence.
5. The short runner snapshots the exact code and normalized JSON arguments, invokes only the
   configured Python, and archives stdout, stderr, typed result, plots, tables, and an optional
   candidate lattice.
6. A long runner uses the same contract inside a detached managed worker. The worker inherits the
   Codex sandbox/runtime policy, publishes replaceable progress, and supports cooperative user
   cancellation.
7. Generic validation checks provenance and output structure. If an explicit scientific gate
   passed, a candidate may advance the provisional working-lattice pointer. Canonical publication
   remains a separate validated action.
8. The chat presents the scientific summary and distinct artifacts. Follow-ups resume the same
   Codex thread and project files.

## Saved-program contract

A program begins with:

```python
from brilliance.research_api import open_research_run

run = open_research_run()
arguments = run.arguments
```

It may load `run.working_lattice_file`, write only through `run.artifact_path(...)`, report generic
progress with `run.progress(...)`, and test `run.cancellation_requested()` at safe numerical
boundaries. It ends with `run.finish(...)`, including a summary, data, tables, plots, methods,
limitations, `scientific_passed`, `request_satisfied`, and optionally a saved `LatticeIR`
candidate. Intermediate calculations set `request_satisfied=False`; only the result that answers
the current scientist request sets it true.

The model must set `scientific_passed=True` explicitly before a candidate can replace the current
provisional working lattice. Output-contract validation alone is never a physics pass.

## Failure boundaries

| Failure | Response |
|---|---|
| invalid program arguments or escaped path | reject before scientific state changes |
| Python/API error | preserve source, inputs, stdout/stderr, then allow at most two bounded repairs |
| JuTrack environment failure | report the configured boundary; do not rediscover installations |
| unstable, unmatched, or infeasible physics | preserve as negative evidence and change the hypothesis when justified |
| long calculation | continue independently, show live progress, stop only on completion/declared timeout/user cancellation |
| model transport failure | preserve thread and files; do not terminate a managed numerical job |

This separation prevents a scientific optimization from becoming an LLM debugging loop and
prevents a missed physics target from being mislabeled as software failure.

## Context and network policy

Large arrays, lattice definitions, and execution logs remain in content-addressed files. The model
receives compact receipts and reads exact artifacts on demand, so conversation context does not grow
with every population member or optics profile.

Native web search, shell networking, apps/connectors, and remote MCP are disabled when
`internet_access=false`. Local stdio MCP remains independently configurable. Model API access is
not research internet access.

## Expert extension boundary

An expert normally contributes only `EXPERT.md`. Knowledge defines applicability, methods,
alternatives, equations, checks, failure interpretation, evidence, and sources. It does not define
a state machine. A Python tool is justified only when the same small deterministic transformation
is reused unchanged across many tasks and is safer than repeatedly generated code.

This makes adding orbit physics, RF, collective effects, controls, measurements, or new lattice
ideas a knowledge/API extension rather than another framework rewrite.
