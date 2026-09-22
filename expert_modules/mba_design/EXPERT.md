---
id: laboratory.mba_design
title: MBA physics references and measurements
summary: Concise MBA design knowledge and optional JuTrack-backed diagnostics; no prescribed design controller.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, emittance, chromatic, sextupole, matching, radiation, optimization, jutrack]
version: 3.1.0
status: draft
knowledge_status: draft
kind: method
enabled: true
---

# MBA research knowledge

Use physics to choose an experiment, then write project-local code using the
configured JuTrack runtime and general numerical tools. This module supplies
references and measurements, not a mandatory sequence, installed lattice or
automatic design certificate. Other accelerator topics belong in other modules.

Read only what the current question needs:

| Question | Reference |
| --- | --- |
| How should radiation, chromatic optics and nonlinear acceptance be designed together? | [Design decisions and measurements](references/hmba_outcome_api.md) |
| How can I initialize and match a conventional MBA? | [Optional TME-derived recipe](references/conventional_api.md) |
| What is the appropriate TME normalization? | [Analytical references](references/physics_reference_api.md) |

The complete periodic cell and its explicit ring determine the result. An
isolated TME unit, stable map, optimizer exit flag or lattice label alone does
not establish that a requested design works. Use JuTrack for simulated optics,
radiation and tracking; analytical expressions are checks and initialization aids.

Keep user requirements separate from design choices and aspirations. State
material assumptions, but do not invent universal magnet limits, tune targets,
search budgets, emittance ceilings or approval gates. The model may choose and
revise a route, geometry, solver or objective using measured evidence.

Co-design radiation, sextupole sites and nonlinear acceptance; retain competing
matched candidates. There is no preferred bend count: honor requested topology,
otherwise compare alternatives. Papers supply hypotheses, not facility templates
or tune targets. Consult the design decisions reference for trade-offs; the TME
recipe is optional.

Use exact current-project candidates for follow-ups and keep trial branches
distinct from the working lattice. Save numerical results before optional
reporting. Present a short physical interpretation, the relevant measurements
and a combined one-cell lattice/optics plot when useful. Report failures as
observations, not proofs of physical impossibility.
