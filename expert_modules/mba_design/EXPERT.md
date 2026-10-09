---
id: laboratory.mba_design
title: MBA physics references and measurements
summary: Concise MBA design knowledge and optional JuTrack-backed diagnostics; no prescribed design controller.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, hmba, h7ba, modified-h7ba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, emittance, chromatic, sextupole, matching, radiation, optimization, jutrack, superperiod, superperiods, super-periodic, ultralow-emittance, sub-10]
version: 3.17.0
status: draft
knowledge_status: draft
kind: method
context_references: [references/hmba_worked_method.md, references/hmba_outcome_api.md, references/injection_acceptance.md]
enabled: true
---

# MBA research knowledge

Use configured JuTrack in project-local code. These draft notes support hypotheses,
not fixed workflows. Cite primary literature for uncertain physics/technology.

| Question | Reference |
| --- | --- |
| HMBA | [Method and measurements](references/hmba_outcome_api.md) |
| Sub-10-pm/superperiods | [Exploration](references/very_low_emittance.md) |
| Conventional initialization | [TME-derived recipe](references/conventional_api.md) |
| TME scale | [Normalization](references/physics_reference_api.md) |

Use complete-cell/ring evidence. Distinguish requirements, assumptions and aspirations;
no universal limits, tunes or budgets. Co-design radiation, correction and acceptance.
There is no preferred bend count; honor explicit topology.
Ordinary HMBA requests use the established guide. Sub-10-pm goals warrant considering
both routes, not automatic topology replacement.

For sub-10-pm/complex-superperiod design, read the complete guide before constructing:
`read_expert(module_id="laboratory.mba_design", asset_id="references/very_low_emittance.md")`.

The exploratory route considers physical magnet counts, ordering and roles,
not just geometry/strengths; it is not concatenated finished MBAs.
The [worked HMBA method](references/hmba_worked_method.md), measurement contract
and injection notes are included below. Translate them into an actual builder,
objectives and candidate comparisons.

Fresh projects generate new candidates, without another project's fitted lattices
unless explicitly requested. Explain hypotheses/knobs, then calculate in the
same request without routine approval pauses. Save before reporting; show scientific
trade-offs and plots. Failed searches are not impossibility proofs.
