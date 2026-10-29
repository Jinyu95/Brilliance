---
id: laboratory.mba_design
title: MBA physics references and measurements
summary: Concise MBA design knowledge and optional JuTrack-backed diagnostics; no prescribed design controller.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, hmba, h7ba, modified-h7ba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, emittance, chromatic, sextupole, matching, radiation, optimization, jutrack, superperiod, superperiods, super-periodic, ultralow-emittance, sub-10]
version: 3.18.0
status: draft
knowledge_status: draft
kind: method
context_references: [references/hmba_worked_method.md, references/hmba_outcome_api.md, references/injection_acceptance.md]
enabled: true
---

# MBA research knowledge

Use configured JuTrack. These are hypotheses, not fixed workflows; consult
primary literature for uncertain physics/technology.

| Question | Reference |
| --- | --- |
| HMBA | [Method and measurements](references/hmba_outcome_api.md) |
| Sub-10-pm/superperiods | [Exploration](references/very_low_emittance.md) |
| Conventional initialization | [TME-derived recipe](references/conventional_api.md) |
| TME scale | [Normalization](references/physics_reference_api.md) |

Use complete-cell/ring evidence; co-design radiation, correction and acceptance.
Honor explicit topology; no universal limits, tunes, budgets or bend count.
Ordinary HMBA requests use the established guide. Sub-10-pm goals can use either route.

For sub-10-pm/complex-superperiod design, read the complete guide before constructing:
`read_expert(module_id="laboratory.mba_design", asset_id="references/very_low_emittance.md")`.

Explore physical counts, ordering and roles, not concatenated finished MBAs.
The bundled method, observation contract and injection notes support comparisons.

Do not import another project's lattice unless requested. Explain knobs and save
trade-offs. Failed searches are not impossibility proofs.
