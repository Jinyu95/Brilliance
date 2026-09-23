---
id: laboratory.mba_design
title: MBA physics references and measurements
summary: Concise MBA design knowledge and optional JuTrack-backed diagnostics; no prescribed design controller.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, emittance, chromatic, sextupole, matching, radiation, optimization, jutrack, superperiod, superperiods, super-periodic, ultralow-emittance, sub-10]
version: 3.12.0
status: draft
knowledge_status: draft
kind: method
enabled: true
---

# MBA research knowledge

Use configured JuTrack and general numerical tools in project-local code,
not fixed sequences or installed lattices.

Read the relevant guide:

| Question | Reference |
| --- | --- |
| Conventional HMBA design? | [Established HMBA architecture](references/hmba_outcome_api.md) |
| Sub-10-pm and superperiod design? | [Very-low-emittance construction recipe](references/very_low_emittance.md) |
| Conventional MBA initialization? | [Optional TME-derived recipe](references/conventional_api.md) |
| TME normalization? | [Analytical references](references/physics_reference_api.md) |

Use complete-cell/ring JuTrack evidence, not isolated-unit performance or solver
termination. Distinguish requirements, assumptions and aspirations; no universal
limits, tunes, budgets or approval gates.

Co-design radiation, correction sites and nonlinear acceptance. There is no preferred bend count;
honor explicit topology. Ordinary HMBA requests use the established guide.
Sub-10-pm goals warrant considering both routes, not automatic topology replacement.

For sub-10-pm/complex-superperiod design, read the complete construction recipe
before the first candidate:
`read_expert(module_id="laboratory.mba_design", asset_id="references/very_low_emittance.md")`.
Explain the hypothesis/knobs; implement with JuTrack.
When choosing the two-arc hypothesis, start independent distributed bending/focusing sections with field-shaped transitions
around ONE localized chromatic-correction region per arc and an intermediate user straight.
Use larger-block linear reflection, not internal arc symmetry:
two correction regions per block, not two per arc, not concatenated finished MBAs.
Compare physical focusing spacing, independent bend lengths/angles, LGB and reverse bends;
numerical slices add no focusing stations. Use shared-interface matching across the complete block,
then co-optimize native radiation and measured bounded chromatic correction before tracking.
These principles do not prove optimality or prescribe a route for all MBA tasks.
Preserve declared optical conditions when optimization objectives change.

Explain geometry and budgets, then calculate in the same request without routine approval pauses.
Papers supply ideas, not templates.
Preserve exact current-project candidates and provenance. Save before reporting;
show physical interpretation and plots. Failed searches are not impossibility proofs.
Continue authorized work or identify the specific decision needed.
