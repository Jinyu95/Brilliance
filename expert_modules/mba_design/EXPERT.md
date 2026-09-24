---
id: laboratory.mba_design
title: MBA physics references and measurements
summary: Concise MBA design knowledge and optional JuTrack-backed diagnostics; no prescribed design controller.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, emittance, chromatic, sextupole, matching, radiation, optimization, jutrack, superperiod, superperiods, super-periodic, ultralow-emittance, sub-10]
version: 3.14.0
status: draft
knowledge_status: draft
kind: method
enabled: true
---

# MBA research knowledge

Use configured JuTrack and general numerical tools in project-local code,
not fixed sequences or installed lattices.

| Question | Reference |
| --- | --- |
| Conventional HMBA design? | [Established HMBA](references/hmba_outcome_api.md) |
| Sub-10-pm and superperiod design? | [Layout and optics exploration](references/very_low_emittance.md) |
| Conventional MBA initialization? | [Optional TME-derived recipe](references/conventional_api.md) |
| TME normalization? | [Analytical references](references/physics_reference_api.md) |

Use complete-cell/ring evidence. Distinguish requirements, assumptions and aspirations;
no universal limits, tunes or budgets. Co-design radiation, correction and acceptance.
There is no preferred bend count; honor explicit topology.
Ordinary HMBA requests use the established guide. Sub-10-pm goals warrant considering
both routes, not automatic topology replacement.

For sub-10-pm/complex-superperiod design, read the complete guide before constructing:
`read_expert(module_id="laboratory.mba_design", asset_id="references/very_low_emittance.md")`.

Search physical magnet counts, ordering and roles, not just geometry/strengths.
Compare dense many-weak-bend layouts with short, independently sized bends;
do not fill spare length with equal dipoles.
For the two-arc hypothesis, use ONE localized chromatic-correction region per arc
and an intermediate user straight; larger-block linear reflection, not internal arc symmetry:
two correction regions per block, not two per arc, not concatenated finished MBAs.
Vary physical focusing spacing, independent bend lengths/angles, LGB and reverse bends;
numerical slices add no focusing stations. Include internal bump quadrupoles creating
complementary SF/SD optics; show actual lengths, angles and K1. Match the complete block and evaluate
native radiation plus measured bounded chromatic correction inside optimization.
Use persistent parallel JuTrack workers.
These principles do not prove optimality or prescribe a route for all MBA tasks.
Preserve declared optical conditions when optimization objectives change.

Fresh projects generate new candidates, without another project's fitted lattices
unless explicitly requested. Explain hypotheses/knobs, then calculate in the
same request without routine approval pauses. Save before reporting; show scientific
trade-offs and plots. Failed searches are not impossibility proofs.
