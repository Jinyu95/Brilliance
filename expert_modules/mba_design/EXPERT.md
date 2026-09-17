---
id: laboratory.mba_design
title: MBA design physics and numerical diagnostics
summary: Practical MBA initialization and diagnosis for model-authored JuTrack experiments; independent physical evidence checks.
domains: [mba, lattice design, electron storage rings, synchrotron light sources]
authors: [Brilliance project team (draft maintainer)]
tags: [mba, 5ba, 6ba, 7ba, 9ba, tme, modified-tme, hybrid-mba, matching, radiation, optimization, jutrack]
version: 2.3.0
status: draft
knowledge_status: draft
kind: method
enabled: true
---

# MBA design: reason, write, calculate, check

The scientist gives requirements; the agent defines candidate structures,
knobs, residuals, objectives and experiments. Use the core `LatticeExperiment`
and `CandidateLedger` for dependent calculations, exact candidate retention and
feasibility-first selection instead of rewriting these in each stage.
Brilliance supplies JuTrack, element editing, unit scans, matching, PSO/MOPSO,
plots and exact-state evidence.

| Task | Read | Purpose |
| --- | --- | --- |
| New conventional MBA | [Conventional physics](references/conventional_api.md) | Fresh TME/M-TME unit, complete-cell repetition, outer-two-unit matching |
| Hybrid MBA construction/conversion | [Physical outcome checks](references/hmba_outcome_api.md) | Independently measured shape and machine requirements, no prescribed solver |
| Analytical scales | [Physics reference](references/physics_reference_api.md) | Individual-bend TME conditions and geometry-weighted reference |

For a greenfield conventional MBA, repeat the measured unit into the complete
MBA, then rematch the first two units and their globally mirrored last two units.
Use `optimize_tme_unit_geometry` to search
unit lengths and strengths jointly; `design_tme_unit` is a fixed-geometry
diagnostic, not the normal design default. `assemble_mba_outer_units` accounts
for geometry and bend closure; `mba_outer_unit_knobs` declares ownership;
`match_mba_outer_units` measures the complete periodic cell through JuTrack.
Do not invent short terminal lines or freeze incoming unit optics and seed tunes.
A seven-bend layout plus stability search is not an equivalent low-emittance
initialization. An alternative requires an explicit scientist request/approval
or a documented, evidence-backed branch; "prototype" is not a reason to skip
the physical initialization. Do not demand exact TME equality or fixed tunes.

Test optional-element removal by native rematching; preserve requirements and
retain the parent when removal fails.

For follow-up work, continue the exact current-project candidate. Put research
code in the project; installed functions must be reusable numerical operations.

## Physics and evidence boundaries

- The complete periodic cell is the design authority. TME is an initialization
  reference; a good unit does not prove a good complete MBA.
- Separate machine requirements, proposed geometry/search allocations and soft
  aspirations. Preserve declared circumference, straight length, angle sum and
  energy-dependent magnet limits. Use exact saved numbers, never rounded tables.
  Declare magnet length ranges before searching; see conventional physics.
- Count bends and mirror partners by physical order, not inherited unit names.
  For the outer-pair hybrid hypothesis both SF_L and SD_L belong in B1–B2;
  their partners belong in B(M−1)–BM, not the suppressor regions. Specify actual
  finite intervals and test the requested shape; a lattice label is not evidence.
- Stable optics, geometry/field bounds, the declared achromat and positive damping
  govern feasibility. Emittance, chromatic burden and nonlinear acceptance are
  distinct results. Do not freeze seed tunes or core optics without a requirement.
- Use measured per-bend radiation and matching residuals to choose the next change.
  A failed local match or short search does not establish rank deficiency,
  convergence or physical impossibility.
- Publish the exact evaluated candidate, interpretable tables and one-cell combined
  lattice/optics plot. Distinguish calculation validity from satisfying the request.

Current expertise covers positive homogeneous bends, without bend gradients,
reverse bends or LGBs. Never import another project's results or facility
strengths as a fresh seed.
