# Hybrid optical-contract correction — 2026-09-10

Historical **v1.11 fixed-source-optics experiment**, not the current default contract.
The scientist subsequently clarified that source beta and dispersion may change.
Version 1.12 permits new periodic optics; preservation is explicitly opt-in. The
results below establish only the behavior of the earlier, more restrictive search.
They do not establish infeasibility of conversion with freely rematched optics.

Module v1.11.0 corrects the conventional-to-hybrid contract, without modifying the
conventional modified-TME/7BA construction pipeline or the saved research projects.

## Corrected requirements

- The source's periodic core optics are measured and compared throughout the core,
  including both interfaces. Keeping its magnets unchanged is not sufficient.
- Both outer regions require usable high-dispersion drift intervals, not just magnet
  peaks: contiguous/total installation length, absolute/core-relative Dx and Dpx.
- A flat dispersion plateau is valid. Its rising/falling shoulders may lie inside
  the bounding bends. Drift transport is checked from physical entrance samples.
- The complete-cell beta envelope, low core dispersion, achromat, stable maps,
  mirror symmetry, machine budgets and magnet limits remain independent checks.
- The outer matcher holds the measured source-core exit Twiss fixed. It supports
  six or more independent outer quads, configurable counts and independent lengths.
  Geometry and inner strength ownership remain separate. No tune target is added.
- Winning line matches are replayed. Reports expose individual residuals, response
  rank/singular values and saturated strengths. Success labels derive from checks.

All domain code lives in expert_modules/mba_design. Final shape evidence is now v2;
old v1 peak-only evidence remains historical, not sufficient for new acceptance.

## Read-only test of the latest research

Source: project-e43802994490b4, conventional ring program-c1161c00a6b82a80d409.
Previous hybrid: program-687856060287b50dc9f2.

The earlier 38-mm result is rejected for missing usable high-dispersion drift space,
insufficient contrast under the declared experiment, excessive beta and changed
periodic core optics. Its stability/achromat still pass; failed design requirements
are not mislabeled as API failures. No existing working lattice was replaced.

The source core reaches 27.896 mm dispersion. The experiment therefore declares
3x separation and an 85.084-mm minimum bump, .2-m contiguous and .5-m total usable
drift per side, |Dpx| <= .002, beta <=100 m, and reference-optics preservation.
These are experimental requirements, not universal H-MBA constants or a copied design.

## Fresh numerical search result — not a successful hybrid design

Four offline geometry scouts evaluated 40 distinct outer hypotheses with independent
JuTrack one-pass matching, including extra quadrupoles, longer magnets, grouped lengths,
and direct versus ID-prefit initialization. No line match passed the declared tolerances.
Counts were 80,878 / 187,822 / 201,067 / 115,683 residual evaluations, taking approximately
42 / 64 / 68 / 46 seconds including setup/diagnostics. Source hashes remained unchanged.
An additional bounded local polish of a promising eight-knob candidate also failed;
it reached a suppressor strength boundary rather than satisfying all six residuals.

Thus this update validates the corrected checks and removes the wrong optical escape
route. It does NOT demonstrate that the new outer matcher reliably produces a good H7BA
under this machine contract. These bounded failures are not proof of topology exhaustion.
Further constructive outer-cell/control-space work is still needed; neither reducing the
contrast silently nor releasing the core optics can be reported as solving this request.

Raw evidence is under research_runs_hybrid_optical_contract* (local validation output,
not expert seed assets). scripts/validate_hybrid_optical_contract.py accepts explicit
source/core boundary arguments and never publishes a candidate. The older validation
script is retained only for independent verification of historical numerical files.

## Regression coverage

Tests cover drift plateaus, narrow magnet spikes, steep high-D drifts, beta limits,
whole-core optical mismatch, missing source reference, rejected free-optics matching,
extra/length-adjusted quadrupoles, winner replay, and tampered acceptance evidence.
The full suite also exposes an unrelated pre-existing environment expectation:
test_research_environment expects a 600-second candidate budget while the local
configured value is 3600 seconds. This change does not modify that configuration.
