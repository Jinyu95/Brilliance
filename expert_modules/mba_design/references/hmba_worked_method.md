# Worked HMBA design experiment

An optional two-bump method, not a saved lattice or guarantee. TME-derived and
exploratory superperiod routes remain alternatives.

## Give the correction region optical controls

One useful layout hypothesis is:

`straight – matching quads – outer bend – correction region – outer bend – low-H bending core – reflected outer region – straight`

Interleave quadrupoles and finite sextupoles inside each non-bending correction
region. Three independently focused stations are one hypothesis for complementary
horizontal/vertical beta functions at useful dispersion. Reserve octupole space
for nonlinear control. Counts, polarities, positions and pairings remain choices.

Separate matching space, correction gaps, core spacing and magnet lengths.
Independent bend groups may vary angles, lengths and permitted gradients/field
profiles; reverse bends need not share one angle. Release unrequired ties when
restrictive. Keep total length, angle and usable straights exact; reject impossible
allocations rather than clipping gaps into ineffective knobs.

## Match for correction leverage, not just emittance

Use JuTrack for complete periodic-cell optics/radiation: stable achromat, physical
damping and declared engineering/beta limits. Tune is initially an outcome.

At each feasible proposal, measure the chromatic response of the actual finite
sextupoles and solve both chromaticity targets. Evaluate candidate pair maps in
normalized coordinates at consistent endpoints. A useful objective vector is:

`[pair-map mismatch / M_scale, sum(|K2| L) / S_scale, sum(|natural chromaticity|) / X_scale, emittance / E_scale]`

Disclose scales and individual components. Enforce requested emittance ceilings;
otherwise retain trade-offs. Minus-identity hypotheses must match the chosen
pairing physics, not arbitrary phases; map errors/chromaticity do not prove acceptance.

Measure site dispersion/betas, full-core dispersion/H and per-bend radiation
using the reporting regions. Poor localization can justify optical redesign
despite improved merit; no universal bump-height/contrast threshold replaces this.

## Compare different optics before committing

Compare independently rematched phase neighborhoods under actual ring periodicity,
not copied facility tune grids. Temporary tune labels are not machine requirements.
Warm-start nearby matches, rematch geometry/strengths and screen several corrected
alternatives with short tracking, not linear merit alone.

For each promising optics, measure the chromatic-response nullspace: its dimension
is `number_of_families - rank` (six families at rank two leave four directions).
Explore these with independent octupoles, respecting powering constraints and
re-correcting chromaticities after finite moves. Compare several optics.

Rank origin-connected DA and origin-launched MA separately; follow the
[observation contract](hmba_outcome_api.md), not a finite-amplitude substitute.
Use `api.connected_transverse_acceptance` and `api.connected_momentum_acceptance`
on separate raw masks in search/reporting. MA needs delta=0 and intermediate offsets;
islands do not extend its interval. Keep each sign/location's deficit visible.
Screen cheaply; validate finalists with longer full-ring tracking. Follow the
observation contract for DA preservation and bounded numerical checks; single-site
survival is not ring-wide MA.

Save candidates and trade-offs; reopen optics/geometry when correction saturates.
Failed searches do not prove impossibility. Publish measurements and plots.
