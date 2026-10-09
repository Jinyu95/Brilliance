# Worked HMBA design experiment

This is an optional method for a conventional two-bump HMBA, distilled from a
productive research campaign. It is not a saved lattice, universal topology or
guarantee of acceptance. Construct fresh candidates from the machine requirements;
the conventional TME-derived route and exploratory superperiod route remain valid.

## Give the correction region optical controls

One useful layout hypothesis is:

`straight – matching quads – outer bend – correction region – outer bend – low-H bending core – reflected outer region – straight`

Inside each non-bending correction region, try several interleaved quadrupole
and finite sextupole stations, rather than putting sextupoles into a passive gap.
Three independently focused stations are one practical hypothesis: their quads
can develop complementary horizontal/vertical beta functions at useful dispersion.
Reserve octupole space when nonlinear control is requested. Counts, polarities,
positions and pairings are choices to investigate, not fixed properties of HMBA.

Parameterize matching space, correction-region gaps, core spacing and magnet
lengths separately. Independent physical bend groups may vary their angles,
lengths and, when allowed, gradients or segmented field distributions. Reverse
bend groups need not share one angle. Start from justified symmetry; release
unrequired strength ties or polarity assumptions if they restrict useful optics.
Keep the requested total length, bend angle and usable straights exact. Reject
an impossible allocation; do not clip all gaps and leave ineffective knobs.

## Match for correction leverage, not just emittance

Use native JuTrack optics and radiation for the complete periodic cell. Obtain
a feasible achromat with stable optics, physical damping and the declared
engineering/beta limits. Tune is initially an outcome, not the seed's obligation.

At each feasible proposal, measure the chromatic response of the actual finite
sextupoles and solve both chromaticity targets. Evaluate candidate pair maps in
normalized coordinates at consistent endpoints. A useful objective vector is:

`[pair-map mismatch / M_scale, sum(|K2| L) / S_scale, sum(|natural chromaticity|) / X_scale, emittance / E_scale]`

Choose and disclose scales from this study. Keep every component visible even
when a scalar optimizer combines them. Use a requested emittance ceiling as a
constraint; otherwise retain the emittance/correction trade-off. A minus-identity
cancellation hypothesis must match the chosen pairing physics, not an arbitrary
phase preference. Neither its map error nor chromaticity alone predicts acceptance.

Alongside this objective, measure correction-site dispersion/betas, full-core
dispersion/H and per-bend radiation using the same regions as the final report.
Poor localization is a reason to revise the optical hypothesis, even when the
weighted objective improves. Do not replace these observations with one universal
bump-height or contrast threshold.

## Compare different optics before committing

Build a bank of independently rematched optics across useful phase-advance
neighborhoods under the actual ring periodicity. Temporary tune constraints can
label an experiment; they are not permanent machine requirements. Warm-start
nearby matches, allow geometry and strengths to rematch, and retain several
feasible alternatives. Do not copy a facility's tune grid. Screen each corrected
alternative with short tracking instead of selecting solely by the linear merit.

For each promising optics, form the measured chromatic-response nullspace using
its numerical rank. With six independent sextupole families and rank two, four
directions remain, not one antisymmetric knob. More generally there are
`number_of_families - rank` directions. Explore the permitted directions together
with independent octupoles; respect actual powering constraints and re-correct
chromaticities after finite changes. Repeat across several optics, not only one.

Rank trials by origin-connected transverse survival and connected local momentum
intervals of both signs, with launch conditions and locations recorded. Include
intermediate momentum samples so surviving islands do not inflate acceptance.
Use `api.connected_momentum_acceptance` and `api.connected_transverse_acceptance`
on the raw survival masks in both optimization and reporting. Track the zero
launch at every momentum-test location. Keep each sign/location's target deficit
visible: good survival elsewhere must not compensate for a failing required
interval. A screen that stops at its scan boundary gives only a lower bound.
Use cheap screens to allocate effort, then longer screens on finalists. Explicit
full-ring tracking with denser locations, amplitudes and integration checks is
separate validation; short training-site survival is not a machine acceptance.

Save the selected candidate, measured trade-offs and limitations. Reopen optics
or geometry when nonlinear correction saturates; a failed bank does not prove
that the architecture cannot work. Publish measurements and plots, not only file paths.
