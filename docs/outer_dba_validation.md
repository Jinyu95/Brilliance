# Outer DBA replacement validation — 2026-09-10

Historical experiment, superseded by module v1.11.0. Its BUMP_FEASIBLE label established
only regional peaks and unchanged magnets, not usable sextupole drifts or preservation of
the core optics. It must not be treated as current H-MBA acceptance. See
[the optical-contract update](hybrid_optical_contract_validation.md).

The conventional modified-TME/unit/assembly method is unchanged. New construction,
line matching, physical-core preservation and staged knowledge live exclusively in
`expert_modules/mba_design/` (module v1.10.0). No existing facility lattice is embedded.

## What changed

Conversion now designs one NEW core-outward two-bend section, then mirrors it around the
preserved middle M-4-bend core. The supplied six-quadrupole connectivity is an unsolved
hypothesis with explicit geometry/strength inputs, not a stored seed. Each outer section's
length and angle come from the complete-cell contract. The ID straight is protected.

The inner solver first initializes the ID achromat, then matches the bump. It uses JuTrack
one-pass Twiss instead of requiring a stable guessed periodic cell. The first mode retains
measured core-exit optics; an optional mode releases the symmetry-plane betas/dispersion
without changing core magnets. Neither mode matches tunes or minimizes emittance/core H.

Station selection matters: demanding zero dispersion slope in an arbitrary drift forced a
flat-dispersion condition. Sampling inside a focusing quadrupole with JuTrack prefix propagation
allowed a genuine maximum. The exported magnet remains whole. Dense whole-cell sampling
independently verifies rising/falling slopes and rejects stationary minima.

## Actual numerical check

Source: project-dd1b326a57ac6e, conventional ring archived under
`program-749eace7b54c024cadb3/candidate_lattice.json`.
Source SHA256: `1ab8c984c794d0777a1e7d5fb6b814adcca8ebf267f8babfc163f7d93aab22d0`.
The full U002:Dend through U004:R:Dend core is unchanged, including all quadrupoles/drifts.

Result directory: `research_runs_outer_dba_preserved_optics/` (not a research-project update).

| Quantity | Measured result |
| --- | ---: |
| Energy / periods | 3.5 GeV / 32 |
| Cell length / circumference | 25 m / 800 m |
| Outer replacement length, each | 7.7 m |
| Left / right interior peak | 30.000 / 30.000 mm |
| Core maximum absolute dispersion | 22.918 mm |
| Bump/core contrast | 1.309 |
| Maximum absolute ID dispersion | 2.58 × 10⁻¹⁵ m |
| Core maximum H | 5.24 × 10⁻⁵ m |
| Full-ring natural emittance | 3,391.93 pm rad |
| Maximum beta | 859.85 m |
| Complete-cell and explicit full-ring transverse stability | Passed |
| Geometry, field limits, reflection, core preservation | Passed |
| Two genuine outer interior maxima and achromat | Passed |
| Positive damping partitions / radiation integral consistency | Passed |

The first geometry and first strength start matched in 1,577 residual evaluations; the
scout including startup and plot export took 22.86 s on this machine. This timing is a
single bounded experiment, not a general performance/reliability guarantee. An independent
full-ring evaluation agreed with the repeated-cell emittance and JuTrack per-bend I2/I4/I5 sums.

### Radiation diagnosis

Fractions below group the 32 repeated instances of each physical bend.

| Bend | I₂ fraction | I₅ fraction |
| --- | ---: | ---: |
| B1, new terminal | 10.949% | 15.904% |
| B2, new inner DBA | 21.898% | 33.883% |
| B3, preserved U002:B | 11.435% | 0.142% |
| B4, preserved U003:B | 11.435% | 0.142% |
| B5, preserved U004:B | 11.435% | 0.142% |
| B6, new inner DBA | 21.898% | 33.883% |
| B7, new terminal | 10.949% | 15.904% |

About 99.57% of I5 comes from the new outer bends. The retained core is not the dominant
emittance problem in this candidate. Recovery should improve outer bend optics/lengths and
beta/chromatic burden while retaining the demonstrated bumps, not rerun TME design.

## Status and limitations

This is **BUMP_FEASIBLE**, not a practical low-emittance light-source design. Its large beta
and 3.39 nm emittance require substantial separate recovery. Chromatic correction, DA/MA,
errors and longitudinal/RF operation have not been validated. In particular, raw ringpara
U0/damping-time/RF fields in this no-RF prototype must not be advertised as operational results;
the radiation checks here establish integral consistency and positive partition numbers only.

The exploratory free-centre-optics branch also passed the first shape gate but was much worse
(33.83 nm and approximately 8.3 km peak beta). It is not the selected initialization. Earlier
fixed-drift-station scouts failed and were not presented as successful designs. Interrupted
multistart files containing zero bytes are invalid evidence and were left untouched.

All validation outputs remain separate from the saved project; no canonical lattice was replaced.
The final hybrid acceptance validator rejects a bump-feasibility report, even if its passed flag
is true. Low-core-quality requirements are deferred, not silently declared satisfied.

173 focused regression tests passed across construction, stage gates, expert discovery,
scientific acceptance, API preflight and native research integration. No live LLM session was
run, so following the revised route autonomously remains a separate end-to-end test.
