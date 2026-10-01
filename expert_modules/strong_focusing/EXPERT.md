---
id: laboratory.strong_focusing
title: Strong-focusing lattice design
summary: Design constrained alternating-gradient cells and match them into rings or transport channels, balancing optics, chromatic response, aperture, and error sensitivity.
domains: [strong focusing, lattice design, periodic optics, matching]
authors: [Brilliance module draft prepared with OpenAI Codex]
tags: [strong-focusing, alternating-gradient, fodo, phase-advance, chromaticity, robustness]
version: 0.1.0
status: draft
knowledge_status: draft
kind: method
context_references: [references/design.md, references/api.md]
enabled: true
---

# Strong-focusing lattice design

## Applicability

Transverse alternating-gradient cells, periodic channels, and their integration
into rings. Start with uncoupled, fixed-energy optics and extend only to effects
supported by the configured model. This module does not cover collider final
focus, longitudinal strong focusing, or injection design.

## Required inputs

Particle and reference energy convention; cell/ring geometry; required optics,
momentum range and beam emittance; available magnet families, lengths, gradient
and aperture bounds; allowed matching regions; error assumptions; acceptance
criteria and computational budget. Distinguish normalized and geometric emittance.

## Method

Follow [Design and recovery reasoning](references/design.md): construct a bounded
linear seed, map feasible phase advance, match the complete lattice, then verify
chromatic response, nonlinear acceptance and robustness. Use [API guidance](references/api.md)
for the periodic-map diagnostic and existing JuTrack optics/tracking interfaces.

## Gates and tolerances

Require stable optics in both planes, feasible fields and lengths, complete-ring
closure/matching where applicable, and acceptance over the specified momentum
and amplitude domain. Choose numerical and physical tolerances from the design
contract. A stable isolated cell is not sufficient evidence of a usable ring.

## Failure and recovery

Separate stability-boundary proximity, matching infeasibility, off-momentum
instability, nonlinear loss, and hardware-bound violations. Change the relevant
layout or degrees of freedom before increasing optimizer effort. Re-evaluate
final candidates independently from their construction path.

## Required evidence

Save the full ordered lattice, family definitions and bounds, on/off-momentum
optics, phase convention, aperture and gradient margins, matching residuals,
tracked launch grid and losses, numerical convergence, and error-seed results.

## Sources

See [the primary references](references/design.md#sources). The Python diagnostic
has automated tests; a full JuTrack design campaign has not been run for this
package. Knowledge remains draft pending review by a named domain expert.
