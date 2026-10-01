---
id: laboratory.quasi_integrable_optics
title: Quasi-integrable octupole optics
summary: Construct realistic octupole channels, match their linear environment, and measure tune-spread versus acceptance and invariant preservation.
domains: [nonlinear optics, ring design, Hamiltonian dynamics]
authors: [Brilliance module draft prepared with OpenAI Codex]
tags: [quasi-integrable, integrable-optics, octupole, iota, invariant, tune-spread, hamiltonian]
version: 0.1.0
status: draft
knowledge_status: draft
kind: method
context_references: [references/design.md, references/api.md]
enabled: true
---

# Quasi-integrable octupole-channel design

## Applicability

Fixed-energy rings using conventional octupoles to approximate an autonomous
normalized transverse Hamiltonian. One invariant for two transverse degrees of
freedom does not establish full integrability. Exact Danilov-Nagaitsev magnets
and electron lenses require their own validated element models and are outside
this implementation.

## Required inputs

Ring or geometry, insert length, beam species/energy, incoming phase-space domain,
magnet lengths/strength limits, apertures, momentum range, desired tune spread,
minimum connected acceptance, and error assumptions. Use physical units and record
the multipole normalization.

## Method

Read [Construction and diagnosis](references/design.md) for the common-beta,
dispersion-free insert and surrounding linear transport conditions. Read
[API and coefficient conventions](references/api.md) before setting strengths.
Keep all particle transport and high-order maps in JuTrack.

## Gates and tolerances

Check the complete-ring linear match before nonlinear performance. Evaluate
connected acceptance and tune spread on the same declared launch domain, including
mixed-plane launches. Measure a candidate invariant at the same observation plane.
Full integrability, infinite stability, or collective stabilization cannot be
inferred from finite-turn single-particle survival.

## Failure and recovery

Separate mismatched linear transport, wrong octupole normalization, finite
segmentation, parasitic nonlinearities, and off-momentum effects. Increasing
octupole strength can reduce acceptance without improving useful tune spread.

## Required evidence

Archive the complete lattice, beta/dispersion/phase profiles, octupole strengths,
launch coordinates and normalization, losses, tune footprints and estimator,
invariant statistics, integration/turn-count checks, and bounded-error comparisons.

## Sources

See [the method references](references/design.md#sources). The construction helper
and diagnostic are tested independently of JuTrack. Physics knowledge is draft;
no completed lattice demonstration or named human review is claimed.
