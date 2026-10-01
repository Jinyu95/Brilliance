---
id: laboratory.machine_correction
title: LOCO and measured-machine linear optics correction
summary: Infer optics and device calibration from measured orbit response and RF dispersion using JuTrack; propose bounded offline corrections with identifiable uncertainties.
domains: [machine correction, linear optics, beam diagnostics, coupling]
authors: [Brilliance module draft prepared with OpenAI Codex]
tags: [LOCO, response-matrix, ORM, BPM, calibration, dispersion, quadrupole, skew, commissioning]
version: 0.1.2
status: draft
knowledge_status: draft
kind: method
context_references: [references/loco.md]
enabled: true
---

# Measured-machine correction

Identify measured optics from a nominal ring, named ORM and dispersion/RF data;
evaluate requested corrections offline. There is no accelerator-control connection.

Read [LOCO reasoning and evidence](references/loco.md) for the complete method and
[Python API](references/api.md) for independently written numerical primitives.
Do not reconstruct the old application's full workflow or import its fitted answers.

One ordinary request authorizes the offline fit and diagnostics. Choose numerical
settings, record provisional acquisition assumptions and test alternatives without
routine approval questions. Reuse validated responses and proceed to fitting;
baseline diagnostics alone do not answer a LOCO request. Cross-plane discrepancies
are limitations of the initial uncoupled benchmark, not a reason to skip it.

All transport and optics come from JuTrack. Preserve the supplied model; unsupported
maps, unknown units and unresolved device identities remain real blockers. Never
fabricate data or certify assumptions. Use finite differences, not AD. Save exact
fit state before reporting; distinguish solver termination, residual agreement,
identifiability and correction validation. Hardware settings require reviewed
calibrations and limits; numerical bounds are not hardware approval.

This module is **draft**. No code from the supplied LOCO application is imported
or copied. Agreement with its real-machine results remains to be tested; primary
sources are linked in the method reference.
