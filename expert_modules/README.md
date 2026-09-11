# Local expert modules

An expert edits one required file and, only when needed, one Python file:

```text
expert_modules/
  my_physics/
    EXPERT.md
    tools.py            # optional
    references/         # optional supporting data or citations
```

Brilliance discovers `EXPERT.md` in each immediate child directory. Project-specific modules may
live under `workspace/experts/`. Additional roots are configured in
`brilliance/config/research_runtime.toml` or `BRILLIANCE_EXPERT_PATH`.

Use `python -m brilliance.expert_cli` with `new`, `doctor`, `test`, `pack`, or `install`. The only
document an expert needs is `docs/EXPERT_AUTHORING_GUIDE.md`. Legacy `expert.toml` packages remain
readable but are not the authoring interface.

Expert modules are not limited to lattice design. Orbit, tracking, RF, impedance, beam-beam,
collective-effects, controls, diagnostics, and data-analysis expertise use the same two-file shape.
The model normally turns this knowledge into a saved, task-specific research program. Optional
tools are reserved for small deterministic operations that genuinely remain unchanged across
many questions; they must not encode a design route or optimization campaign.
