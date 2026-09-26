# Local expert modules

This `no-expert-module` ablation branch intentionally ships no optional expert
modules. `[experts].enabled = false` in `brilliance/config/research_runtime.toml`
disables repository, configured, environment and project-local module discovery.
The built-in execution/API module remains because it supplies ordinary and durable
research-program execution, not a lattice-design method.

Start a fresh project for the ablation. Existing conversations and saved expert
assets retain their original history. See [ablation notes](../docs/NO_EXPERT_ABLATION.md).
The extension instructions below describe the full application's interface;
installing a module and re-enabling discovery changes the experimental condition.

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

When selected knowledge needs supporting text in every initial and continuation prompt,
declare an ordered list in the `EXPERT.md` front matter, for example:
`context_references: [references/method.md, references/measurement_scope.md]`.
Brilliance delivers the primary text and those complete Markdown files as one coherent
context bundle, with source and knowledge-status labels. Paths must stay inside the module;
duplicate references are included once and the primary file cannot include itself.
Includes are explicit and do not recurse through other links or front matter. Ordinary
references remain available on demand. If a custom context budget is too small, the
incomplete-context notice points to `read_expert` on the primary asset; its ordered pages
contain the same entire bundle. The module must still match the request before its bundle
is supplied, and its scientific method remains guidance rather than a mandatory controller.
