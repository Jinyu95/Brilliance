# Expert modules

New expert modules use one required `EXPERT.md` file and an optional decorated `tools.py`. The
complete public contract is [Brilliance expert authoring guide](EXPERT_AUTHORING_GUIDE.md).

Brilliance owns the general agent/runtime, configured JuTrack environment, lattice primitives,
optimization, plotting, and evidence transport. Expert modules own topic-specific knowledge,
numerical methods, request recognition, and scientific acceptance rules. Installing a topic must
not require edits to core prompts, central API exports, or a workflow engine.

Use the smallest contribution that fits:

| Need | Add inside the expert module |
| --- | --- |
| Practical methods and limitations | `EXPERT.md` |
| Ordinary reusable Python calculations/callbacks | `@python_tool()` functions in `tools.py` |
| Typed agent-callable JSON operation | Existing `@tool(...)` and `@validator(...)` |
| A scientific property required to satisfy a request | `@acceptance(...)` with an independent evidence validator |

Saved programs access Python tools with
`api.expert_module("laboratory.my_topic").function(...)`. The module's knowledge documents the
tools and their scientific meaning. Discovery publishes signatures, instructions, and checks;
the agent chooses its approach using the current request and evidence, not a fixed task pipeline.

Optional sibling files can be imported normally, for example `from .helpers import evaluate`.
Package loading is isolated and source-aware, including helper edits. Disabled/removed modules
do not provide capabilities; a required check cannot pass just because its implementation is
absent. Acceptance IDs and migration aliases must be unique across enabled modules.

The MBA module is one consumer of these interfaces, not a special case in the core. The authoring
guide includes an unrelated orbit-residual example showing how to add tools and acceptance without
changing Brilliance internals.

Legacy `expert.toml` modules remain readable for compatibility, but experts should not create or
edit manifests, JSON schemas, runtime adapters, registry records, or fingerprints by hand.
