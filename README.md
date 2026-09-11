# Brilliance

Brilliance is a chat application for autonomous accelerator-physics research. Its coding-agent
runtime interprets the scientist's request, retrieves locally installed expert knowledge, writes an
inspectable research program, and uses JuTrack for numerical beam-dynamics claims. Results return
to the chat as explanations, real tables, plots, and reproducible files.

Brilliance is not a fixed lattice-design workflow and is not tied to one model. The model decides
what program is appropriate; Brilliance owns the configured runtime, project history, durable
execution, and evidence boundary.

Development status: experimental. Unattended completion of the default MBA design task is not
yet release-qualified; see the [first-use validation report](docs/first_use_validation.md).

## Requirements

- Git
- Python 3.10 or newer
- Julia 1.10 or newer, or permission for JuliaCall to install it during setup
- one configured inference backend: OpenAI Codex, DeepSeek, Ollama, LM Studio, or another
  OpenAI-compatible chat-completions service

Windows is the primary tested platform. Linux and macOS are also supported by the setup script.

## Install

Clone the JuTrack submodule and run the managed setup:

```powershell
git submodule update --init --recursive
python scripts/setup_env.py
```

The script creates `.venv`, installs Brilliance and pyJuTrack, provisions JuliaCall, pre-warms the
Julia/Python bridge, and records the exact tested runtime in
[`brilliance/config/research_runtime.toml`](brilliance/config/research_runtime.toml). Do not start
the application until it prints `JuTrack runtime ready`.

To verify the exact application boundary again without starting the chat server:

```powershell
.\.venv\Scripts\brilliance.exe --check
```

This constructs canonical elements with Brilliance, calls JuTrack `findm66` and
`findelem(name=...)`, and reports the configured Python/Julia/JuTrack paths. The normal launcher
runs the same versioned gate when its cached readiness record is missing, stale, or incompatible.

To validate the complete optimization boundary, including two isolated Julia workers,
multi-objective parent/offspring selection, and an inner two-coordinate dispersion match:

```powershell
.\.venv\Scripts\brilliance.exe --check-optimization
```

This explicit check takes longer because each worker starts its own Julia runtime. Production
worker count and threads per worker are configured under `[compute]` in
`brilliance/config/research_runtime.toml`; the research model does not select them.

On Linux or macOS, use `python3 scripts/setup_env.py`.

The configured interpreter is a hard invariant. The research model never chooses Python, Julia,
the Julia depot, or a pyJuTrack path. Every model turn, saved program, and managed worker uses the
same configured environment with UTF-8 input/output. The setup also validates NumPy, SciPy,
Matplotlib, pandas, SymPy, h5py, and joblib so research programs never install dependencies or
implement emergency numerical substitutes during a run.

## Configure a model

Copy or edit `.env` and select one provider. Brilliance does not silently fall back to another
backend. The launcher loads the repository `.env` before activating the configured interpreter,
regardless of the terminal's current directory. Set `BRILLIANCE_ENV_FILE` only when using a
different configuration file.

DeepSeek example:

```dotenv
BRILLIANCE_CODEX_PROVIDER=deepseek
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=your-api-key
BRILLIANCE_CODEX_CONTEXT_WINDOW=128000
```

Local Ollama example:

```dotenv
BRILLIANCE_CODEX_PROVIDER=ollama
BRILLIANCE_CODEX_MODEL=deepseek-r1:14b
BRILLIANCE_CODEX_CONTEXT_WINDOW=32768
```

Local LM Studio example:

```dotenv
BRILLIANCE_CODEX_PROVIDER=lmstudio
BRILLIANCE_CODEX_MODEL=your-loaded-model-id
BRILLIANCE_CODEX_CONTEXT_WINDOW=32768
```

OpenAI example:

```powershell
codex login
```

```dotenv
BRILLIANCE_CODEX_PROVIDER=openai
```

See [`.env.example`](.env.example) for transport timeouts, context limits, and custom Responses API
settings.

## Start

```powershell
& '.\.venv\Scripts\brilliance.exe'
```

The browser opens at <http://127.0.0.1:8501>. Useful options are:

```powershell
brilliance --no-browser
brilliance --host 0.0.0.0 --port 8501 --no-browser
brilliance --project-root D:\accelerator-research\projects
```

## Use the research chat

The interface is a persistent coding-agent workbench, not a sequence of research forms.
Restart Brilliance after updating and reload the browser; existing projects are retained.

- **Chat while it works.** Send corrections and questions in the same composer. Native
  provider sessions incorporate queued messages at model/tool boundaries, including messages
  received during generation. Recent public conversation is included in later turns.
- **Inspect the work.** Open **Workspace → Programs** to read the actual Python source.
  Edit it and **Save new draft**, then ask the agent to review or run the named file. Saving
  does not execute code or overwrite a program already used for evidence.
- **See activity and results.** Public model responses stream into chat; tables, code blocks,
  and figures remain inline. **Workspace → Tool activity** shows recent calls and exact errors.
  Reconnecting restores the session; status updates do not recreate unchanged plot elements.
- **Pause separately from compute.** **Pause agent** is available for native provider sessions.
  It finishes the current operation without executing the next proposed action. A running
  JuTrack call may need to return first. Durable numerical jobs continue and have their own
  explicit stop control. Send another message after the agent has paused to resume.
- **Recover a model-response pause.** **Retry from checkpoint** resumes the latest unfinished
  request using saved evidence; it does not start the original design over. The app first tries
  two bounded recovery retries, validates the entire tool batch before execution, and shows
  the actual failure category. **Change model** explains the local `.env` settings and restart
  needed; it never switches providers automatically. **Stop here** leaves research paused
  without cancelling independent calculations. Repeated result cards are collapsed while
  retaining their calculation receipts.

Streaming is enabled for DeepSeek and compatible/local chat endpoints. Set
`BRILLIANCE_NATIVE_AGENT_STREAM=false` in `.env` if your endpoint does not support streaming;
tool activity and saved numerical milestones still update. The existing OpenAI Codex adapter
remains supported; its runtime capabilities differ from the native-provider adapter.

For native-provider responses, configure the output allowance in `.env` to suit the
selected model and endpoint:

```dotenv
BRILLIANCE_NATIVE_AGENT_MAX_TOKENS=16384
BRILLIANCE_NATIVE_AGENT_MAX_TOKENS_CEILING=65536
BRILLIANCE_NATIVE_AGENT_STREAM_USAGE=false
```

On an output-token-limit failure, retries double the allowance up to the ceiling
(by default 16,384 → 32,768 → 65,536) and ask for a smaller incremental program edit.
The increased allowance is retained for the current agent session. Set both values
within your provider's supported output/context limits; a lower ceiling caps the
initial allowance too. Equal values disable growth. Larger allowances can increase
inference time and cost. They do **not** change numerical-job time limits.

If the endpoint supports `stream_options.include_usage`, set
`BRILLIANCE_NATIVE_AGENT_STREAM_USAGE=true` to request streaming token counts.
Stop reason, requested budget, and any returned token counts are saved in the project's
`codex/protocol_diagnostics/`; private reasoning is never saved. Missing usage is unknown,
not zero. Interrupted streams and malformed tool arguments have separate recovery messages.
Restart Brilliance after changing `.env`, then use **Retry from checkpoint** on the existing
project. Partial tool calls are discarded, never executed or appended to saved code.

Ask for the scientific outcome, not an internal tool or workflow. Examples:

```text
Design a 3 GeV 7BA electron storage ring below 1000 m. Treat 100 pm rad as an
aspiration, show the one-cell optics and per-bend radiation contributions, and
preserve the executable lattice.
```

```text
Track particles launched at x = 0.5, 1.0, and 2.0 mm for 2048 turns in the
current ring and plot x-px phase space.
```

```text
Fit this measured orbit-response matrix, report the singular spectrum and
corrector solution, and compare the residual with the previous measurement.
```

The same project retains its current subject, code, evidence, and constraints. Follow-up messages
may inspect, modify, compare, plot, or run a new calculation without replaying the original task.

For a short calculation, the selected model uses a compact write-run-inspect loop to create a named
Python program under `workspace/`; Brilliance runs it through the general saved-program boundary
and returns either compact JuTrack evidence or the exact execution error to the same model turn.
DeepSeek, Ollama, LM Studio, and compatible services use their native function-calling API. OpenAI
uses the Codex runtime. Both paths produce the same inspectable program, immutable execution
record, progress events, and typed result.
For an hours- or days-long calculation, Brilliance submits the program to a durable worker. The
worker publishes generic progress metrics and continues after the model turn or browser session
ends. It stops only on completion, its declared program timeout, or an explicit user cancellation.

Reasoning updates and distinct numerical milestones are retained as normal chat messages while a
turn is running. A saved program can attach intermediate data tables and plots to each milestone;
repeated counters update their current stage instead of flooding the conversation. If neither the
model nor the program reports a new milestone, Brilliance posts one short status after the interval
set by `BRILLIANCE_CHAT_UPDATE_INTERVAL_S` (60 seconds by default). It does not repeat the same
error/status every minute; a material update starts a new quiet interval.

Program source is written atomically through the agent's confined workspace operation, never placed
on a PowerShell command line. The equivalent Codex path uses the bundled local stdio MCP writer.
Both work with internet access disabled, run zero-cost source preflight immediately, and prevent
long payloads from hitting Windows process-command limits. Large research routes are executed as
several bounded, reproducible stages instead of one monolithic script.

Population optimizations use process-isolated JuTrack lattices. Completed candidates are cached,
each completed generation is checkpointed, and a compatible rerun resumes automatically when the
program uses `run.optimization_options(...)`. This prevents a cancellation or worker failure from
discarding completed accelerator calculations.

If a background program encounters a repairable Python coding error, Brilliance returns the
saved source and diagnostic to the agent automatically. The chat shows a short repair update;
the full traceback remains in `workspace/runtime_failures/`. Repairs preserve existing numerical
results and cannot change the configured interpreter or physical requirements. The default limit
is two attempts per user request (`BRILLIANCE_AUTO_REPAIR_ATTEMPTS=2`; use `0` to disable).
Unchanged failing source is not repeatedly retried. Environment failures, physics infeasibility,
user cancellation, and superseding requests do not trigger automatic code repair.

Completed background programs also return their saved evidence to the agent for interpretation
and, if needed, the next calculation within your request. This is bounded by
`BRILLIANCE_RESULT_REVIEW_LIMIT` (default 8; `0` disables it). Reaching this limit does not cancel
running calculations. A newer user message or pause takes precedence over automatic handoffs.

An unsuccessful bounded matching or feasible-population attempt is recorded as a negative
scientific result, with its diagnostics available for the agent's next decision. It is not
presented as an application crash or a successful design. API and environment errors remain
distinct operational failures.

## How it works

The scientific runtime is independent of the interface. The active architecture has six layers:

| Layer | Responsibility |
|---|---|
| research session + web client | bounded conversation context, live steering, cooperative pause, incremental events, program inspection |
| selected model + authoring adapter | understand the request, retrieve knowledge, write and repair task-specific code |
| `brilliance.accelerator_api` | stable JuTrack loading, lattice I/O/editing, evaluation, plotting, and general PSO primitives |
| installed expert modules | topic-specific knowledge, Python tools, request guidance, and scientific acceptance validators |
| saved-program runner | fixed interpreter, immutable code and arguments, progress, cancellation, typed results, and artifacts |
| project archive and chat | persistent thread, history, lattice revisions, evidence, tables, plots, and managed jobs |

There is no built-in `design 7BA`, `optimize conventional MBA`, or `plot tracking` state machine.
The program defines its own data structure, element knobs, objective function, matching logic, and
evidence based on the request and retrieved knowledge. A 6BA, 9BA, orbit study, RF problem, error
ensemble, or unforeseen analysis therefore uses the same substrate.

JuTrack remains the numerical authority for beam dynamics. Generic validation checks the saved
program identity and result/artifact structure; scientific authority comes from the documented
method, JuTrack outputs, and explicit checks recorded by that program. Missing a requested target
is valid negative evidence, not a software crash.

The framework-level execution and presentation invariants are specified in
[`docs/basic_application_contract.md`](docs/basic_application_contract.md). They are intentionally
independent of MBA design or any other accelerator-physics workflow.

## Add expert knowledge

Most expertise is one Markdown file:

```text
expert_modules/my_topic/EXPERT.md
```

It should state applicability, physical assumptions, methods and alternatives, gates, failure
interpretation, required evidence, and sources. The model retrieves relevant sections and writes
the task-specific program. Add a registered Python tool only for a small, stable operation that is
reused unchanged across many problems—not for a scientific workflow or optimizer campaign.

Optional `tools.py` functions and acceptance validators are discovered from the same module
folder. No core edits are required to add a topic. Research programs call its Python tools via
`api.expert_module(module_id)`; the core runs its registered evidence checks without knowing
the topic's physics rules.

See the [expert authoring guide](docs/EXPERT_AUTHORING_GUIDE.md).

The MBA note includes a constructive modified-TME unit method, not a stored lattice.
See [method validation and current limitations](docs/modified_tme_validation.md) for
reproducible JuTrack checks and the distinction between a unit seed and a complete MBA design.

```powershell
& '.\.venv\Scripts\python.exe' -m brilliance.expert_cli doctor expert_modules\mba_design
```

## Projects and network policy

Projects are preserved under `research_projects/project-.../`. Important folders are:

```text
conversation/       append-only user and assistant messages
workspace/          generated source programs and current working files
workspace/artifacts immutable run snapshots, arguments, results, tables, and plots
state/lattices/      canonical lattice definitions and reproduction files
jobs/                model turns and durable numerical jobs
revisions/           immutable project revisions
manifests/           archive integrity records
```

Internet research is disabled by default in `research_runtime.toml`:

```toml
[execution]
internet_access = false

[mcp]
local_stdio_enabled = true
remote_http_enabled = false
```

This does not disable the selected model API. It blocks research web search, browser/network tools,
and remote MCP while retaining configured local stdio MCP services. Enable internet access only
when online research is intended.

The Python program runner also rejects external DNS/connect/send operations in offline mode,
while allowing loopback worker connections. This is a guard against accidental online research,
not an operating-system sandbox against hostile programs or native subprocesses. The browser
workbench loads its assets locally and needs no CDN or external search service.

## Troubleshooting

Check the exact configured accelerator runtime without changing it:

```powershell
& '.\.venv\Scripts\python.exe' scripts\setup_env.py --check-only
```

If the known Windows PythonCall long-path/cache problem is detected, the setup script backs up only
the affected compiled cache and rebuilds it. It never deletes the Julia depot.

If a model backend fails, verify the provider/model and credential or local server in `.env`. A
model transport timeout does not terminate an already submitted numerical worker.
For rejected model output, bounded diagnostics are saved under the project's
`codex/protocol_diagnostics/`, outside the downloadable workspace. They contain rejected public
output/tool fragments, not prompts or the private reasoning channel; configured credentials are
redacted. Do not share diagnostic files without reviewing their research content.

## Development

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m ruff check brilliance tests
```

See [the architecture note](docs/ARCHITECTURE_CODEX.md) for the execution and evidence contracts.

Unit tests do not establish autonomous design readiness. Test the actual welcome-screen task
in a separate project, with your configured model and research internet access disabled:

```powershell
& '.\.venv\Scripts\python.exe' scripts/validate_default_task.py start --root research_runs_default_validation
# Use the project path printed above once its background jobs finish:
& '.\.venv\Scripts\python.exe' scripts/validate_default_task.py audit '<project path>' --verify-jutrack
```

The audit exits nonzero for incomplete work, missing chat artifacts, an isolated unit passed off
as an MBA, or a failed independent JuTrack stability/radiation/achromaticity check. It never stops
a running calculation. A pass qualifies only the first linear candidate—not low-emittance
optimality, nonlinear acceptance, or a production accelerator design. Human review of the combined
one-cell optics plot and physical assumptions remains necessary.
