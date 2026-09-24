# Brilliance

**Beam Research Intelligence with Large Language-model Integration for Accelerator Numerical Computation and Exploration**

Brilliance is a conversational research assistant for accelerator physics. A language model writes and runs research programs using **JuTrack**, presents numerical results, tables and plots, and works with you to refine the study. Research code and evidence are saved with each project.

Different model providers and expandable expert knowledge are supported. MBA lattice design is the first developed research topic; Brilliance is not restricted to lattice design. This is research software: review numerical evidence and assumptions before publication or engineering use.

## Install

Requirements: Git, Python 3.10 or newer, and a supported model backend. Setup installs dependencies and configures Julia (1.10 or newer) and the bundled JuTrack integration. Initial setup needs network access and may take several minutes.

From the repository directory:

```powershell
git submodule update --init --recursive
python scripts/setup_env.py
.\.venv\Scripts\brilliance.exe --check
```

On Linux/macOS, use `python3` for setup and `.venv/bin/brilliance` for the launcher. Resolve any setup or runtime-check failure before starting a calculation. The agent uses the configured interpreter, not an environment of its own choosing.

## Configure

Copy `.env.example` to `.env` and add your provider credentials. Never commit `.env`.

| File | Purpose |
| --- | --- |
| `.env` | API keys and optional configuration-file locations |
| `brilliance/config/models.toml` | Provider endpoints, model profiles and selected backend |
| `brilliance/config/research_runtime.toml` | Simulation paths, compute resources, execution and network policy |

Use **Models & account** in the application to select a profile and inspect authentication:

- **ChatGPT subscription:** sign in through the official Codex runtime. No API key is required for this profile; model availability and usage depend on the account/runtime.
- **OpenAI API:** supply `OPENAI_API_KEY`. API billing is separate from a ChatGPT subscription.
- **IHEP/HepAI or another compatible API:** supply the key named by the profile's `api_key_env`, and set its endpoint and model in `models.toml`.

Research web access is enabled in the shipped configuration. For offline research, set `[execution].internet_access = false` and `[mcp].remote_http_enabled = false`. Local MCP tools can remain enabled. Remote model providers still require a network connection.

Set `[compute].local_worker_processes` and `threads_per_worker` to fit your machine. Research programs can use the shared parallel evaluator; increasing this budget does not automatically parallelize arbitrary code.

## Use

```powershell
.\.venv\Scripts\brilliance.exe
```

Open the address printed by the launcher (normally `http://127.0.0.1:8501`). For example:

> Design a 3.5 GeV, 32-cell 7BA storage-ring prototype around 800 m. Start with linear optics and radiation, explain the assumptions, and show one-cell optics together with the lattice.

Ask follow-up questions, change requirements, or request comparisons and plots in the same conversation. **Workspace** provides saved programs and artifacts. **Branch from here** restores an available conversation checkpoint into a separate project without overwriting the original.

Long numerical jobs can continue independently of the model conversation. Pausing the agent does not stop these jobs; use their own controls and check their state before restarting the application. Recovery depends on the numerical program's checkpoint support.

Unfinished requests continue automatically by default, with a shared allowance of
20 model turns per scientist message. Background-result reviews and recovery use
the same allowance; they do not reset it. Configure `auto_continue` and
`max_autonomous_increments` under `[interaction]` in `research_runtime.toml`.
The one-hour `turn_wall_time_s` is a model-turn checkpoint, not a total research
duration or a limit on independent numerical jobs. The agent stops when the
request is answered, a decision or blocker needs attention, you pause it, or its
allowance is exhausted. The existing consecutive-timeout guard also pauses a
stalled conversation after two full model time slices. Send a new message to resume from saved work with a new
allowance. Changing configuration does not restart paused projects.

Projects are saved under `research_projects/`. Back up this directory to preserve your work. New projects, credentials and local runtime state are excluded from Git by default.

## Demonstration

Two read-only snapshots of research conversations are viewable in the browser via GitHub Pages ([all examples](https://jinyu95.github.io/Brilliance/examples/index.html)):

- **MBA design** — [view online](https://jinyu95.github.io/Brilliance/examples/mba-design/index.html) · [open locally](examples/mba-design/index.html): the original browser-saved page from project-f82e15cdfbda87, unchanged, with its referenced figures, styles and script. Original application controls and artifact links are preserved; they may require the original local Brilliance server and project.
- **Space-charge study** — [view online](https://jinyu95.github.io/Brilliance/examples/space-charge/index.html) · [open locally](examples/space-charge/index.html): a coasting proton beam's space-charge tune footprint driven across the 4Qx&nbsp;=&nbsp;18 structure resonance in a fixed 1 GeV FODO ring, diagnosed with turn-by-turn emittance, phase space, and finite-time frequency-map (FMA) analysis (project-40c9bb49d1383e). Rendered from the saved transcript by `scripts/build_space_charge_example.py`.

After downloading the repository you can also open either `index.html` directly in a browser. Full research projects, private analysis and lattice exports are not bundled.

## Extend

Add an `EXPERT.md` under `expert_modules/<topic>/` to provide research knowledge. Optional Python tools expose reusable operations. See the [expert authoring guide](docs/EXPERT_AUTHORING_GUIDE.md) and [architecture overview](docs/ARCHITECTURE_CODEX.md).

JuTrack supplies the simulation engine; the model chooses and writes the research program. Saved candidates and numerical evidence, not model prose alone, establish what was calculated.
