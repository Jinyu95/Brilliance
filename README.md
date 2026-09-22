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

Projects are saved under `research_projects/`. Back up this directory to preserve your work. New projects, credentials and local runtime state are excluded from Git by default.

## Demonstration

[MBA design demonstration](examples/mba-design/README.md) presents selected results and an offline conversation with tables and figures from project-f82e15cdfbda87 (about 4.7 MiB). Open `examples/mba-design/index.html` in a browser; no server or model account is needed. This is a presentation of recorded design exploration, not a reproducibility package or a fully validated operational machine. Full research projects, source programs, private analysis and lattice exports are not bundled.

## Extend

Add an `EXPERT.md` under `expert_modules/<topic>/` to provide research knowledge. Optional Python tools expose reusable operations. See the [expert authoring guide](docs/EXPERT_AUTHORING_GUIDE.md) and [architecture overview](docs/ARCHITECTURE_CODEX.md).

JuTrack supplies the simulation engine; the model chooses and writes the research program. Saved candidates and numerical evidence, not model prose alone, establish what was calculated.
