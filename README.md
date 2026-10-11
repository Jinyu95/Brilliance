# Brilliance

**Beam Research Intelligence with Large Language-model Integration for Accelerator Numerical Computation and Exploration**

Brilliance is a conversational research agent for accelerator physics. A language model writes and runs research programs using **JuTrack**, presents numerical results, tables and plots, and works with you to refine the study. Research code and evidence are saved with each project.

## Install

Requirements: Git, Python 3.10 or newer, and a supported LLM backend. Setup installs dependencies and configures Julia (1.10 or newer) and the bundled JuTrack integration. Initial setup needs network access and may take several minutes.

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
- **DeepSeek official API:** supply `DEEPSEEK_API_KEY`. The profile uses `deepseek-flash`; the Model field also accepts `deepseek-v4-pro` ([official API documentation](https://api-docs.deepseek.com/)).
- **IHEP/HepAI or another compatible API:** supply the key named by the profile's `api_key_env`, and set its endpoint and model in `models.toml`.

Research web access is enabled in the shipped configuration. For offline research, set `[execution].internet_access = false` and `[mcp].remote_http_enabled = false`. Remote model providers still require a network connection.

Set `[compute].local_worker_processes` and `threads_per_worker` to fit your machine. Research programs can use the shared parallel evaluator; increasing this budget does not automatically parallelize arbitrary code.

## Use

On Windows PowerShell:
```powershell
.\Brilliance.cmd
```
or you can double-click **Brilliance.cmd** in the project folder after completing
installation and configuration. You can also create a desktop shortcut to this file.

On Linux/macOS, use 
```
.venv/bin/brilliance
```

Open the address printed by the launcher (normally `http://127.0.0.1:8501`). Ask follow-up questions, change requirements, or request comparisons and plots in the same conversation.

Configure `auto_continue` and `max_autonomous_increments` under `[interaction]` in `research_runtime.toml`. The agent stops when the request is answered, a decision or blocker needs attention, you pause it, or its allowance is exhausted. The existing consecutive-timeout guard also pauses a stalled conversation after two full model time slices. Send a new message to resume from saved work with a new allowance. Changing configuration does not restart paused projects.

Research results are saved under `research_projects/`. Back up this directory to preserve your work.

## Demonstration

Read-only snapshots of research conversations are available in [the example gallery](https://jinyu95.github.io/Brilliance/examples/index.html).


## Extend

The full application supports `EXPERT.md` modules and optional Python tools. See the [expert authoring guide](docs/EXPERT_AUTHORING_GUIDE.md). Add an `EXPERT.md` under `expert_modules/<topic>/` to provide research knowledge. Optional Python tools expose reusable operations.
