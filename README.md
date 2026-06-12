# BRILLIANCE

**Beam Research with Intelligent LLM-Informed Accelerator Numerical Computation Engine**

AI agent for synchrotron storage ring lattice design. A four-agent pipeline (TaskPlanner → CodeWriter → CodeRunner → CodeReviewer) generates and validates [pyJuTrack](https://github.com/MSU-Beam-Dynamics/JuTrack.jl) scripts from plain-language physics requirements.

---

## Quick Start (Docker)

The only prerequisite is [Docker Desktop](https://docs.docker.com/get-docker/).

```bash
git clone https://github.com/your-org/brilliance.git
cd brilliance
cp .env.example .env          # then edit .env — set LLM_API_KEY
docker compose up --build     # first run: ~5 min (compiles Julia + JuTrack)
```

Open **http://localhost:8501** in your browser.

Subsequent launches skip compilation and start in seconds:

```bash
docker compose up
```

> **Restricted network** — if Docker Hub, julialang.org, or PyPI are unreachable,
> use the `mirror` profile which routes all downloads through alternative mirrors
> (dockerpull.org / BFSU Julia / Tsinghua pip):
> ```bash
> docker compose --profile mirror up --build
> ```

### LLM configuration

Edit `.env` and set these three variables:

```
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
```

Any OpenAI-compatible provider works — DeepSeek, OpenAI, OpenRouter, Ollama, vLLM.

---

## Agent Pipeline

| Agent | Role |
|---|---|
| **TaskPlanner** | Parses the request into a design specification; estimates feasibility |
| **CodeWriter** | Translates the specification into a runnable pyJuTrack Python script |
| **CodeRunner** | Executes the script in the sandbox and captures output |
| **CodeReviewer** | Evaluates stability, tunes, emittance, chromaticity; routes back to CodeWriter on failure |

The Planner emits `INFEASIBLE_DESIGN` for physically impossible requests (stage-10 benchmark) and routes directly to the Reviewer for diagnosis, skipping code generation entirely.

---

## Benchmark Suite

Eight staged tasks covering single-element matrices through full-ring assembly and impossible-design detection:

```bash
docker compose run --rm app python run_benchmarks.py
```

Tasks are defined in [benchmarks/tasks.json](benchmarks/tasks.json). Each benchmark also appears as a suggested prompt in the Streamlit sidebar.

---

## Project Structure

```
brilliance/
├── app.py                   # Streamlit web UI
├── run_design.py            # CLI entry point
├── run_benchmarks.py        # End-to-end benchmark harness
├── physics_core.py          # Deterministic scoring and result parsing
├── pyproject.toml           # Package metadata
├── juliapkg.json            # Pins bundled JuTrack.jl for juliacall
├── docker-compose.yml       # User-facing Docker entry point
├── agents/
│   ├── team.py              # Agent definitions and routing
│   ├── prompts.py           # System prompts with pyJuTrack API rules
│   ├── config.py            # LLM client and executor factories
│   ├── session.py           # Session persistence
│   ├── tools.py             # Planner tools (emittance estimator, reference lookup)
│   ├── benchmark.py         # Benchmark harness
│   └── constraints.py       # Constraint parsing from prompt text
├── benchmarks/
│   └── tasks.json           # Benchmark task definitions
├── templates/plain/         # Reference lattice scripts (FODO, DBA, 7BA, HEPS)
├── knowledge/               # Accelerator physics knowledge graph
├── JuTrack.jl/              # Bundled JuTrack source (Julia tracking library)
├── scripts/
│   ├── Dockerfile.jutrack   # Multi-stage Docker build
│   └── setup_env.py         # Local venv setup (alternative to Docker)
├── sessions/                # Saved design sessions (runtime, git-ignored)
└── workspace/               # Code execution sandbox (runtime, git-ignored)
```

---

## Supported Lattice Types

| Type | Description | Example |
|---|---|---|
| **FODO** | Simple booster rings | 2.5 GeV, 20 cells |
| **DBA** | Double-bend achromat (3rd-gen) | 3 GeV, 12 cells |
| **7BA / MBA** | Multi-bend achromat (4th-gen, ultra-low emittance) | 6 GeV, 24 superperiods |

Reference templates in [templates/plain/](templates/plain/) serve as starting points for the code writer and cover 2–6 GeV across all three lattice families.

---

## Local Install (without Docker)

Requires Python ≥ 3.10 and [Julia ≥ 1.10](https://julialang.org/downloads/).

```bash
python scripts/setup_env.py   # creates .venv, installs all deps
# edit .env
.venv/Scripts/streamlit run app.py    # Windows
# or
.venv/bin/streamlit run app.py        # Linux / macOS
```

---

## Knowledge Graph

The knowledge graph in `knowledge/.graph/` is pre-built and ships with the repo. To rebuild it from your own reference PDFs placed in `knowledge/`:

```bash
python scripts/build_knowledge_graph.py
```
