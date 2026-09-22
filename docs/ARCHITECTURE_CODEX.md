# Architecture

Brilliance combines a conversational interface, a selectable model backend, and a configured numerical runtime. It does not prescribe one accelerator-design workflow.

## Research loop

1. The scientist supplies a question and machine requirements.
2. The model receives project context, tool documentation and relevant expert knowledge.
3. The model writes a saved research program or invokes a research tool.
4. Programs execute with the configured Python/Julia/JuTrack environment. Durable jobs can outlive a model turn.
5. Results, artifacts and scientific evidence appear in the conversation for review and follow-up.

Execution completion and scientific acceptance are separate. A program can succeed without meeting its physical target. Reports must identify the candidate being discussed.

## Components

| Component | Responsibility |
| --- | --- |
| `web_app.py`, `web/static/` | Web application, conversation and artifacts |
| `chat_service.py` | Conversation lifecycle and research handoff |
| `codex_runtime.py`, `codex_account.py` | Official Codex runtime and managed subscription authentication |
| `native_research_agent.py`, `model_settings.py` | Native API tool loop and selectable provider profiles |
| `research_program.py`, `jobs.py` | Saved programs, durable execution and progress |
| `accelerator_api.py` | Research-facing JuTrack operations |
| `parallel_evaluation.py` | Resource-bounded evaluation with isolated worker processes |
| `projects.py`, `conversation_checkpoints.py` | Persistence and checkpoint branching |
| `experts/`, `expert_modules/` | Topic knowledge and optional reusable tools |

Both model backends use the research tool and project infrastructure. Subscription authentication is managed by Codex, not treated as an API key. API profiles have separate endpoints and credentials.

## Configuration and storage

`config/models.toml` selects the provider and model. `config/research_runtime.toml` defines simulation paths, network policy and compute resources. `.env` supplies credentials and optional configuration paths.

Each project owns its conversation and workspace. Programs, exact candidates, evaluations and plots remain associated with it. Checkpoint branching restores recoverable research state into a new project; it does not adopt running jobs.

Expert modules supply domain knowledge without turning the core application into an MBA-specific designer. JuTrack remains the simulation source. Model-authored programs use the installed scientific Python stack for optimization, analysis and plotting.
