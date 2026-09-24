# No-expert-module condition

This branch removes the optional MBA module (knowledge and its module-specific
measurement helpers) and the separate `knowledge/accelerator_physics_design.md`.
It is an **expert-module ablation**, not a claim to remove the model's pretrained
physics knowledge or a knowledge-text-only experiment.

## What remains identical

- Configured model/provider, Python, Julia and JuTrack are not changed by this ablation.
- Native JuTrack API documentation and focused API help.
- General lattice, optics, tracking, matching, optimization and parallel APIs.
- Research-program execution, durable jobs, repair, saved projects and evidence checks.
- The general runtime/evidence handbook and built-in `brilliance.accelerator_core`.

The file `brilliance/resources/expert.toml` is deliberately retained: despite its
name, it registers ordinary and durable program execution. Removing it would
confound the experiment by disabling computation.

## Run this condition

1. Keep `[experts].enabled = false` in the effective runtime TOML. This blocks all
   optional discovery roots, including `BRILLIANCE_EXPERT_PATH` and project modules.
   If `.env` selects another runtime TOML, set the same flag there.
2. Restart the application and start a **new project and model thread**. Do not
   resume or restore an old conversation: it has already seen expert knowledge.
3. Confirm the new project's `workspace/RESEARCH_ENVIRONMENT.json` records
   `experts.enabled: false` and `EXPERT_CATALOG.json` contains only the core module.
   No `laboratory.mba_design` guide, asset or tool should be staged or advertised.
4. Use requirements-only task prompts, identical across conditions. A prompt that
   teaches a TME/HMBA construction sequence reintroduces the ablated expertise.

Existing projects and their expert snapshots are intentionally preserved. Historical
results are not relabeled as ablated runs. Native simulation checks are unchanged;
exporting or accepting a result does not bypass scientific evaluation.

## Isolation for publication

Disabling discovery does not sandbox shell/file access. This development checkout
still contains old projects, private benchmark results, example exports and Git
history. For publishable runs, use an isolated account/workspace with only the
condition's app/runtime, common API/docs and task inputs. Do not expose prior
solutions, reference lattices, expert backups, other conditions' histories or the
offline grader. Start from fresh process/thread state, not just an empty chat view.

Keep model snapshot/provider/reasoning, shared numerical APIs, network tools,
compute allocation, budgets and intervention policy matched. Live web access can
rediscover expert knowledge; report this as "without supplied expert modules",
not "without access to expertise". Use a common frozen source collection if the
study needs controlled retrieval.

Full-versus-no-expert isolates the optional module bundle. Generic Codex versus
Brilliance also changes orchestration unless the same harness is used; describe
that comparison as a system baseline rather than attributing every difference
to evidence handling. All conditions must be judged by the same independent
numerical evaluator, including an optional no-acceptance-feedback condition.

The previous demos remain case studies. Matched, fresh full-Brilliance repetitions
are needed alongside the ablations for causal success-rate or efficiency claims.
