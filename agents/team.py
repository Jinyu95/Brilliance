"""Assemble and run the BRILLIANCE agent team.

Pipeline:
  TaskPlanner -> CodeWriter -> CodeRunner -> CodeReviewer

Routing is deterministic:
  user        -> TaskPlanner
  TaskPlanner -> CodeWriter
  CodeWriter  -> CodeRunner
  CodeRunner  -> CodeReviewer
  CodeReviewer NEEDS_FIX  -> CodeWriter  (max 2 fix attempts)
  CodeReviewer TASK_COMPLETE / TASK_FAILED -> terminate
"""

from __future__ import annotations

from typing import Sequence

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, TextMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import (
    TextMentionTermination,
    MaxMessageTermination,
)
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

from .config import create_model_client, create_code_executor
from .prompts import (
    planner_prompt,
    codewriter_prompt,
    exploration_codewriter_prompt,
    reviewer_prompt,
    mba_stage_prompt,
    mba_reviewer_prompt,
)

def selector_prompt() -> str:
    return (
        "You are the speaker selector for a JuTrack lattice design team.\n\n"
        "Available agents (pick ONE next speaker):\n"
        "- TaskPlanner: parses user/AI requests into a structured TASK SPEC.\n"
        "  Has tools: check_design_feasibility, estimate_natural_emittance,\n"
        "  compare_lattice_families, get_reference_design, validate_task_spec.\n"
        "- CodeWriter: writes runnable pyJuTrack Python scripts from the TASK SPEC.\n"
        "- CodeRunner: executes the Python code and returns stdout/stderr.\n"
        "- CodeReviewer: inspects CodeRunner output, emits TASK_COMPLETE\n"
        "  (success) or NEEDS_FIX (retry with fix instructions) or TASK_FAILED.\n\n"
        "Decision logic (approximate — use your judgment):\n"
        "  First message MUST go to TaskPlanner.\n"
        "  TaskPlanner → CodeWriter (write code from the TASK SPEC).\n"
        "  If TaskPlanner says INFEASIBLE → CodeReviewer (validate diagnosis).\n"
        "  CodeWriter → CodeRunner (execute the code).\n"
        "  CodeRunner → CodeReviewer (review the output).\n"
        "  CodeReviewer NEEDS_FIX → CodeWriter (fix, max 3 loops then terminate).\n"
        "  CodeReviewer TASK_COMPLETE/TASK_FAILED → terminate (return None).\n"
        "  CRITICAL: if CodeWriter produced empty output or no code block,\n"
        "  send them back: 'No code found — please write a Python script.'\n"
        "  If CodeRunner reports 'No code blocks found', route BACK to CodeWriter.\n\n"
        "You MUST respond with EXACTLY ONE agent name: TaskPlanner, CodeWriter,\n"
        "CodeRunner, or CodeReviewer.  Do not add any other text."
    )
from .tools import PLANNER_TOOLS

# ── Agent name constants ──────────────────────────────────────────
PLANNER   = "TaskPlanner"
CODEWRITER = "CodeWriter"
RUNNER    = "CodeRunner"
REVIEWER  = "CodeReviewer"

# Legacy aliases
ANALYST   = PLANNER
ARCHITECT = PLANNER
REQUIREMENTS = "RequirementsEngineer"
PHYSICIST = "AcceleratorPhysicist"
DESIGN_CRITIC = "DesignCritic"
RESEARCH_MANAGER = "ResearchLead"


async def run_design(
    task: str,
    *,
    use_docker: bool = False,
    max_messages: int = 28,
    exploration_mode: bool = False,
):
    """Run the full pipeline and return the TaskResult."""
    selector_model_client = create_model_client(role="selector")
    executor = create_code_executor(use_docker)

    _cw_prompt = exploration_codewriter_prompt() if exploration_mode else codewriter_prompt()

    async with executor:
        planner = AssistantAgent(
            name=PLANNER,
            description=(
                "Parses the user's request and outputs a structured TASK SPEC "
                "(task type, lattice elements, variables, targets)."
            ),
            model_client=create_model_client(role="planner"),
            system_message=planner_prompt(),
            tools=PLANNER_TOOLS,
            reflect_on_tool_use=True,
        )
        codewriter = AssistantAgent(
            name=CODEWRITER,
            description=(
                "Translates the TASK SPEC into a complete, runnable pyJuTrack "
                "Python script using verified code templates."
            ),
            model_client=create_model_client(role="codewriter"),
            system_message=_cw_prompt,
        )
        code_runner = CodeExecutorAgent(
            name=RUNNER,
            description=(
                "Executes the Python code block from the previous message "
                "inside a sandboxed environment with pyJuTrack."
            ),
            code_executor=executor,
        )
        reviewer = AssistantAgent(
            name=REVIEWER,
            description=(
                "Checks execution output for errors. "
                "Replies TASK_COMPLETE, NEEDS_FIX, or TASK_FAILED."
            ),
            model_client=create_model_client(role="reviewer"),
            system_message=reviewer_prompt(),
        )

        termination = (
            TextMentionTermination("TASK_COMPLETE")
            | TextMentionTermination("TASK_FAILED")
            | TextMentionTermination("INFEASIBLE_DESIGN")
            | MaxMessageTermination(max_messages)
        )

        team = SelectorGroupChat(
            participants=[planner, codewriter, code_runner, reviewer],
            model_client=selector_model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt(),
        )

        return await Console(team.run_stream(task=task))


async def run_design_stream(
    task: str,
    *,
    use_docker: bool = False,
    max_messages: int = 28,
    on_message=None,
    exploration_mode: bool = False,
):
    """Run the pipeline with a per-message callback for UI integration."""
    selector_model_client = create_model_client(role="selector")
    executor = create_code_executor(use_docker)

    _cw_prompt = exploration_codewriter_prompt() if exploration_mode else codewriter_prompt()

    async with executor:
        planner = AssistantAgent(
            name=PLANNER,
            description="Parses request into a structured TASK SPEC.",
            model_client=create_model_client(role="planner"),
            system_message=planner_prompt(),
        )
        codewriter = AssistantAgent(
            name=CODEWRITER,
            description="Translates TASK SPEC into a pyJuTrack Python script.",
            model_client=create_model_client(role="codewriter"),
            system_message=_cw_prompt,
        )
        code_runner = CodeExecutorAgent(
            name=RUNNER,
            description="Executes Python code blocks from pyJuTrack.",
            code_executor=executor,
        )
        reviewer = AssistantAgent(
            name=REVIEWER,
            description="Checks output. Replies TASK_COMPLETE, NEEDS_FIX, or TASK_FAILED.",
            model_client=create_model_client(role="reviewer"),
            system_message=reviewer_prompt(),
        )

        termination = (
            TextMentionTermination("TASK_COMPLETE")
            | TextMentionTermination("TASK_FAILED")
            | TextMentionTermination("INFEASIBLE_DESIGN")
            | MaxMessageTermination(max_messages)
        )

        team = SelectorGroupChat(
            participants=[planner, codewriter, code_runner, reviewer],
            model_client=selector_model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt(),
        )

        last_result = None
        async for event in team.run_stream(task=task):
            if hasattr(event, "messages") and hasattr(event, "stop_reason"):
                last_result = event
                continue

            source = getattr(event, "source", "system")
            content = getattr(event, "content", "")
            if not isinstance(content, str) or not content.strip():
                continue

            msg_type = "thought" if isinstance(event, BaseAgentEvent) else "text"
            if on_message:
                on_message(source, content, msg_type)

        return last_result


# ── MBA multi-stage workflow ────────────────────────────────────────────

async def run_mba_design(
    task: str,
    *,
    use_docker: bool = False,
    max_messages_per_stage: int = 28,
) -> dict[str, Any]:
    """Run the full MBA construction methodology (10 sequential stages).

    Each stage uses the existing 4-agent pipeline, but scoped to one
    construction step. Accumulated state flows from stage to stage.

    Returns the final accumulated state dict.
    """
    from .mba_workflow import MBAWorkflowRunner

    runner = MBAWorkflowRunner()

    for stage in runner.stages:
        # Check required inputs
        missing = runner.check_required_inputs(stage)
        if missing:
            runner.log_stage(stage, False, {},
                             f"SKIPPED — missing inputs: {missing}")
            continue

        # Build prompt
        prompt = runner.build_stage_prompt(stage, task)

        # Run the 4-agent pipeline for this stage
        max_fixes = stage.max_fix_attempts + 1  # +1 for the initial attempt
        success = False

        for attempt in range(max_fixes):
            # Use stage-specific prompts
            _cw_prompt = mba_stage_prompt(stage.name)
            _rv_prompt = mba_reviewer_prompt(stage.name)

            selector_model_client = create_model_client(role="selector")
            executor = create_code_executor(use_docker)

            async with executor:
                planner = AssistantAgent(
                    name=f"{PLANNER}_{stage.name}",
                    description=f"Plans the '{stage.name}' MBA construction step.",
                    model_client=create_model_client(role="planner"),
                    system_message=_cw_prompt,
                    tools=PLANNER_TOOLS,
                    reflect_on_tool_use=True,
                )
                codewriter = AssistantAgent(
                    name=f"{CODEWRITER}_{stage.name}",
                    description=f"Implements the '{stage.name}' MBA construction step.",
                    model_client=create_model_client(role="codewriter"),
                    system_message=_cw_prompt,
                )
                code_runner = CodeExecutorAgent(
                    name=f"{RUNNER}_{stage.name}",
                    description="Executes Python code for MBA stage.",
                    code_executor=executor,
                )
                reviewer = AssistantAgent(
                    name=f"{REVIEWER}_{stage.name}",
                    description=f"Reviews the '{stage.name}' MBA stage output.",
                    model_client=create_model_client(role="reviewer"),
                    system_message=_rv_prompt,
                )

                termination = (
                    TextMentionTermination("STAGE_COMPLETE")
                    | TextMentionTermination("STAGE_FAILED")
                    | MaxMessageTermination(max_messages_per_stage)
                )

                from autogen_agentchat.teams import SelectorGroupChat

                team = SelectorGroupChat(
                    participants=[planner, codewriter, code_runner, reviewer],
                    model_client=selector_model_client,
                    termination_condition=termination,
                    selector_prompt=(
                        "Route: TaskPlanner -> CodeWriter -> CodeRunner -> CodeReviewer. "
                        "If NEEDS_FIX, route back to CodeWriter (max {max_fixes} attempts)."
                    ),
                )

                result = await Console(team.run_stream(task=prompt))

            # Extract output from result
            response_text = ""
            if result and hasattr(result, "messages"):
                for msg in result.messages:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        response_text += msg.content + "\n"

            if "STAGE_COMPLETE" in response_text:
                output = runner.extract_output_from_response(response_text)
                runner.accumulate(stage, output)
                runner.log_stage(stage, True, output, "STAGE_COMPLETE")
                success = True
                break
            elif "STAGE_FAILED" in response_text:
                runner.log_stage(stage, False, {}, "STAGE_FAILED")
                break
            # Otherwise retry (NEEDS_FIX or timeout)
        if not success:
            runner.log_stage(stage, False, {}, "Exhausted fix attempts")
            break  # stop the workflow; don't propagate garbage

    report = runner.build_final_report()
    print(report)
    return runner.accumulated_state


# ── Step-by-step single-agent calls (for user-in-the-loop UI) ────

_PROMPTS = {
    PLANNER:    planner_prompt,
    CODEWRITER: codewriter_prompt,
    REVIEWER:   reviewer_prompt,
}

_EXPLORATION_PROMPTS = {
    PLANNER:    planner_prompt,
    CODEWRITER: exploration_codewriter_prompt,
    REVIEWER:   reviewer_prompt,
}

_ROLE_NAMES = {
    PLANNER:    "planner",
    CODEWRITER: "codewriter",
    REVIEWER:   "reviewer",
}


def _latest_runner_messages(messages: list[dict]) -> list[TextMessage]:
    """Return only the latest code-bearing message for execution.

    Running the full conversation history can re-execute stale code blocks from
    earlier rounds, which corrupts autonomous feedback and repair loops.
    """
    fallback: dict | None = None
    for message in reversed(messages):
        if message.get("type") == "thought":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if message.get("source") == CODEWRITER:
            return [TextMessage(content=content, source=message["source"])]
        if fallback is None and "```" in content:
            fallback = message
    if fallback:
        return [TextMessage(content=fallback["content"], source=fallback["source"])]
    return []


async def call_agent(
    agent_name: str,
    messages: list[dict],
    *,
    use_docker: bool = False,
    exploration_mode: bool = False,
) -> dict:
    """Call a single agent with the conversation history.

    Parameters
    ----------
    agent_name : str
        One of PLANNER, CODEWRITER, RUNNER, REVIEWER.
    messages : list[dict]
        Conversation history as [{"source": str, "content": str, "type": str}].
    use_docker : bool
        Whether to use Docker for code execution (RUNNER only).
    exploration_mode : bool
        If True, use the exploration CodeWriter prompt that removes human priors.

    Returns
    -------
    dict with keys ``source``, ``content``, and optionally ``thoughts``.
    """
    auto_msgs = [
        TextMessage(content=m["content"], source=m["source"])
        for m in messages
        if m.get("type") != "thought"
    ]
    ct = CancellationToken()

    # CodeRunner needs the executor context manager
    if agent_name == RUNNER:
        runner_msgs = _latest_runner_messages(messages)
        if not runner_msgs:
            return {
                "source": RUNNER,
                "content": (
                    "The script ran, then exited with an error (POSIX exit code: 1)\n"
                    "Its output was:\n"
                    "No runnable code block was found in the latest code-generation step."
                ),
            }
        executor = create_code_executor(use_docker)
        async with executor:
            agent = CodeExecutorAgent(name=RUNNER, code_executor=executor)
            response = await agent.on_messages(runner_msgs, ct)
            return {"source": RUNNER, "content": response.chat_message.content}

    _prompts_dict = _EXPLORATION_PROMPTS if exploration_mode else _PROMPTS
    if agent_name not in _prompts_dict:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Valid: {list(_prompts_dict.keys()) + [RUNNER]}"
        )

    model_client = create_model_client(role=_ROLE_NAMES.get(agent_name))
    _tools = PLANNER_TOOLS if agent_name == PLANNER else None
    agent = AssistantAgent(
        name=agent_name,
        model_client=model_client,
        system_message=_prompts_dict[agent_name](),
        **({"tools": _tools, "reflect_on_tool_use": True} if _tools else {}),
    )
    response = await agent.on_messages(auto_msgs, ct)

    content = response.chat_message.content

    # Capture chain-of-thought from reasoning models (e.g. deepseek-reasoner)
    thoughts_text: str = ""
    if response.inner_messages:
        thoughts = []
        for inner in response.inner_messages:
            if isinstance(inner, BaseAgentEvent) and hasattr(inner, "content"):
                text = str(inner.content)
                if text.strip():
                    thoughts.append(text)
        if thoughts:
            thoughts_text = "\n\n".join(thoughts)

    # Some reasoning models (deepseek-reasoner, o1, etc.) emit the full
    # analysis in the thinking trace and produce only an empty or stub string
    # as the final content.  When this happens to ANY agent, promote the
    # thoughts text so downstream agents (especially CodeRunner) have something
    # to work with.
    _DECISION_WORDS = ("TASK_COMPLETE", "NEEDS_FIX", "TASK_FAILED",
                       "INFEASIBLE_DESIGN", "TASK SPEC", "```python",
                       "```Python", "import pyJuTrack", "import numpy")
    _is_stub = (
        len(content.strip()) < 80
        or (len(content.strip()) < 200
            and not any(kw in content for kw in ("```", "TASK", "NEEDS", "COMPLETE")))
    )
    if _is_stub and thoughts_text:
        content = thoughts_text
        thoughts_text = ""

    result: dict = {"source": agent_name, "content": content}
    if thoughts_text:
        result["thoughts"] = thoughts_text

    return result
