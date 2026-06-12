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
    reviewer_prompt,
)

def selector_prompt() -> str:
    return (
        "You are managing a JuTrack simulation pipeline. "
        "Route messages in order: TaskPlanner → CodeWriter → CodeRunner → CodeReviewer."
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
# Kept as string constants so import sites don't break,
# but these agents are no longer registered or used.
REQUIREMENTS = "RequirementsEngineer"
PHYSICIST = "AcceleratorPhysicist"
DESIGN_CRITIC = "DesignCritic"
RESEARCH_MANAGER = "ResearchLead"


def _route(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    """Deterministic speaker routing.

    Returns agent name or None to terminate.
    """
    if not messages:
        return PLANNER

    last: TextMessage | None = None
    for msg in reversed(messages):
        if isinstance(msg, TextMessage):
            last = msg
            break

    if last is None:
        return PLANNER

    source = last.source
    content = last.content

    if source == "user":
        return PLANNER

    if source == PLANNER:
        # If the Planner declares infeasibility, skip CodeWriter and go directly
        # to Reviewer so it can validate the diagnosis and emit TASK_COMPLETE.
        if "INFEASIBLE" in content and "feasibility_status" in content:
            return REVIEWER
        return CODEWRITER

    if source == CODEWRITER:
        return RUNNER

    if source == RUNNER:
        return REVIEWER

    if source == REVIEWER:
        if "NEEDS_FIX" in content:
            # Raise fix-attempt cap: 3 for ring designs, 5 for matching tasks.
            # Default cap is 3 (was 2).
            _max_fixes = 3
            fix_count = sum(
                1 for m in messages
                if isinstance(m, TextMessage)
                and m.source == REVIEWER
                and "NEEDS_FIX" in m.content
            )
            if fix_count >= _max_fixes:
                return None
            return CODEWRITER
        # TASK_COMPLETE, TASK_FAILED, or unknown → terminate
        return None

    return None


async def run_design(
    task: str,
    *,
    use_docker: bool = False,
    max_messages: int = 28,
):
    """Run the full pipeline and return the TaskResult."""
    selector_model_client = create_model_client(role="selector")
    executor = create_code_executor(use_docker)

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
            system_message=codewriter_prompt(),
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
            selector_func=_route,
        )

        return await Console(team.run_stream(task=task))


async def run_design_stream(
    task: str,
    *,
    use_docker: bool = False,
    max_messages: int = 28,
    on_message=None,
):
    """Run the pipeline with a per-message callback for UI integration."""
    selector_model_client = create_model_client(role="selector")
    executor = create_code_executor(use_docker)

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
            system_message=codewriter_prompt(),
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
            selector_func=_route,
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


# ── Step-by-step single-agent calls (for user-in-the-loop UI) ────

_PROMPTS = {
    PLANNER:    planner_prompt,
    CODEWRITER: codewriter_prompt,
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

    if agent_name not in _PROMPTS:
        raise ValueError(
            f"Unknown agent '{agent_name}'. Valid: {list(_PROMPTS.keys()) + [RUNNER]}"
        )

    model_client = create_model_client(role=_ROLE_NAMES.get(agent_name))
    _tools = PLANNER_TOOLS if agent_name == PLANNER else None
    agent = AssistantAgent(
        name=agent_name,
        model_client=model_client,
        system_message=_PROMPTS[agent_name](),
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

    # Some reasoning models emit the full analysis inside the thinking trace
    # and produce only a stub (e.g. "We") as the final token.  When the actual
    # content is suspiciously short and the key decision words are absent,
    # promote the thoughts text to be the displayed content so the reviewer
    # verdict is always visible in the chat.
    _DECISION_WORDS = ("TASK_COMPLETE", "NEEDS_FIX", "TASK_FAILED",
                       "INFEASIBLE_DESIGN", "TASK SPEC", "```python")
    if (
        len(content.strip()) < 120
        and thoughts_text
        and not any(kw in content for kw in _DECISION_WORDS)
    ):
        content = thoughts_text
        thoughts_text = ""   # already promoted; nothing extra to show

    result: dict = {"source": agent_name, "content": content}
    if thoughts_text:
        result["thoughts"] = thoughts_text

    return result
