"""
Multi-Agent Orchestrator — 4-Agent Pipeline
--------------------------------------------
Every task runs through four specialised agents:

  Agent 1 — DecompositionAgent
    Input : big task (str)
    Output: ordered list of subtasks stored in TaskState.plan

  Agent 2 — ToolSelectionAgent          (per loop iteration)
    Input : current subtask + context + previous observations for this step
    Output: which tool to call + what arguments to pass

  Agent 3 — ToolExecutorAgent           (per loop iteration, no LLM)
    Input : tool_name + tool_args
    Output: raw tool observation (JSON str)

  Agent 4 — EvaluationAgent             (per loop iteration)
    Input : subtask goal + all observations collected so far for this step
    Output: { status: "done" | "retry" | "failed", reason, answer }

  Loop per subtask:
    ┌─────────────────────────────────────────────────────┐
    │  while not done and retries < max_retries:           │
    │    selected = ToolSelectionAgent.select(...)         │
    │    observation = ToolExecutorAgent.execute(...)      │
    │    evaluation = EvaluationAgent.evaluate(...)        │
    │    if done  → store result, move to next subtask     │
    │    if retry → loop again with updated observations   │
    │    if failed → mark step failed, continue            │
    └─────────────────────────────────────────────────────┘

  OrchestratorAgent drives everything, fires events so
  main.py can render live progress to the console.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import selector_cache as sc
from llm_agent import (
    BedrockClient,
    _user_message,
    _extract_text,
    _extract_tool_uses,
    execute_tool_call,
    get_all_tool_schemas,
)
import playwright_tools as pt

logger = logging.getLogger(__name__)


# ===========================================================================
# State model
# ===========================================================================

class StepStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"
    SKIPPED  = "skipped"


class LoopStatus(str, Enum):
    DONE   = "done"
    RETRY  = "retry"
    FAILED = "failed"


@dataclass
class LoopIteration:
    """One Select → Execute → Evaluate cycle inside a single subtask."""
    number: int                  # 1-based within the step
    tool_name: str               # chosen by ToolSelectionAgent
    tool_args: dict              # args chosen by ToolSelectionAgent
    selection_reasoning: str     # why the agent chose this tool
    observation: str             # raw JSON output from tool execution
    eval_status: LoopStatus = LoopStatus.RETRY
    eval_reason: str = ""        # evaluator's explanation
    eval_answer: str = ""        # populated when eval_status == DONE


@dataclass
class PlanStep:
    """One subtask in the decomposed plan."""
    index: int                   # 1-based
    description: str
    status: StepStatus = StepStatus.PENDING
    result: str = ""             # final answer when DONE
    error: str = ""              # reason when FAILED
    iterations: list[LoopIteration] = field(default_factory=list)

    @property
    def all_observations(self) -> list[str]:
        return [it.observation for it in self.iterations]

    @property
    def all_tool_calls(self) -> list[dict]:
        return [{"tool": it.tool_name, "args": it.tool_args} for it in self.iterations]


@dataclass
class TaskState:
    """Single source of truth for the whole task execution."""
    original_task: str
    plan: list[PlanStep] = field(default_factory=list)
    current_step_index: int = 0

    @property
    def current_step(self) -> Optional[PlanStep]:
        if self.current_step_index < len(self.plan):
            return self.plan[self.current_step_index]
        return None

    @property
    def completed_steps(self) -> list[PlanStep]:
        return [s for s in self.plan if s.status == StepStatus.DONE]

    @property
    def all_tool_calls(self) -> list[dict]:
        calls = []
        for step in self.plan:
            for it in step.iterations:
                calls.append({"step": step.index, "tool": it.tool_name, "args": it.tool_args})
        return calls

    def context_for_step(self) -> str:
        """
        Summary of all DONE steps injected as context into each agent call.
        Tells the agent what has already been accomplished and what data was found.
        """
        done = self.completed_steps
        if not done:
            return "No steps completed yet. This is the first step."
        lines = ["Previously completed steps:"]
        for s in done:
            lines.append(f"\n  Step {s.index}: {s.description}")
            lines.append(f"  Result: {s.result[:400]}" + ("..." if len(s.result) > 400 else ""))
        return "\n".join(lines)


# ===========================================================================
# Orchestrator events — for main.py to render live progress
# ===========================================================================

@dataclass
class OrchestratorEvent:
    """
    Fired by OrchestratorAgent at every significant moment.

    event_type:
      "plan"         — decomposition done, plan ready
      "step_start"   — about to begin a subtask
      "loop_select"  — ToolSelectionAgent chose a tool
      "loop_execute" — tool executed, observation received
      "loop_eval"    — EvaluationAgent returned verdict
      "step_done"    — subtask completed successfully
      "step_failed"  — subtask failed (all retries exhausted)
      "done"         — all subtasks done, final answer ready
    """
    event_type: str
    state: TaskState
    step: Optional[PlanStep] = None
    iteration: Optional[LoopIteration] = None
    final_answer: str = ""


# ===========================================================================
# Agent 1 — DecompositionAgent
# ===========================================================================

class DecompositionAgent:
    """
    Calls the LLM with a single forced tool submit_plan(steps, reasoning).
    Forces structured output so we always get a clean ordered list.
    """

    _TOOL = {
        "toolSpec": {
            "name": "submit_plan",
            "description": (
                "Submit the ordered step-by-step plan for the task. "
                "Each step must be a single concrete, self-contained browser action "
                "that a Playwright automation agent can execute independently."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "description": "Ordered list of step descriptions.",
                            "items": {"type": "string"},
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why you chose this breakdown.",
                        },
                    },
                    "required": ["steps", "reasoning"],
                }
            },
        }
    }

    _SYSTEM = (
        "You are a task decomposition agent for a browser automation system. "
        "Break the given task into ordered steps. Each step must have a single observable "
        "success condition — something verifiable in the page URL, title, or visible text.\n"
        "RULES:\n"
        "- A step that involves entering data AND submitting must be split into two steps.\n"
        "- Never create a step whose only purpose is waiting.\n"
        "- Do not create separate steps for finding a page and going to it — that is one step. "
        "  If a destination has a known URL, prefer navigating to it directly.\n"
        "- Do not duplicate steps with overlapping goals.\n"
        "Always call submit_plan to return your plan."
    )

    def __init__(self, llm: BedrockClient):
        self.llm = llm

    async def decompose(self, task: str) -> tuple[list[str], str]:
        """
        Returns (steps, reasoning).
        Falls back to [task] as single step if the model doesn't call the tool.
        """
        response = await self.llm.converse(
            messages=[_user_message(f"Decompose this task into steps:\n\n{task}")],
            system=self._SYSTEM,
            tools=[self._TOOL],
        )
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)

        if not tool_uses:
            logger.warning("DecompositionAgent: no tool call — single-step fallback")
            return [task], "Single step fallback."

        inp = tool_uses[0].get("input", {})
        steps = [s.strip() for s in inp.get("steps", [task]) if s.strip()]
        reasoning = inp.get("reasoning", "")
        return steps, reasoning


# ===========================================================================
# Tool group definitions — each specialist agent sees only its group
# ===========================================================================

TOOL_GROUPS: dict[str, dict] = {
    "navigation": {
        "description": "Go to URLs, go back/forward, reload the page",
        "tools": ["navigate", "go_back", "go_forward", "reload", "get_current_url"],
    },
    "page_reading": {
        "description": "Read the current page structure, HTML, text, or element attributes",
        "tools": ["get_page_html", "get_page_snapshot", "get_text_blocks",
                  "get_text", "get_all_text", "get_attribute", "get_input_value",
                  "get_page_content", "evaluate_js", "evaluate_js_on_element"],
    },
    "form_input": {
        "description": "Type into inputs, fill forms, select dropdowns, check boxes, upload files",
        "tools": ["fill", "type_text", "press_key", "select_option",
                  "check_checkbox", "uncheck_checkbox", "upload_file"],
    },
    "clicking": {
        "description": "Click, double-click, hover, drag, or press keyboard keys on elements",
        "tools": ["click", "hover", "drag_and_drop", "focus", "press_key"],
    },
    "waiting": {
        "description": "Wait for elements, URLs, or page load states before acting",
        "tools": ["wait_for_selector", "wait_for_url", "wait_for_load_state", "wait_for_timeout"],
    },
    "network": {
        "description": "Make direct HTTP requests, intercept or block network calls",
        "tools": ["fetch_url", "intercept_route", "abort_route"],
    },
    "tabs_frames": {
        "description": "Open new tabs, switch tabs, interact with iframes",
        "tools": ["new_tab", "switch_tab", "close_tab", "list_tabs",
                  "switch_to_frame", "get_frame_text", "click_in_frame"],
    },
    "storage": {
        "description": "Read or write cookies, localStorage, or save session state",
        "tools": ["get_cookies", "set_cookies", "clear_cookies",
                  "get_local_storage", "set_local_storage", "save_storage_state"],
    },
    "capture": {
        "description": "Take screenshots or export the page as PDF",
        "tools": ["screenshot", "save_pdf"],
    },
    "assertions": {
        "description": "Assert or verify page state — check visibility, text content, URL, or title",
        "tools": ["assert_visible", "assert_text", "assert_url_contains", "assert_title"],
    },
    "debug": {
        "description": "Record a Playwright trace for debugging failed runs",
        "tools": ["start_tracing", "stop_tracing"],
    },
}

# Groups available before any navigation has happened
_PRE_NAV_GROUPS = {"navigation", "network"}

# Groups that are relevant on any loaded page
_POST_NAV_GROUPS = {"navigation", "page_reading", "form_input", "clicking",
                    "waiting", "tabs_frames", "capture", "assertions", "storage", "network"}


# ===========================================================================
# Agent 2a — ToolRouterAgent  (picks the group, ~8 choices)
# ===========================================================================

class ToolRouterAgent:
    """
    Layer 1 of the 2-layer tool selector.
    Sees only group names (~8 choices) and picks which specialist group
    is most appropriate for the current step.
    Tiny choice set → near-zero hallucination.
    """

    _ROUTE_TOOL = {
        "toolSpec": {
            "name": "select_group",
            "description": "Select the tool group most appropriate for the next action.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "group": {
                            "type": "string",
                            "description": "The tool group name to use.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this group is the right one.",
                        },
                    },
                    "required": ["group", "reasoning"],
                }
            },
        }
    }

    _SYSTEM_TEMPLATE = (
        "You are a routing agent for a browser automation system. "
        "Your only job is to pick which TOOL GROUP is needed next.\n\n"
        "AVAILABLE GROUPS:\n{group_list}\n\n"
        "Rules:\n"
        "- The browser is already open. Never pick a group for starting/stopping the browser.\n"
        "- Always call select_group with your choice."
    )

    def __init__(self, llm: BedrockClient):
        self.llm = llm
        self._valid_groups = set(TOOL_GROUPS.keys())

    def _build_group_list(self, available: set[str]) -> str:
        lines = []
        for name in available:
            grp = TOOL_GROUPS[name]
            lines.append(f"  {name} — {grp['description']}")
        return "\n".join(lines)

    async def route(
        self,
        step_description: str,
        context: str,
        observations: list[str],
    ) -> tuple[str, str]:
        """
        Pick a tool group.

        Returns:
            (group_name, reasoning)
        """
        has_page = any("html" in o or "navigated_to" in o or "page_url" in o
                       for o in observations)
        if not has_page:
            try:
                current_url = pt._state["page"].url if pt._state["page"] else ""
                has_page = bool(current_url and current_url not in ("about:blank", ""))
            except Exception:
                pass
        available = _POST_NAV_GROUPS if has_page else _PRE_NAV_GROUPS

        # Give router current URL + last non-HTML observation so it has page context
        page_context = ""
        try:
            if pt._state.get("page"):
                page_context = f"\nCurrent URL: {pt._state['page'].url}"
        except Exception:
            pass

        obs_summary = ""
        if observations:
            # Strip html blobs, show last observation status
            for o in reversed(observations):
                try:
                    parsed = json.loads(o)
                    display = {k: v for k, v in parsed.items()
                               if k not in ("html", "page_html") and len(str(v)) < 300}
                    obs_summary = f"\nLast observation: {json.dumps(display)[:400]}"
                    break
                except Exception:
                    obs_summary = f"\nLast observation: {o[:300]}"
                    break

        prompt = (
            f"CURRENT STEP GOAL:\n{step_description}\n\n"
            f"CONTEXT:\n{context}"
            f"{page_context}"
            f"{obs_summary}\n\n"
            "Which tool GROUP should be used for the next action?"
        )

        system = self._SYSTEM_TEMPLATE.format(
            group_list=self._build_group_list(available)
        )

        response = await self.llm.converse(
            messages=[_user_message(prompt)],
            system=system,
            tools=[self._ROUTE_TOOL],
        )
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)

        if not tool_uses:
            fallback = "page_reading" if has_page else "navigation"
            logger.warning("ToolRouterAgent: no tool call — fallback to '%s'", fallback)
            return fallback, "Fallback"

        inp = tool_uses[0].get("input", {})
        group = inp.get("group", "page_reading")

        # Correct hallucinated group names
        if group not in self._valid_groups:
            import difflib
            matches = difflib.get_close_matches(group, available, n=1, cutoff=0.4)
            group = matches[0] if matches else ("page_reading" if has_page else "navigation")
            logger.warning("ToolRouterAgent: corrected group to '%s'", group)

        return group, inp.get("reasoning", "")


# ===========================================================================
# Agent 2b — ToolSpecialistAgent  (picks exact tool, ~5 choices)
# ===========================================================================

class ToolSpecialistAgent:
    """
    Layer 2 of the 2-layer tool selector.
    Receives a specific tool group (chosen by ToolRouterAgent) and picks
    the exact tool + arguments from that group's small tool list (~3–9 tools).
    Tiny choice set → near-zero hallucination.
    """

    _SELECT_TOOL = {
        "toolSpec": {
            "name": "select_tool",
            "description": "Select the exact tool to call and specify its arguments.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Exact tool name from the available list.",
                        },
                        "tool_args": {
                            "type": "object",
                            "description": "Arguments as key-value pairs.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this tool and these args.",
                        },
                    },
                    "required": ["tool_name", "tool_args", "reasoning"],
                }
            },
        }
    }

    _SYSTEM_TEMPLATE = (
        "You are a specialist tool agent for browser automation. "
        "You have been assigned the '{group}' group. "
        "Choose the SINGLE best tool from the list below to make progress on the current step.\n\n"
        "AVAILABLE TOOLS IN THIS GROUP:\n{tool_list}\n\n"
        "SELECTOR SYNTAX — you may use any of these:\n"
        "  #id                          when the element has an id\n"
        "  a[href*='careers']           match part of a URL\n"
        "  button:has-text(\"Company\")   match by visible text  ← USE THIS when no id/aria-label\n"
        "  text=Careers                 exact visible text\n"
        "  a >> nth=3                   positional, last resort\n"
        "Never invent a URL. Only navigate to hrefs you have READ from the page HTML.\n\n"
        "RULES:\n"
        "1. Only use tool names exactly as listed above.\n"
        "2. Derive selectors from the CURRENT PAGE HTML in the prompt — "
        "   prefer id, aria-label/title, name, text, then href for links.\n"
        "3. Never pass a 'timeout' parameter.\n"
        "4. To submit a search form: use press_key(selector, 'Enter') on the input.\n"
        "5. Always call select_tool."
    )

    def __init__(self, llm: BedrockClient, all_tool_schemas: list[dict]):
        self.llm = llm
        self._schema_map = {s["toolSpec"]["name"]: s for s in all_tool_schemas}

    def _tool_list_for_group(self, group: str) -> str:
        tools = TOOL_GROUPS.get(group, {}).get("tools", [])
        lines = []
        for name in tools:
            schema = self._schema_map.get(name)
            if not schema:
                continue
            spec = schema["toolSpec"]
            props = spec["inputSchema"]["json"].get("properties", {})
            required = spec["inputSchema"]["json"].get("required", [])
            param_str = ", ".join(props.keys())
            req_note  = f" [required: {', '.join(required)}]" if required else ""
            lines.append(f"  {name}({param_str}){req_note} — {spec['description'][:100]}")
        return "\n".join(lines)

    def _valid_tools_for_group(self, group: str) -> set[str]:
        return set(TOOL_GROUPS.get(group, {}).get("tools", []))

    async def select(
        self,
        group: str,
        step_description: str,
        context: str,
        observations: list[str],
        feedback: str = "",
        tried: list[dict] | None = None,
    ) -> tuple[str, dict, str]:
        """
        Pick exact tool + args within the given group.

        Returns:
            (tool_name, tool_args, reasoning)
        """
        obs_section = ""
        available_selectors = []
        latest_html = ""

        if observations:
            obs_lines = []
            for i, o in enumerate(observations):
                try:
                    parsed = json.loads(o)
                    html_chunk = parsed.get("html") or parsed.get("page_html") or ""
                    if html_chunk:
                        latest_html = html_chunk

                    # Extract selectors from structured elements list
                    for el in parsed.get("elements", []):
                        sel = el.get("selector", "")
                        tag = el.get("tag", "")
                        label = (el.get("aria_label") or el.get("placeholder")
                                 or el.get("text") or el.get("name") or "")
                        if sel:
                            available_selectors.append(f"  {sel}  ({tag} — {label[:60]})")

                    # Also extract selectors from error messages produced by _validate_selector
                    # Error format: "...VISIBLE + ENABLED SELECTORS (copy one exactly):\n  sel1\n  sel2..."
                    error_text = parsed.get("error", "")
                    if "SELECTOR ERROR" in error_text and "SELECTORS" in error_text:
                        for line in error_text.splitlines():
                            stripped = line.strip()
                            if stripped.startswith(("input", "button", "a[", "textarea",
                                                     "select", "a#", "button#")):
                                # Pull just the selector part (before the parenthesis)
                                sel = stripped.split("  (")[0].strip()
                                if sel:
                                    available_selectors.append(f"  {sel}  (from error — use this)")
                        # Also use the latest HTML from the error if present
                        if "CURRENT PAGE HTML:" in error_text and not latest_html:
                            html_start = error_text.find("CURRENT PAGE HTML:") + len("CURRENT PAGE HTML:")
                            latest_html = error_text[html_start:].strip()[:8000]

                    display = {k: v for k, v in parsed.items()
                               if k not in ("html", "page_html")}
                    obs_lines.append(f"  Observation {i+1}: {json.dumps(display)[:400]}")
                except (json.JSONDecodeError, TypeError):
                    obs_lines.append(f"  Observation {i+1}: {o[:400]}")

            obs_section = "\nOBSERVATIONS:\n" + "\n".join(obs_lines)

        selector_section = ""
        if available_selectors:
            selector_section = (
                "\n\nAVAILABLE SELECTORS (copy exactly — do not modify):\n"
                + "\n".join(available_selectors)
            )

        html_section = ""
        if latest_html:
            html_section = f"\n\nCURRENT PAGE HTML:\n{latest_html}"

        # Inject selector memory from cache for the current domain
        cache_hints = ""
        try:
            current_url = pt._state["page"].url if pt._state.get("page") else ""
            if current_url:
                hints = sc.get_domain_hints(current_url)
                if hints:
                    cache_hints = f"\n\n{hints}"
        except Exception:
            pass

        tool_list = self._tool_list_for_group(group)
        system = self._SYSTEM_TEMPLATE.format(group=group, tool_list=tool_list)

        # Evaluator feedback (recency: placed near end of prompt)
        feedback_section = ""
        if feedback:
            feedback_section = (
                f"\n\nEVALUATOR FEEDBACK — act on this, do something DIFFERENT:\n{feedback}"
            )

        tried_section = ""
        if tried:
            sigs = [f"  {c['tool']}({json.dumps(c['args'])[:120]})" for c in (tried or [])]
            tried_section = (
                "\n\nALREADY TRIED THIS STEP — do NOT repeat any of these exactly:\n"
                + "\n".join(sigs)
            )

        prompt = (
            f"STEP GOAL:\n{step_description}\n\n"
            f"CONTEXT:\n{context}"
            f"{obs_section}"
            f"{cache_hints}"
            f"{html_section}"
            f"{tried_section}"
            f"{feedback_section}\n\n"
            f"You are in the '{group}' group. Which tool should be called next?"
        )

        response = await self.llm.converse(
            messages=[_user_message(prompt)],
            system=system,
            tools=[self._SELECT_TOOL],
        )
        logger.info("SPECIALIST prompt=%d chars | html=%d | selectors=%d",
                    len(prompt), len(latest_html), len(available_selectors))
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)

        valid = self._valid_tools_for_group(group)

        if not tool_uses:
            fallback = "get_page_html"
            logger.warning("ToolSpecialistAgent(%s): no tool call — fallback to '%s'", group, fallback)
            return fallback, {}, "Fallback"

        inp = tool_uses[0].get("input", {})
        raw_name = inp.get("tool_name", "")

        # Validate + fuzzy-correct within the group's tool list
        if raw_name not in valid:
            import difflib
            matches = difflib.get_close_matches(raw_name, valid, n=1, cutoff=0.4)
            if matches:
                logger.warning("ToolSpecialistAgent: '%s' → corrected to '%s'", raw_name, matches[0])
                raw_name = matches[0]
            else:
                raw_name = next(iter(valid), "get_page_html")
                logger.warning("ToolSpecialistAgent: no match for '%s' — fallback to '%s'", inp.get("tool_name"), raw_name)

        return raw_name, inp.get("tool_args", {}), inp.get("reasoning", "")



# ===========================================================================
# Agent 3 — ToolExecutorAgent  (no LLM — pure execution)
# ===========================================================================

class ToolExecutorAgent:
    """
    Executes a Playwright tool by name with given args.
    No LLM call — just runs the function and returns the observation.
    Kept as a separate agent class so the pipeline is symmetric and each
    stage can be individually logged, mocked, or replaced.
    """

    async def execute(self, tool_name: str, tool_args: dict) -> str:
        """
        Execute a tool and return the JSON observation string.
        Never raises — errors are returned as JSON {"error": "..."}.
        """
        observation = await execute_tool_call(tool_name, tool_args)
        logger.info("Executed %s → %s", tool_name, observation[:120])
        return observation


# ===========================================================================
# Agent 4 — EvaluationAgent
# ===========================================================================

class EvaluationAgent:
    """
    Reads all observations collected so far for the current step and decides:
      "done"   — the step goal has been fully achieved
      "retry"  — not done yet, the executor should try another tool
      "failed" — cannot proceed (error, impossible, loop detected)

    Uses a forced tool call evaluate_step(status, reason, answer).
    """

    _EVAL_TOOL = {
        "toolSpec": {
            "name": "evaluate_step",
            "description": (
                "Evaluate whether the current step goal has been achieved "
                "based on the observations collected so far."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": (
                                "'done' if the step goal is fully achieved, "
                                "'retry' if more tool calls are needed, "
                                "'failed' if it is impossible to complete."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "Explanation for your verdict.",
                        },
                        "answer": {
                            "type": "string",
                            "description": (
                                "Concise result summary for this step. "
                                "Required when status is 'done'."
                            ),
                        },
                    },
                    "required": ["status", "reason", "answer"],
                }
            },
        }
    }

    _SYSTEM = (
        "You are a step evaluation agent for a browser automation system. "
        "You receive a step goal, the LIVE PAGE STATE (ground truth), and tool call observations. "
        "Decide: done (goal achieved), retry (more work needed), failed (impossible).\n"
        "RULES:\n"
        "1. A tool returning 'status: ok' means the CALL executed, NOT that the goal was reached.\n"
        "2. For NAVIGATION steps: the URL in LIVE PAGE STATE must reflect the destination. "
        "   If the title contains '404', 'Not Found', or 'Page not found', the navigation FAILED — "
        "   the URL does not exist. Return retry and suggest navigating to a correct URL.\n"
        "3. A URL the agent constructed from memory is NOT evidence the page exists. "
        "   Verify the title is not an error page before returning done.\n"
        "4. For IN-PAGE steps (expand menu, open dialog): URL will NOT change — "
        "   judge by whether new elements appeared.\n"
        "5. Never return failed because a tool was repeated — repetition is detected mechanically.\n"
        "6. When retrying: state exactly what is missing AND suggest a concrete next action.\n"
        "Always call evaluate_step."
    )

    def __init__(self, llm: BedrockClient):
        self.llm = llm

    async def evaluate(
        self,
        step_description: str,
        observations: list[str],
        tool_calls: list[dict],
    ) -> tuple[LoopStatus, str, str]:
        """
        Evaluate step progress.

        Args:
            step_description: What this step needs to accomplish
            observations: All tool outputs collected for this step so far
            tool_calls: Corresponding tool name + args for each observation

        Returns:
            (LoopStatus, reason, answer)
        """
        # ── Hard loop detection (no LLM needed) ────────────────────────────
        # Count across the whole step, not just the last 3 — interleaved
        # repetition slips through a consecutive-only check.
        if len(tool_calls) >= 3:
            from collections import Counter
            sig_counts = Counter(
                f"{c['tool']}:{json.dumps(c['args'], sort_keys=True)}"
                for c in tool_calls
            )
            worst, n = sig_counts.most_common(1)[0]
            if n >= 3:
                return (
                    LoopStatus.FAILED,
                    f"'{worst.split(':')[0]}' attempted {n} times across this step with no progress.",
                    "",
                )

        # ── Selector error storm ──────────────────────────────────────────
        if len(observations) >= 3:
            if sum(1 for o in observations[-3:] if "SELECTOR ERROR" in o) == 3:
                return (
                    LoopStatus.FAILED,
                    "3 consecutive selector validation failures — cannot find a valid selector. "
                    "Try navigating directly to the target URL instead.",
                    "",
                )

        obs_lines = []
        for i, (obs, call) in enumerate(zip(observations, tool_calls)):
            try:
                parsed = json.loads(obs)
                display = {k: v for k, v in parsed.items() if k not in ("html", "page_html")}
                body = json.dumps(display)[:600]
            except Exception:
                body = obs[:600]
            obs_lines.append(
                f"  Call {i+1}: {call['tool']}({json.dumps(call['args'])[:100]})\n"
                f"  Result: {body}"
            )

        # Settle the page before reading live state — evaluate_js navigation
        # may not have completed by the time we read the URL.
        live_state = ""
        try:
            if pt._state.get("page"):
                p = pt._state["page"]
                try:
                    await p.wait_for_load_state("domcontentloaded", timeout=3000)
                except Exception:
                    pass
                title = await p.title()
                live_state = (
                    f"\nLIVE PAGE STATE (ground truth from the browser — judge against this):\n"
                    f"  URL:   {p.url}\n"
                    f"  Title: {title}\n"
                )
        except Exception:
            pass

        prompt = (
            f"STEP GOAL:\n{step_description}\n"
            f"{live_state}\n"
            f"TOOL CALLS AND OBSERVATIONS:\n" + "\n\n".join(obs_lines) + "\n\n"
            "Has this step goal been fully achieved? "
            "Judge against LIVE PAGE STATE, not against tool call status."
        )

        response = await self.llm.converse(
            messages=[_user_message(prompt)],
            system=self._SYSTEM,
            tools=[self._EVAL_TOOL],
        )
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)

        if not tool_uses:
            # Fallback: assume retry if evaluator doesn't respond
            return LoopStatus.RETRY, "Evaluator gave no verdict — retrying", ""

        inp = tool_uses[0].get("input", {})
        raw_status = inp.get("status", "retry").lower()
        reason = inp.get("reason", "")
        answer = inp.get("answer", "")

        try:
            status = LoopStatus(raw_status)
        except ValueError:
            status = LoopStatus.RETRY

        return status, reason, answer


# ===========================================================================
# OrchestratorAgent — drives the full 4-agent pipeline
# ===========================================================================

class OrchestratorAgent:
    """
    Coordinates all agents for every task.

    Flow:
      1. DecompositionAgent       → plan stored in TaskState
      2. For each PlanStep:
           while not done and retries < max_retries_per_step:
             a. ToolRouterAgent     → picks tool GROUP   (~8 choices)
             b. ToolSpecialistAgent → picks exact tool + args  (~3-9 choices)
             c. ToolExecutorAgent   → executes (no LLM)
             d. EvaluationAgent     → done / retry / failed
      3. _build_summary()          → final LLM call
    """

    _SUMMARY_SYSTEM = (
        "You are a summarisation agent. Given the original task and the results "
        "of each completed step, write a clear, concise final answer."
    )

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries_per_step: int = 8,
        on_event: Optional[Callable[[OrchestratorEvent], Any]] = None,
    ):
        """
        Args:
            model_id: Bedrock model ID
            region: AWS region
            profile: AWS CLI profile name
            max_tokens: Max response tokens per LLM call
            temperature: Sampling temperature
            max_retries_per_step: How many Select→Execute→Evaluate loops per step
            on_event: Callback(OrchestratorEvent) for live display in main.py
        """
        self._llm = BedrockClient(
            model_id=model_id,
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._playwright_tools = get_all_tool_schemas()
        self._max_retries = max_retries_per_step
        self.on_event = on_event

        # Agents (shared LLM client)
        self._decomposer  = DecompositionAgent(self._llm)
        self._router      = ToolRouterAgent(self._llm)
        self._specialist  = ToolSpecialistAgent(self._llm, self._playwright_tools)
        self._executor    = ToolExecutorAgent()
        self._evaluator   = EvaluationAgent(self._llm)

    # ── event helper ────────────────────────────────────────────────────────

    async def _fire(self, event: OrchestratorEvent) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def _task_already_complete(self, original_task: str, state: TaskState) -> tuple[bool, str]:
        """
        After a step completes, quickly check whether the original task is already
        fully answered by the accumulated step results — if so, skip remaining steps.
        Uses a single small LLM call with forced yes/no answer.
        """
        done_steps = state.completed_steps
        if not done_steps:
            return False, ""

        results_summary = "\n".join(
            f"  Step {s.index}: {s.description}\n  Result: {s.result[:300]}"
            for s in done_steps
        )

        _TOOL = {
            "toolSpec": {
                "name": "task_verdict",
                "description": "Report whether the original task is fully answered.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "complete": {"type": "boolean",
                                         "description": "True if the original task is fully answered."},
                            "answer":   {"type": "string",
                                         "description": "The final answer if complete=true, else empty."},
                        },
                        "required": ["complete", "answer"],
                    }
                },
            }
        }

        prompt = (
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"COMPLETED STEPS AND RESULTS:\n{results_summary}\n\n"
            "Is the original task FULLY answered by these results? "
            "Call task_verdict with complete=true only if every piece of "
            "information the task asked for is present."
        )

        try:
            response = await self._llm.converse(
                messages=[_user_message(prompt)],
                system=(
                    "You are a task completion checker. "
                    "Return complete=true only if the task is fully and concretely answered. "
                    "Always call task_verdict."
                ),
                tools=[_TOOL],
            )
            blocks = response["output"]["message"].get("content", [])
            tool_uses = _extract_tool_uses(blocks)
            if tool_uses:
                inp = tool_uses[0].get("input", {})
                if inp.get("complete"):
                    return True, inp.get("answer", "")
        except Exception as e:
            logger.debug("_task_already_complete check failed: %s", e)

        return False, ""

    def _correct_selector(
        self, tool_name: str, tool_args: dict, observations: list[str]
    ) -> dict:
        """
        If the specialist returned a selector that isn't on the live page,
        find the closest real selector from the observations and substitute it.
        This is a pure string-matching correction — no LLM call.
        """
        _SELECTOR_PARAM_TOOLS = {
            "fill", "type_text", "press_key", "click", "hover", "focus",
            "select_option", "check_checkbox", "uncheck_checkbox",
            "get_text", "get_attribute", "get_input_value",
            "wait_for_selector", "assert_visible", "assert_text",
        }
        if tool_name not in _SELECTOR_PARAM_TOOLS:
            return tool_args

        chosen = tool_args.get("selector", "")
        if not chosen:
            return tool_args

        # Collect all real selectors from observations
        real_selectors: list[str] = []
        for o in observations:
            try:
                parsed = json.loads(o)
                for el in parsed.get("elements", []):
                    s = el.get("selector", "")
                    if s:
                        real_selectors.append(s)
                # Also pull from error message selectors
                error_text = parsed.get("error", "")
                if "SELECTOR ERROR" in error_text:
                    for line in error_text.splitlines():
                        stripped = line.strip()
                        if stripped.startswith(("input", "button", "a[", "textarea",
                                                "select", "a#", "button#")):
                            s = stripped.split("  (")[0].strip()
                            if s:
                                real_selectors.append(s)
            except (json.JSONDecodeError, TypeError):
                pass

        if not real_selectors or chosen in real_selectors:
            return tool_args  # selector is valid or nothing to compare against

        # Try to find the best matching real selector
        import difflib
        matches = difflib.get_close_matches(chosen, real_selectors, n=1, cutoff=0.3)
        if matches:
            corrected = matches[0]
            if corrected != chosen:
                logger.warning(
                    "Selector auto-corrected: '%s' → '%s'", chosen, corrected
                )
                new_args = dict(tool_args)
                new_args["selector"] = corrected
                return new_args

        return tool_args

    # ── main entry point ────────────────────────────────────────────────────

    async def run(self, task: str) -> str:
        """
        Execute the full 4-agent pipeline for a task.

        Args:
            task: High-level task in plain English

        Returns:
            Final answer string.
        """
        state = TaskState(original_task=task)

        # ── Phase 1: Decompose ───────────────────────────────────────────
        steps, decomp_reasoning = await self._decomposer.decompose(task)
        logger.info("Decomposition reasoning: %s", decomp_reasoning)

        state.plan = [
            PlanStep(index=i + 1, description=desc)
            for i, desc in enumerate(steps)
        ]

        await self._fire(OrchestratorEvent(
            event_type="plan",
            state=state,
        ))

        # ── Phase 2: Start browser once for the whole task ────────────────
        browser_start_obs = await execute_tool_call("start_browser", {})
        logger.info("Browser started: %s", browser_start_obs)
        browser_result = json.loads(browser_start_obs)
        if "error" in browser_result:
            raise RuntimeError(f"Failed to start browser: {browser_result['error']}")
        # ── Phase 3: Execute each step ───────────────────────────────────────
        for i, plan_step in enumerate(state.plan):
            state.current_step_index = i
            plan_step.status = StepStatus.RUNNING

            await self._fire(OrchestratorEvent(
                event_type="step_start",
                state=state,
                step=plan_step,
            ))

            step_done = False
            last_feedback = ""  # evaluator's retry reason threaded to next iteration

            for loop_num in range(1, self._max_retries + 1):

                # Re-fetch page state every iteration so specialist is never blind
                page_obs = None
                if pt._state.get("page"):
                    try:
                        ph   = await pt.get_page_html()
                        snap = await pt.get_page_snapshot()
                        page_obs = json.dumps({
                            "page_state": True,
                            "url":      ph["url"],
                            "title":    ph["title"],
                            "html":     ph["html"],
                            "elements": snap.get("elements", []),
                        })
                    except Exception:
                        pass

                context = state.context_for_step()
                # Always prepend fresh page state; append prior step observations after
                observations = ([page_obs] if page_obs else []) + plan_step.all_observations
                tool_calls   = plan_step.all_tool_calls

                # ── Agent 2a: Route to group ─────────────────────────────
                group, route_reasoning = await self._router.route(
                    plan_step.description, context, observations
                )

                # ── Agent 2b: Pick exact tool in group ──────────────────
                tool_name, tool_args, sel_reasoning = await self._specialist.select(
                    group, plan_step.description, context, observations,
                    feedback=last_feedback,
                    tried=plan_step.all_tool_calls,
                )
                sel_reasoning = f"[{group}] {sel_reasoning}"

                # Build a partial iteration to carry state through the loop
                iteration = LoopIteration(
                    number=loop_num,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    selection_reasoning=sel_reasoning,
                    observation="",
                )

                await self._fire(OrchestratorEvent(
                    event_type="loop_select",
                    state=state,
                    step=plan_step,
                    iteration=iteration,
                ))

                # ── Agent 3: Execute ─────────────────────────────────────
                observation = await self._executor.execute(tool_name, tool_args)
                iteration.observation = observation

                await self._fire(OrchestratorEvent(
                    event_type="loop_execute",
                    state=state,
                    step=plan_step,
                    iteration=iteration,
                ))

                # Add to step history so evaluator sees everything
                plan_step.iterations.append(iteration)

                # ── Agent 4: Evaluate ────────────────────────────────────
                eval_status, eval_reason, eval_answer = await self._evaluator.evaluate(
                    plan_step.description,
                    plan_step.all_observations,
                    plan_step.all_tool_calls,
                )

                iteration.eval_status = eval_status
                iteration.eval_reason = eval_reason
                iteration.eval_answer = eval_answer

                await self._fire(OrchestratorEvent(
                    event_type="loop_eval",
                    state=state,
                    step=plan_step,
                    iteration=iteration,
                ))

                if eval_status == LoopStatus.DONE:
                    plan_step.result = eval_answer
                    plan_step.status = StepStatus.DONE
                    step_done = True
                    last_feedback = ""
                    await self._fire(OrchestratorEvent(
                        event_type="step_done",
                        state=state,
                        step=plan_step,
                    ))

                    # ── Early exit: check if the original task is already answered ──
                    if i < len(state.plan) - 1:  # only if there are remaining steps
                        complete, early_answer = await self._task_already_complete(
                            state.original_task, state
                        )
                        if complete:
                            logger.info("Task complete after step %d — skipping %d remaining steps",
                                        plan_step.index, len(state.plan) - i - 1)
                            # Mark remaining steps as skipped
                            for remaining in state.plan[i + 1:]:
                                remaining.status = StepStatus.SKIPPED
                                remaining.result = "Skipped — task already complete."
                            # Store the early answer and jump to summary
                            if early_answer:
                                plan_step.result = early_answer
                            await self._fire(OrchestratorEvent(
                                event_type="done",
                                state=state,
                                final_answer=early_answer or state.context_summary(),
                            ))
                            return early_answer or await self._build_summary(state)
                    break

                if eval_status == LoopStatus.FAILED:
                    plan_step.error = eval_reason
                    plan_step.status = StepStatus.FAILED
                    last_feedback = ""
                    await self._fire(OrchestratorEvent(
                        event_type="step_failed",
                        state=state,
                        step=plan_step,
                    ))
                    break

                # RETRY — thread feedback to next iteration
                last_feedback = eval_reason

            if not step_done and plan_step.status != StepStatus.FAILED:
                # Exhausted retries without a DONE verdict
                plan_step.status = StepStatus.FAILED
                plan_step.error = f"Max retries ({self._max_retries}) reached without completion."
                await self._fire(OrchestratorEvent(
                    event_type="step_failed",
                    state=state,
                    step=plan_step,
                ))

        # ── Phase 4: Stop browser ─────────────────────────────────────────
        if pt._state["browser"] is not None:
            await pt.stop_browser()

        # ── Phase 5: Final summary ───────────────────────────────────────
        final_answer = await self._build_summary(state)

        await self._fire(OrchestratorEvent(
            event_type="done",
            state=state,
            final_answer=final_answer,
        ))

        return final_answer

    async def _build_summary(self, state: TaskState) -> str:
        """One final LLM call to synthesise all step results into an answer."""
        lines = [f"Original task:\n{state.original_task}\n\nStep results:"]
        for s in state.plan:
            lines.append(f"\nStep {s.index}: {s.description}")
            if s.status == StepStatus.DONE:
                lines.append(f"  Result: {s.result}")
            else:
                lines.append(f"  FAILED: {s.error}")

        response = await self._llm.converse(
            messages=[_user_message("\n".join(lines))],
            system=self._SUMMARY_SYSTEM,
        )
        blocks = response["output"]["message"].get("content", [])
        return _extract_text(blocks)


# ===========================================================================
# Sync convenience wrapper
# ===========================================================================

def run_orchestrated(
    task: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    region: str = "us-east-1",
    profile: Optional[str] = None,
) -> str:
    """
    Synchronous entry point for the full 4-agent orchestrated run.

    Example:
        from orchestrator import run_orchestrated
        print(run_orchestrated("Go to HN and return the top 3 story titles."))
    """
    async def _run():
        agent = OrchestratorAgent(model_id=model_id, region=region, profile=profile)
        return await agent.run(task)

    return asyncio.run(_run())
