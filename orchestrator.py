"""
Multi-Agent Orchestrator — 2-Role Pipeline
--------------------------------------------
Every task runs through two kinds of specialised agents:

  Orchestrator role — PlanningAgent
    Input : big task (str) + finished-steps-so-far + live page state
    Output: ordered list of REMAINING subtasks, EACH TAGGED WITH ITS TOOL
            GROUP, stored in TaskState.plan (called once upfront, then
            again after every finished step). Routing is decided here,
            once per step — not re-decided on every retry.

  Specialist role — SpecialistAgent     (one "flavor" per tool group)
    Input : subtask goal + context + all observations/tool-calls so far
            for this step + LIVE PAGE STATE (ground truth)
    Output: ONE call does BOTH jobs:
              1. Evaluate the previous tool result (if any) against the
                 step goal → "done" | "retry" | "failed"
              2. If not done/failed, choose the next tool + args from its
                 group's tool list (full docstrings, not just names)
            No separate router or evaluator call — self-contained.

  ToolExecutorAgent                      (per loop iteration, no LLM)
    Input : tool_name + tool_args
    Output: raw tool observation (JSON str)

  Loop per subtask:
    ┌───────────────────────────────────────────────────────────┐
    │  while True:                                                │
    │    result = SpecialistAgent.act(...)  # evaluates prev +    │
    │                                        # picks next tool    │
    │    if result.status == "done"   → store result, next step  │
    │    if result.status == "failed" → mark step failed         │
    │    else → execute chosen tool, append observation, loop     │
    │    (hard, non-LLM loop/error-storm detection runs after      │
    │     every execution as a safety net the LLM can't argue      │
    │     past)                                                    │
    └───────────────────────────────────────────────────────────┘

  OrchestratorAgent drives everything, fires the same event stream
  (plan / step_start / loop_select / loop_execute / loop_eval /
  step_done / step_failed / done) so main.py's rendering is unchanged.
"""

import asyncio
import difflib
import inspect
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import selector_cache as sc
from llm_agent import (
    BedrockClient,
    _user_message,
    _tool_result_message,
    _extract_text,
    _extract_tool_uses,
    execute_tool_call,
    get_all_tool_schemas,
    reset_loop_detection,
    empty_usage,
    add_usage,
    usage_from_response,
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
    screenshot: Optional[bytes] = None  # viewport JPEG after this tool ran — UI-only, never sent to the LLM
    usage: dict = field(default_factory=empty_usage)  # tokens for the SpecialistAgent call behind this iteration


@dataclass
class PlanStep:
    """One subtask in the decomposed plan."""
    index: int                   # 1-based
    description: str
    group: str = ""               # tool group assigned by PlanningAgent
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
      "plan"         — planning done, remaining steps (with groups) ready
      "step_start"   — about to begin a subtask
      "loop_select"  — SpecialistAgent chose the next tool
      "loop_execute" — tool executed, observation received
      "loop_eval"    — SpecialistAgent's evaluation of the previous result
      "step_done"    — subtask completed successfully
      "step_failed"  — subtask failed (all retries exhausted)
      "done"         — all subtasks done, final answer ready

    usage/cumulative_usage carry token counts ({"input_tokens","output_tokens",
    "total_tokens"}) for events backed by an actual LLM call ("plan",
    "loop_select"/"loop_eval" [same call], "done"); zeroed for purely
    mechanical events ("step_start", "loop_execute", "step_done", "step_failed").
    """
    event_type: str
    state: TaskState
    step: Optional[PlanStep] = None
    iteration: Optional[LoopIteration] = None
    final_answer: str = ""
    usage: dict = field(default_factory=empty_usage)             # tokens for the LLM call behind THIS event
    cumulative_usage: dict = field(default_factory=empty_usage)  # running total across the whole run so far


# ===========================================================================
# Agent 1 — PlanningAgent
# ===========================================================================

class PlanningAgent:
    """
    Produces (and repeatedly revises) the REMAINING plan for a task, AND
    assigns each step the tool group that will handle it — routing is
    decided here, once per step, not re-decided on every retry inside the
    step (the group needed for a subtask essentially never changes across
    retries of that SAME subtask; when it does — e.g. an unexpected cookie
    banner — the step ends as failed/blocked and gets replanned into a new
    step with the right group, matching the "replan after every step"
    design rather than adding a mid-step group-switch escape hatch).

    Called once before the first step (with no finished steps yet), and
    again after EVERY step reaches DONE or FAILED (with the finished steps'
    results and the live page state). Returns an ordered list of remaining
    (step description, tool group) pairs — the LLM decides whether that's
    one step at a time (just-in-time) or several (a full plan); an EMPTY
    list means the task is already fully answered by the finished steps.
    """

    _TOOL = {
        "toolSpec": {
            "name": "submit_plan",
            "description": (
                "Submit the ordered list of REMAINING steps needed to finish the task, "
                "each tagged with the tool group that will handle it. Each step must be "
                "a single concrete, self-contained browser action that a Playwright "
                "automation specialist can execute independently. Return an EMPTY steps "
                "list if the task is already fully answered."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "description": (
                                "Ordered list of remaining steps. "
                                "Empty if the task is already fully answered."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {
                                        "type": "string",
                                        "description": "The concrete step to perform.",
                                    },
                                    "group": {
                                        "type": "string",
                                        "description": "Exact tool group name that handles this step.",
                                    },
                                },
                                "required": ["description", "group"],
                            },
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this plan (or why the task is already complete).",
                        },
                    },
                    "required": ["steps", "reasoning"],
                }
            },
        }
    }

    _SYSTEM_TEMPLATE = (
        "You are a task planning agent for a browser automation system. "
        "You are called repeatedly as the task progresses: once at the start, then again "
        "after every step finishes. Decide the REMAINING steps needed to finish the task, "
        "and which TOOL GROUP handles each one.\n\n"
        "AVAILABLE TOOL GROUPS:\n{group_list}\n\n"
        "You may OPTIONALLY call search_elements(terms) first to confirm whether something "
        "specific already exists on the LIVE PAGE (e.g. a login button, cookie banner, search "
        "box) instead of guessing from the sampled elements alone — do this at most twice, "
        "then you MUST call submit_plan. If search_elements returns a selector for a step's "
        "target element, put that selector directly in the step's description (e.g. \"Click "
        "the login link (selector: a[aria-label='Log in'])\") so the specialist can use it "
        "immediately instead of searching again.\n\n"
        "RULES:\n"
        "- Each step must have a single observable success condition — something "
        "  verifiable in the page URL, title, or visible text.\n"
        "- A step that involves entering data AND submitting must be split into two steps.\n"
        "- Never create a step whose only purpose is waiting.\n"
        "- Do not create separate steps for finding a page and going to it — that is one step. "
        "  If a destination has a known URL, prefer navigating to it directly.\n"
        "- Do not duplicate steps with overlapping goals.\n"
        "- If FINISHED STEPS already fully answer the ORIGINAL TASK, return an EMPTY steps list.\n"
        "- If LIVE PAGE STATE reveals something unexpected (cookie banner, login wall, error "
        "  page), insert a step in the right group to handle it before continuing.\n"
        "- Keep remaining steps that are still valid — do not needlessly rewrite steps that "
        "  do not need to change.\n"
        "- The browser is already open. Never assign a step to a group for starting/stopping it.\n"
        "- Assign EVERY step the single best-matching group from the list above.\n"
        "Always call submit_plan."
    )

    def __init__(self, llm: BedrockClient):
        self.llm = llm
        self._search_tool = next(
            (s for s in get_all_tool_schemas() if s["toolSpec"]["name"] == "search_elements"),
            None,
        )

    def _build_group_list(self, available: set[str]) -> str:
        lines = []
        for name in available:
            grp = TOOL_GROUPS[name]
            lines.append(f"  {name} — {grp['description']}")
        return "\n".join(lines)

    async def plan_next(
        self,
        original_task: str,
        finished_steps: list["PlanStep"],
        live_state: str,
        max_search_rounds: int = 2,
    ) -> tuple[list[tuple[str, str]], str, dict]:
        """
        Returns (remaining_steps, reasoning, usage) where remaining_steps is a
        list of (description, group) pairs — group is validated against
        TOOL_GROUPS and corrected/defaulted if hallucinated. `usage` is the
        summed token usage of every converse() call made during this
        plan_next() invocation (a search round can trigger more than one).

        Before committing to a plan, the model may call search_elements(terms)
        up to max_search_rounds times to confirm what's actually on the LIVE
        PAGE (e.g. does a login wall really exist) instead of guessing from the
        live_state summary alone. search_elements is read-only, so this is safe
        to allow during planning without blurring planning vs. execution —
        unlike exposing an action tool (click/fill/...) would be.

        Fallback if the model doesn't call the tool: on the very first call
        (no finished steps yet) falls back to a single step (whole task,
        'navigation' group) rather than risk looping forever on a broken
        planner response; on later calls falls back to [] (assume complete).
        """
        if finished_steps:
            finished_summary = "\n".join(
                f"  Step {s.index}: {s.description}\n"
                f"  Status: {s.status.value}\n"
                f"  Result: {(s.result or s.error)[:400]}"
                for s in finished_steps
            )
        else:
            finished_summary = "  (none yet — this is the initial plan)"

        try:
            page = pt._state.get("page")
            has_page = bool(page and page.url not in ("about:blank", "", None))
        except Exception:
            has_page = False
        available = _POST_NAV_GROUPS if has_page else _PRE_NAV_GROUPS
        default_group = "discovery" if has_page else "navigation"

        prompt = (
            f"ORIGINAL TASK:\n{original_task}\n\n"
            f"FINISHED STEPS:\n{finished_summary}\n\n"
            f"LIVE PAGE STATE:\n{live_state or '(browser not yet navigated)'}\n\n"
            "What are the REMAINING steps needed? Return an empty list if already done."
        )
        system = self._SYSTEM_TEMPLATE.format(group_list=self._build_group_list(available))

        messages = [_user_message(prompt)]
        # search_elements only makes sense once a page actually exists to search
        can_search = has_page and self._search_tool is not None
        usage = empty_usage()

        for round_num in range(max_search_rounds + 1):
            tools = [self._TOOL]
            if can_search and round_num < max_search_rounds:
                tools.append(self._search_tool)

            response = await self.llm.converse(messages=messages, system=system, tools=tools)
            usage = add_usage(usage, usage_from_response(response))
            blocks = response["output"]["message"].get("content", [])
            tool_uses = _extract_tool_uses(blocks)

            if not tool_uses:
                if not finished_steps:
                    logger.warning("PlanningAgent: no tool call on initial plan — single-step fallback")
                    return [(original_task, default_group)], "Single step fallback.", usage
                logger.warning("PlanningAgent: no tool call on replan — assuming task complete")
                return [], "No tool call from planner; assuming complete.", usage

            submit_call = next((t for t in tool_uses if t.get("name") == "submit_plan"), None)

            if submit_call is None:
                search_call = next((t for t in tool_uses if t.get("name") == "search_elements"), None)
                if search_call is not None:
                    logger.info("PlanningAgent: searching screen before planning — terms=%s",
                                search_call.get("input", {}).get("terms"))
                    result_json = await execute_tool_call("search_elements", search_call.get("input", {}))
                    messages.append(response["output"]["message"])
                    messages.append(_tool_result_message(search_call["toolUseId"], result_json))
                    continue
                # Unrecognized tool call — treat like no tool call
                logger.warning("PlanningAgent: unrecognized tool call — fallback")
                if not finished_steps:
                    return [(original_task, default_group)], "Single step fallback.", usage
                return [], "Unrecognized tool call from planner; assuming complete.", usage

            inp = submit_call.get("input", {})
            reasoning = inp.get("reasoning", "")

            steps: list[tuple[str, str]] = []
            for raw in inp.get("steps", []):
                desc = str(raw.get("description", "")).strip()
                if not desc:
                    continue
                group = raw.get("group", "")
                if group not in TOOL_GROUPS:
                    matches = difflib.get_close_matches(group, available, n=1, cutoff=0.4)
                    corrected = matches[0] if matches else default_group
                    logger.warning("PlanningAgent: corrected group '%s' → '%s'", group, corrected)
                    group = corrected
                steps.append((desc, group))

            return steps, reasoning, usage

        # Exhausted search rounds without ever getting a submit_plan call
        logger.warning("PlanningAgent: exhausted %d search rounds without a plan — fallback",
                       max_search_rounds)
        if not finished_steps:
            return [(original_task, default_group)], "Exhausted search rounds fallback.", usage
        return [], "Exhausted search rounds; assuming complete.", usage


# ===========================================================================
# Tool group definitions — each specialist agent sees only its group
# ===========================================================================

TOOL_GROUPS: dict[str, dict] = {
    "navigation": {
        "description": "Go to URLs, go back/forward, reload the page",
        "tools": ["navigate", "go_back", "go_forward", "reload", "get_current_url"],
    },
    "discovery": {
        "description": (
            "Find AND read anything on the page that isn't a form/click action: locate an "
            "element by guessed keywords, inspect one candidate in detail, or read page "
            "structure/HTML/text/attributes — including extracting visible data like counts, "
            "prices, or dates. Use search_elements FIRST when you don't already have a selector; "
            "reach for the page/text-reading tools when you need the actual content, not just "
            "a selector. A single step needing both (e.g. 'find and read the view count') stays "
            "in this ONE group — never split across two."
        ),
        # get_page_html/get_page_snapshot are deliberately absent: get_page_structure
        # is now the single page representation (see playwright_tools._page_view).
        "tools": ["search_elements", "expand_element",
                  "get_page_structure", "get_text_blocks", "get_text", "get_all_text",
                  "get_attribute", "get_input_value", "get_page_content", "evaluate_js",
                  "evaluate_js_on_element"],
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
_POST_NAV_GROUPS = {"navigation", "discovery", "form_input", "clicking",
                    "waiting", "tabs_frames", "capture", "assertions", "storage", "network"}


# ===========================================================================
# Specialist role — SpecialistAgent  (self-contained: evaluates + selects)
# ===========================================================================

@dataclass
class ActResult:
    """Result of one SpecialistAgent.act() call."""
    status: str             # "done" | "retry" | "failed"
    eval_reason: str = ""
    answer: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    reasoning: str = ""
    usage: dict = field(default_factory=empty_usage)


class SpecialistAgent:
    """
    One specialist "flavor" per tool group (parameterised by `group` at call
    time, same as the pool of TOOL_GROUPS). Replaces the old ToolSpecialistAgent
    + EvaluationAgent split: every call does BOTH jobs in one LLM round trip:

      1. EVALUATE — if there's a previous result for this step, judge whether
         the step goal is now achieved, against LIVE PAGE STATE (ground
         truth), not just whether the tool call itself reported success.
      2. ACT — if not done/failed, pick the next tool + args from this
         group's tool list (shown with FULL docstrings, not truncated).

    Self-grading bias (an agent rubber-stamping its own last action) is
    mitigated by carrying the same skeptical rules the old EvaluationAgent
    had — verbatim — into this prompt (see EVALUATION RULES below), and by
    fetching a FRESH live URL/title each call rather than trusting the tool
    call's own self-reported status.

    Hard, non-LLM loop/error-storm detection is NOT here — see
    _check_hard_stop(), a deterministic safety net the LLM can't argue past.
    """

    _ACT_TOOL = {
        "toolSpec": {
            "name": "act",
            "description": (
                "Evaluate the previous tool result for this step (if any) and choose what "
                "happens next: 'done' (goal fully achieved, provide answer), 'failed' "
                "(impossible, explain why), or 'retry' (provide the next tool_name + tool_args)."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "description": "'done', 'retry', or 'failed'.",
                        },
                        "eval_reason": {
                            "type": "string",
                            "description": "Why this status — what LIVE PAGE STATE / OBSERVATIONS show.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "Concise result summary. Required when status is 'done'.",
                        },
                        "tool_name": {
                            "type": "string",
                            "description": "Exact next tool name. Required when status is 'retry'.",
                        },
                        "tool_args": {
                            "type": "object",
                            "description": "Arguments for tool_name as key-value pairs.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this tool and these args.",
                        },
                    },
                    "required": ["status", "eval_reason"],
                }
            },
        }
    }

    _SYSTEM_TEMPLATE = (
        "You are a specialist tool agent for browser automation, assigned to the '{group}' group. "
        "Every turn you must do BOTH of these, in order:\n"
        "  1. EVALUATE — if OBSERVATIONS below already contains a result from a previous action "
        "     on this step, judge whether the CURRENT STEP GOAL is now fully achieved. Judge "
        "     against LIVE PAGE STATE (the actual URL/title), NOT just whether the tool call "
        "     itself reported success.\n"
        "  2. ACT — if the goal is not yet achieved and is not impossible, choose the SINGLE "
        "     best next tool call from the list below to make progress.\n\n"
        "AVAILABLE TOOLS IN THIS GROUP:\n{tool_list}\n\n"
        "SELECTOR SYNTAX — you may use any of these:\n"
        "  aria-ref=e12                 a ref copied from get_page_structure's outline ← PREFER THIS\n"
        "  #id                          when the element has an id\n"
        "  a[href*='careers']           match part of a URL\n"
        "  button:has-text(\"Company\")   match by visible text  ← use when no id/aria-label/ref\n"
        "  text=Careers                 exact visible text\n"
        "  a >> nth=3                   positional, last resort\n"
        "Never invent a URL. Only navigate to hrefs you have READ from the page HTML.\n\n"
        "EVALUATION RULES:\n"
        "1. A tool returning 'status: ok' means the CALL executed, NOT that the goal was reached.\n"
        "2. For NAVIGATION steps: the URL in LIVE PAGE STATE must reflect the destination. "
        "   If the title contains '404', 'Not Found', or 'Page not found', the navigation FAILED — "
        "   the URL does not exist. Set status='retry' and pick a corrected next action.\n"
        "3. A URL you constructed from memory is NOT evidence the page exists. Verify the title "
        "   is not an error page before setting status='done'.\n"
        "4. For IN-PAGE changes (menus, dialogs): URL will NOT change — judge by whether new "
        "   elements appeared in OBSERVATIONS.\n"
        "5. Repetition and selector-error storms are caught mechanically — you don't need to set "
        "   status='failed' just because a tool call is being retried.\n"
        "6. If OBSERVATIONS has no result yet for THIS step, set status='retry' and simply choose "
        "   your first tool call — there is nothing to evaluate yet.\n"
        "7. NEVER set status='done' unless the 'answer' text is copied or directly derived from "
        "   content that literally appears in OBSERVATIONS or LIVE PAGE STATE above. Do not "
        "   invent, estimate, or recall a plausible-sounding value from general knowledge (a view "
        "   count, price, date, name, etc. you have not actually seen in this conversation). If "
        "   the needed data is not yet visible, set status='retry' and pick a tool that would "
        "   reveal it instead of guessing.\n\n"
        "TOOL SELECTION RULES:\n"
        "1. Only use tool names exactly as listed above.\n"
        "2. If the STEP GOAL already names a specific selector (e.g. \"(selector: ...)\"), "
        "   use it directly — it was already confirmed by the planner, no need to search again.\n"
        "3. Otherwise, if this group includes get_page_structure, call it FIRST — it's a "
        "   compact outline with visible text AND refs ([ref=e12]) you can copy directly as "
        "   \"aria-ref=e12\". Use search_elements(terms) with a few guessed words (e.g. "
        "   ['login', 'sign in']) to filter it when the page is too large to read whole. Use "
        "   expand_element(selector) to see full detail on one candidate before acting on it.\n"
        "4. Otherwise (or if the outline is missing something, e.g. a raw href) derive selectors "
        "   from the CURRENT PAGE HTML in the prompt — prefer id, aria-label/title, name, text, "
        "   then href for links.\n"
        "5. Never pass a 'timeout' parameter.\n"
        "6. To submit a search form: use press_key(selector, 'Enter') on the input.\n"
        "7. Always call act."
    )

    def __init__(self, llm: BedrockClient, all_tool_schemas: list[dict]):
        self.llm = llm
        self._schema_map = {s["toolSpec"]["name"]: s for s in all_tool_schemas}

    def _tool_list_for_group(self, group: str) -> str:
        """Full docstrings (not truncated) for every tool in this group."""
        tools = TOOL_GROUPS.get(group, {}).get("tools", [])
        sections = []
        for name in tools:
            func = getattr(pt, name, None)
            if func is None:
                continue
            doc = inspect.getdoc(func) or "(no description)"
            sections.append(f"### {name}\n{doc}")
        return "\n\n".join(sections)

    def _valid_tools_for_group(self, group: str) -> set[str]:
        return set(TOOL_GROUPS.get(group, {}).get("tools", []))

    async def act(
        self,
        group: str,
        step_description: str,
        context: str,
        observations: list[str],
        tool_calls: list[dict],
        tried: list[dict] | None = None,
    ) -> ActResult:
        """
        Evaluate the previous observation for this step (if any) and choose
        the next tool call, in ONE LLM call.

        Returns:
            ActResult with status "done" | "retry" | "failed".
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

        tried_section = ""
        if tried:
            sigs = [f"  {c['tool']}({json.dumps(c['args'])[:120]})" for c in (tried or [])]
            tried_section = (
                "\n\nALREADY TRIED THIS STEP — do NOT repeat any of these exactly:\n"
                + "\n".join(sigs)
            )

        # Fresh LIVE PAGE STATE (ground truth) — the anti-self-grading-bias check.
        # Settle the page first — a previous evaluate_js navigation may not have
        # completed by the time we read the URL.
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
                    f"\n\nLIVE PAGE STATE (ground truth — judge against this, not tool status):\n"
                    f"  URL:   {p.url}\n"
                    f"  Title: {title}\n"
                )
        except Exception:
            pass

        tool_list = self._tool_list_for_group(group)
        system = self._SYSTEM_TEMPLATE.format(group=group, tool_list=tool_list)

        prompt = (
            f"STEP GOAL:\n{step_description}\n\n"
            f"CONTEXT:\n{context}"
            f"{obs_section}"
            f"{cache_hints}"
            f"{html_section}"
            f"{selector_section}"
            f"{tried_section}"
            f"{live_state}\n\n"
            f"You are in the '{group}' group. Evaluate any previous result, then decide what happens next."
        )

        response = await self.llm.converse(
            messages=[_user_message(prompt)],
            system=system,
            tools=[self._ACT_TOOL],
        )
        usage = usage_from_response(response)
        logger.info("SPECIALIST(%s) prompt=%d chars | html=%d | selectors=%d",
                    group, len(prompt), len(latest_html), len(available_selectors))
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)

        if not tool_uses:
            logger.warning("SpecialistAgent(%s): no tool call — retry fallback", group)
            return ActResult(status="retry", eval_reason="No tool call from specialist.", usage=usage)

        inp = tool_uses[0].get("input", {})
        raw_status = str(inp.get("status", "retry")).lower()
        if raw_status not in ("done", "retry", "failed"):
            raw_status = "retry"
        eval_reason = inp.get("eval_reason", "")
        answer = inp.get("answer", "")

        # Deterministic anti-hallucination guard: an empty answer on "done" means
        # the model declared success without actually grounding a value in
        # anything observed — force it back to retry rather than accept a
        # silent no-op success (see EVALUATION RULES rule 7 above).
        if raw_status == "done" and not answer.strip():
            logger.warning("SpecialistAgent(%s): 'done' with empty answer — forcing retry", group)
            raw_status = "retry"
            eval_reason = (
                "Rejected: status was 'done' but 'answer' was empty — you must ground the "
                "answer in something actually observed before marking this step complete."
            )

        if raw_status != "retry":
            return ActResult(status=raw_status, eval_reason=eval_reason, answer=answer, usage=usage)

        # status == "retry" -> validate/correct the chosen tool_name
        valid = self._valid_tools_for_group(group)
        raw_name = inp.get("tool_name", "")
        tool_args = inp.get("tool_args", {})
        reasoning = inp.get("reasoning", "")

        if raw_name not in valid:
            # Some models occasionally emit a real tool name followed by
            # garbled pseudo-tool-call syntax inside the SAME string, e.g.
            # 'get_text_blocks>\n<__parameter=tool_args>{"limit": 1}'.
            # Try recovering the leading identifier before fuzzy-matching
            # the whole garbled blob (which rarely matches anything well).
            prefix_match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", raw_name)
            prefix = prefix_match.group(0) if prefix_match else ""

            if prefix in valid:
                logger.warning("SpecialistAgent: recovered '%s' from garbled tool_name '%s'",
                                prefix, raw_name[:80])
                raw_name = prefix
            else:
                matches = difflib.get_close_matches(raw_name, valid, n=1, cutoff=0.4)
                if matches:
                    logger.warning("SpecialistAgent: '%s' → corrected to '%s'", raw_name, matches[0])
                    raw_name = matches[0]
                else:
                    # No confident match at all — tool_args were meant for a
                    # different/unclear tool and may not even be valid kwargs
                    # here, so fall back to a tool with NO required params in
                    # this group (safe to call with no args) and drop them.
                    safe_fallback = next(
                        (name for name in valid
                         if not self._schema_map.get(name, {})
                                .get("toolSpec", {}).get("inputSchema", {})
                                .get("json", {}).get("required")),
                        next(iter(valid)),
                    )
                    logger.warning(
                        "SpecialistAgent: no match for '%s' — safe fallback to '%s()' with no args",
                        inp.get("tool_name"), safe_fallback,
                    )
                    raw_name = safe_fallback
                    tool_args = {}

        return ActResult(
            status="retry", eval_reason=eval_reason,
            tool_name=raw_name, tool_args=tool_args, reasoning=reasoning,
            usage=usage,
        )



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
# Hard, non-LLM safety nets (moved out of the old EvaluationAgent)
# ===========================================================================
# The LLM-judgment part of evaluation (was the previous EvaluationAgent) now
# lives inside SpecialistAgent.act() — it self-evaluates its own last tool
# call, since it has richer per-group context than a generic evaluator did.
# These two checks stay as PLAIN PYTHON, deliberately separate from any LLM
# call, so a model can never talk its way past an actual infinite loop or a
# storm of selector failures.

def _check_hard_stop(tool_calls: list[dict], observations: list[str]) -> tuple[bool, str]:
    """
    Deterministic loop/error-storm detection across the whole step so far.
    Returns (should_fail, reason).
    """
    # ── Hard loop detection ─────────────────────────────────────────────
    # Count across the whole step, not just the last 3 — interleaved
    # repetition slips through a consecutive-only check.
    if len(tool_calls) >= 3:
        sig_counts = Counter(
            f"{c['tool']}:{json.dumps(c['args'], sort_keys=True)}"
            for c in tool_calls
        )
        worst, n = sig_counts.most_common(1)[0]
        if n >= 3:
            return True, f"'{worst.split(':')[0]}' attempted {n} times across this step with no progress."

    # ── Selector error storm ─────────────────────────────────────────────
    if len(observations) >= 3:
        if sum(1 for o in observations[-3:] if "SELECTOR ERROR" in o) == 3:
            return True, (
                "3 consecutive selector validation failures — cannot find a valid selector. "
                "Try navigating directly to the target URL instead."
            )

    return False, ""


# ===========================================================================
# OrchestratorAgent — drives the full 4-agent pipeline
# ===========================================================================

class OrchestratorAgent:
    """
    Coordinates both agent roles for every task.

    Flow:
      1. PlanningAgent  → plan (remaining steps, EACH TAGGED WITH ITS GROUP)
                          stored in TaskState
      2. For each PlanStep:
           while True:
             a. SpecialistAgent.act(group, ...) → evaluates the previous
                tool result for this step (if any) AND, in the SAME call,
                picks the next tool + args — no separate router/evaluator
                LLM call.
             b. if status == done/failed → store result, stop retrying
             c. else → ToolExecutorAgent executes the chosen tool (no LLM),
                then a hard, non-LLM loop/error-storm check runs before
                looping back to (a)
         then PlanningAgent.plan_next() runs again to revise the remaining
         plan (and re-assign groups for any new steps)
      3. _build_summary()  → final LLM call
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
        max_total_steps: int = 25,
        on_event: Optional[Callable[[OrchestratorEvent], Any]] = None,
        reasoning_budget_tokens: Optional[int] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Args:
            model_id: Bedrock model ID
            region: AWS region
            profile: AWS CLI profile name
            max_tokens: Max response tokens per LLM call
            temperature: Sampling temperature
            reasoning_budget_tokens: Enable model reasoning/"thinking" on every
                agent in the pipeline. Provider-neutral — see BedrockClient.
            max_retries_per_step: How many Select→Execute→Evaluate loops per step
            max_total_steps: Safety cap on how many steps will be executed in total.
                             The plan is now regenerated after every step (see
                             PlanningAgent), so unlike a fixed upfront plan there is
                             no natural end unless the planner returns an empty
                             remaining-list — this cap bounds worst-case cost/looping.
            on_event: Callback(OrchestratorEvent) for live display in main.py
            on_delta: Callback(kind, text) fired per chunk as any agent in the
                pipeline produces it, so planning/selection/evaluation text
                appears live rather than only at event boundaries. Shared by
                every agent here since they share one client.
        """
        self._llm = BedrockClient.create(
            model_id=model_id,
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_budget_tokens=reasoning_budget_tokens,
            on_delta=on_delta,
        )
        self._playwright_tools = get_all_tool_schemas()
        self._max_retries = max_retries_per_step
        self._max_total_steps = max_total_steps
        self.on_event = on_event
        self._total_usage: dict = empty_usage()

        # Agents (shared LLM client)
        self._planner     = PlanningAgent(self._llm)
        self._specialist  = SpecialistAgent(self._llm, self._playwright_tools)
        self._executor    = ToolExecutorAgent()

    # ── event helper ────────────────────────────────────────────────────────

    def _track(self, usage: dict) -> dict:
        """Add `usage` to the running total and return the new cumulative total."""
        self._total_usage = add_usage(self._total_usage, usage)
        return self._total_usage

    async def _fire(self, event: OrchestratorEvent) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if asyncio.iscoroutine(result):
            await result

    # ── main entry point ────────────────────────────────────────────────────

    async def _live_state_summary(self) -> str:
        """Compact live page state for the planner — URL/title + a sample of
        indexed elements. Never the full HTML (would defeat the point)."""
        if not pt._state.get("page"):
            return ""
        try:
            page = pt._state["page"]
            summary = f"URL: {page.url}\nTitle: {await page.title()}"
            nodes = await pt._ensure_page_index()
            if nodes:
                sample = [
                    f"  {n.get('kind', '')}: "
                    f"{n.get('aria_label') or n.get('text') or n.get('id') or n.get('tag')}"
                    for n in nodes[:15]
                ]
                summary += "\nVisible elements (sample):\n" + "\n".join(sample)
            return summary
        except Exception:
            return ""

    async def run(self, task: str) -> str:
        """
        Execute the pipeline for a task: plan → (route → select → execute →
        evaluate)* per step → replan from live state → repeat until the
        planner reports nothing remains → summarise.

        Args:
            task: High-level task in plain English

        Returns:
            Final answer string.
        """
        state = TaskState(original_task=task)
        reset_loop_detection()

        # ── Phase 1: Start browser once for the whole task ────────────────
        browser_start_obs = await execute_tool_call("start_browser", {})
        logger.info("Browser started: %s", browser_start_obs)
        browser_result = json.loads(browser_start_obs)
        if "error" in browser_result:
            raise RuntimeError(f"Failed to start browser: {browser_result['error']}")

        # ── Phase 2: Initial plan ──────────────────────────────────────────
        initial_steps, plan_reasoning, plan_usage = await self._planner.plan_next(task, [], "")
        logger.info("Initial plan reasoning: %s", plan_reasoning)
        state.plan = [
            PlanStep(index=i + 1, description=desc, group=group)
            for i, (desc, group) in enumerate(initial_steps)
        ]
        await self._fire(OrchestratorEvent(
            event_type="plan", state=state,
            usage=plan_usage, cumulative_usage=self._track(plan_usage),
        ))

        # ── Phase 3: Execute steps, replanning after each one ──────────────
        i = 0
        while i < len(state.plan) and i < self._max_total_steps:
            plan_step = state.plan[i]
            state.current_step_index = i
            plan_step.status = StepStatus.RUNNING

            await self._fire(OrchestratorEvent(
                event_type="step_start",
                state=state,
                step=plan_step,
                cumulative_usage=self._total_usage,
            ))

            pending_iteration: Optional[LoopIteration] = None  # awaiting evaluation
            loop_num = 0

            while True:

                # Re-fetch page state every iteration so the specialist is never
                # blind. One view (structure), not html + elements — this runs on
                # EVERY loop iteration, so it was the single most repeated payload
                # in the whole pipeline.
                page_obs = None
                if pt._state.get("page"):
                    try:
                        page_obs = json.dumps({"page_state": True, **await pt._page_view()})
                    except Exception:
                        pass

                context = state.context_for_step()
                # Always prepend fresh page state; append prior step observations after
                observations = ([page_obs] if page_obs else []) + plan_step.all_observations

                # ── Specialist: evaluate the previous result (if any) AND
                #    pick the next tool call, in ONE call ───────────────────
                result = await self._specialist.act(
                    group=plan_step.group,
                    step_description=plan_step.description,
                    context=context,
                    observations=observations,
                    tool_calls=plan_step.all_tool_calls,
                    tried=plan_step.all_tool_calls,
                )
                # One LLM call serves BOTH the loop_eval and loop_select events
                # below — count it toward the running total exactly once here.
                specialist_cumulative = self._track(result.usage)

                # Close out whatever iteration was awaiting evaluation
                if pending_iteration is not None:
                    pending_iteration.eval_status = (
                        LoopStatus(result.status) if result.status in ("done", "retry", "failed")
                        else LoopStatus.RETRY
                    )
                    pending_iteration.eval_reason = result.eval_reason
                    pending_iteration.eval_answer = result.answer
                    await self._fire(OrchestratorEvent(
                        event_type="loop_eval",
                        state=state,
                        step=plan_step,
                        iteration=pending_iteration,
                        usage=result.usage,
                        cumulative_usage=specialist_cumulative,
                    ))
                    pending_iteration = None

                if result.status == "done":
                    plan_step.result = result.answer
                    plan_step.status = StepStatus.DONE
                    await self._fire(OrchestratorEvent(
                        event_type="step_done",
                        state=state,
                        step=plan_step,
                        cumulative_usage=self._total_usage,
                    ))
                    break

                if result.status == "failed":
                    plan_step.error = result.eval_reason
                    plan_step.status = StepStatus.FAILED
                    await self._fire(OrchestratorEvent(
                        event_type="step_failed",
                        state=state,
                        step=plan_step,
                        cumulative_usage=self._total_usage,
                    ))
                    break

                # status == "retry" — execute the chosen tool, if still within budget
                loop_num += 1
                if loop_num > self._max_retries:
                    plan_step.status = StepStatus.FAILED
                    plan_step.error = f"Max retries ({self._max_retries}) reached without completion."
                    await self._fire(OrchestratorEvent(
                        event_type="step_failed",
                        state=state,
                        step=plan_step,
                        cumulative_usage=self._total_usage,
                    ))
                    break

                iteration = LoopIteration(
                    number=loop_num,
                    tool_name=result.tool_name,
                    tool_args=result.tool_args,
                    selection_reasoning=f"[{plan_step.group}] {result.reasoning}",
                    observation="",
                    usage=result.usage,
                )
                await self._fire(OrchestratorEvent(
                    event_type="loop_select",
                    state=state,
                    step=plan_step,
                    iteration=iteration,
                    usage=result.usage,
                    cumulative_usage=specialist_cumulative,
                ))

                observation = await self._executor.execute(result.tool_name, result.tool_args)
                iteration.observation = observation
                iteration.screenshot = pt._state.get("last_screenshot")
                await self._fire(OrchestratorEvent(
                    event_type="loop_execute",
                    state=state,
                    step=plan_step,
                    iteration=iteration,
                    cumulative_usage=self._total_usage,
                ))
                plan_step.iterations.append(iteration)

                # Hard, non-LLM safety net — a model can never talk its way past this
                hard_failed, hard_reason = _check_hard_stop(
                    plan_step.all_tool_calls, plan_step.all_observations
                )
                if hard_failed:
                    iteration.eval_status = LoopStatus.FAILED
                    iteration.eval_reason = hard_reason
                    await self._fire(OrchestratorEvent(
                        event_type="loop_eval",
                        state=state,
                        step=plan_step,
                        iteration=iteration,
                        cumulative_usage=self._total_usage,
                    ))
                    plan_step.status = StepStatus.FAILED
                    plan_step.error = hard_reason
                    await self._fire(OrchestratorEvent(
                        event_type="step_failed",
                        state=state,
                        step=plan_step,
                        cumulative_usage=self._total_usage,
                    ))
                    break

                pending_iteration = iteration  # evaluated at the top of the next loop

            # ── Replan: regenerate the remaining plan from live state ──────
            finished_so_far = [
                s for s in state.plan[: i + 1]
                if s.status in (StepStatus.DONE, StepStatus.FAILED)
            ]
            live_state = await self._live_state_summary()
            remaining_steps, replan_reasoning, replan_usage = await self._planner.plan_next(
                task, finished_so_far, live_state
            )
            logger.info("Replan after step %d: %s", plan_step.index, replan_reasoning)

            next_index = plan_step.index + 1
            state.plan = state.plan[: i + 1] + [
                PlanStep(index=next_index + j, description=desc, group=group)
                for j, (desc, group) in enumerate(remaining_steps)
            ]
            await self._fire(OrchestratorEvent(
                event_type="plan", state=state,
                usage=replan_usage, cumulative_usage=self._track(replan_usage),
            ))

            i += 1

        if i >= self._max_total_steps and i < len(state.plan):
            logger.warning("Reached max_total_steps (%d) — stopping with partial results.",
                           self._max_total_steps)

        # ── Phase 4: Stop browser ─────────────────────────────────────────
        if pt._state["browser"] is not None:
            await pt.stop_browser()

        # ── Phase 5: Final summary ───────────────────────────────────────
        final_answer, summary_usage = await self._build_summary(state)

        await self._fire(OrchestratorEvent(
            event_type="done",
            state=state,
            final_answer=final_answer,
            usage=summary_usage,
            cumulative_usage=self._track(summary_usage),
        ))

        return final_answer

    async def _build_summary(self, state: TaskState) -> tuple[str, dict]:
        """One final LLM call to synthesise all step results into an answer.
        Returns (answer, usage) — usage is this one call's token count."""
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
        return _extract_text(blocks), usage_from_response(response)


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
