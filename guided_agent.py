"""
Guided Agent — Thinking + Actor loop (third mode, alongside ReAct and Orchestrator)
------------------------------------------------------------------------------------
A tighter, per-ACTION loop (closer to ReAct's granularity) than OrchestratorAgent's
per-STEP planning, but split into two LLM roles instead of ReAct's single agent:

  ThinkingAgent (stateful — a REAL growing conversation, like ReAct)
    Sees the page only through READ-ONLY tools: search_elements, expand_element,
    get_page_structure (a nested role/name outline — NOT raw HTML, NOT a flat
    element list), get_current_url. Calls these freely to gather information
    (executed directly, no second LLM call), then hands off ONE concrete next
    action via the decide_action tool: a short rolling "current_goal" (for
    live-progress display), a natural-language instruction, and the exact
    target selector. Sets task_complete+final_answer when the whole task is done.

  ActorAgent (stateless — one call per action, like PlanningAgent/SpecialistAgent)
    Never decides WHAT to do — only translates the ThinkingAgent's instruction
    into an exact tool_name + tool_args from the FULL flat tool list (no
    grouping, per explicit design choice — the instruction is already narrow
    enough that routing isn't needed). Always an LLM call, even when the
    mapping looks mechanical (explicit choice — a deterministic fast path
    would be more efficient but wasn't wanted here).

  ToolExecutorAgent (reused from orchestrator.py, no LLM) executes the
  resolved tool call; the observation goes back to ThinkingAgent for its
  next decision.

Trade-off vs. OrchestratorAgent: this is 2 LLM calls per action (thinking +
actor) instead of Orchestrator's 1 (SpecialistAgent.act() self-evaluates AND
selects in one call) — slower/costlier per action, in exchange for a tighter,
continuously-adaptive loop with no upfront multi-step plan to go stale.
"""

import asyncio
import copy
import difflib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from llm_agent import (
    BedrockClient,
    _user_message,
    _extract_tool_uses,
    execute_tool_call,
    get_all_tool_schemas,
    build_tool_schema,
    compact_history,
    reset_loop_detection,
    empty_usage,
    add_usage,
    usage_from_response,
)
from orchestrator import ToolExecutorAgent
import playwright_tools as pt

logger = logging.getLogger(__name__)


def _tool_result_block(tool_use_id: str, text: str, is_error: bool = False) -> dict:
    """One toolResult content block (not wrapped in a message) — used to batch
    multiple resolved tool_uses from a single turn into one user message."""
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [{"text": text}],
            **({"status": "error"} if is_error else {}),
        }
    }


# ===========================================================================
# Data model
# ===========================================================================

@dataclass
class DeliberationConfig:
    """
    How hard to make the ThinkingAgent deliberate before it commits to an action.

    Large reasoning models deliberate natively — they weigh alternatives, predict
    consequences, and notice when a prediction was wrong, all inside one forward
    pass. Small, cheap models (Haiku, Nova Lite) do not do this spontaneously:
    asked for a decision, they emit the first plausible one. Each flag below
    rebuilds one piece of that behaviour in orchestration code instead of hoping
    the model does it unprompted.

    Ordered by cost. The two schema-level layers are free — they change what the
    model must emit, not how often it is called — and are on by default. The two
    multi-call layers multiply the cost of every action and are off by default;
    turn them on for hard tasks or weak models.

        forced_schema       +0 calls/action   (weigh alternatives)
        predict_verify      +0 calls/action   (notice mistakes)
        self_consistency_k  xK calls/action   (converge on a choice)
        critic              +1 call/action    (challenge the choice)
    """

    #: Require decide_action to enumerate >= 2 candidates, each with a case for
    #: and against it and a score, before naming the one it picked. A small model
    #: skips deliberation unless the output schema makes skipping impossible.
    forced_schema: bool = True

    #: Require the model to state what it expects to observe after the action,
    #: then compare that against real page state in pure code and feed back any
    #: mismatch. Surfaces a wrong choice on the very next turn instead of several
    #: steps later, once the trail has gone cold.
    predict_verify: bool = True

    #: Sample the whole decision K times over forked copies of the conversation
    #: and take the majority target. 1 disables it. Only safe because the
    #: ThinkingAgent is read-only — re-running its perception loop cannot touch
    #: the page. Costs roughly K x the decision step.
    self_consistency_k: int = 1

    #: After a decision, have a separate agent argue that the choice is wrong. A
    #: veto sends the objection back for one re-decide. Adds one call per action.
    critic: bool = False

    def __post_init__(self) -> None:
        if self.self_consistency_k < 1:
            raise ValueError(f"self_consistency_k must be >= 1, got {self.self_consistency_k}")


@dataclass
class Decision:
    """One hand-off from ThinkingAgent to ActorAgent, or a final answer."""
    current_goal: str = ""
    instruction: str = ""
    target_selector: str = ""
    task_complete: bool = False
    final_answer: str = ""
    #: Candidates weighed before choosing, when forced_schema is on. Each entry
    #: is {ref, label, supports, against, score}. Empty when the layer is off or
    #: the decision is a task_complete.
    options_considered: list[dict] = field(default_factory=list)
    #: What the model expects to hold after the action, when predict_verify is
    #: on: {url_changes, expect_text, target_should_vanish}.
    expected_outcome: dict = field(default_factory=dict)
    #: Native model reasoning (Bedrock reasoningContent) emitted while reaching
    #: this decision, concatenated across the perception turns. Only populated
    #: on reasoning-capable models with a reasoning budget set — otherwise "".
    model_reasoning: str = ""
    #: Summed token usage of every converse() call behind this decision (every
    #: perception-loop turn, and every self-consistency fork if k > 1).
    usage: dict = field(default_factory=empty_usage)


# ===========================================================================
# Predict -> verify — pure-code checking of the model's stated expectation
# ===========================================================================

async def _capture_page_state() -> dict:
    """URL/title snapshot taken immediately before an action runs."""
    page = pt._state.get("page")
    if page is None:
        return {}
    try:
        return {"url": page.url, "title": await page.title()}
    except Exception:
        return {}


def _parse_observation_payload(observation_text: str) -> dict:
    """Best-effort parse of tool observation JSON payload."""
    if not observation_text:
        return {}
    try:
        payload = json.loads(observation_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _expected_text_variants(expect_text: str) -> set[str]:
    """Small synonym expansion to reduce brittle lexical-only checks."""
    base = (expect_text or "").strip().lower()
    if not base:
        return set()

    variants: set[str] = {base}
    synonym_map: dict[str, list[str]] = {
        "login": ["log in", "sign in", "signin"],
        "sign in": ["login", "log in", "signin"],
        "password": ["passcode", "pwd"],
    }
    variants.update(synonym_map.get(base, []))
    return variants


async def _verify_prediction(
    expected: dict,
    before: dict,
    target_selector: str,
    observation_text: str = "",
) -> str:
    """
    Compare the model's stated expected_outcome against what actually happened.

    Deliberately contains no LLM call: the point is a check the model cannot
    argue its way past, in the same spirit as orchestrator._check_hard_stop.
    Returns "" when every prediction held, otherwise a PREDICTION MISMATCH note
    to append to the observation the ThinkingAgent sees on its next turn.
    """
    page = pt._state.get("page")
    if page is None or not expected or not before:
        return ""

    mismatches: list[str] = []

    if expected.get("url_changes") is not None:
        try:
            changed = page.url != before.get("url")
            if expected["url_changes"] and not changed:
                mismatches.append(
                    f"you expected the URL to change, but it is still {page.url}"
                )
            elif not expected["url_changes"] and changed:
                mismatches.append(
                    f"you expected to stay on {before.get('url')}, but navigated "
                    f"to {page.url}"
                )
        except Exception:
            pass

    observation = _parse_observation_payload(observation_text)

    expect_text = (expected.get("expect_text") or "").strip()
    if expect_text:
        variants = _expected_text_variants(expect_text)

        # First use immediate tool evidence (structure/text/verified value)
        # because it often includes frame content that top-level body text misses.
        evidence_chunks: list[str] = []
        for key in ("structure", "text", "verified_value", "title"):
            value = observation.get(key)
            if isinstance(value, str) and value:
                evidence_chunks.append(value.lower())

        texts = observation.get("texts")
        if isinstance(texts, list):
            evidence_chunks.extend(str(t).lower() for t in texts if t is not None)

        found_in_observation = any(
            variant in chunk
            for variant in variants
            for chunk in evidence_chunks
        )

        if not found_in_observation:
            found_on_page = False

            # Then scan top page + iframes, where many enterprise login flows live.
            for frame in page.frames:
                try:
                    body = await frame.inner_text("body", timeout=2000)
                    lower = body.lower()
                    if any(v in lower for v in variants):
                        found_on_page = True
                        break
                except Exception:
                    continue

            if not found_on_page:
                mismatches.append(
                    f"you expected the text {expect_text!r} to appear, and it is "
                    f"not on the page"
                )

    # aria-ref=eN is scoped to one snapshot generation and is invalidated by any
    # mutation, so "did it vanish?" cannot be answered for it — a stale ref
    # always looks gone regardless of what actually happened to the element.
    # Only CSS selectors are checkable here.
    if expected.get("target_should_vanish") and target_selector \
            and not target_selector.startswith("aria-ref="):
        try:
            still_visible = await page.locator(target_selector).first.is_visible(timeout=1500)
            if still_visible:
                mismatches.append(
                    f"you expected {target_selector!r} to disappear, and it is "
                    f"still visible"
                )
        except Exception:
            pass  # gone, or unresolvable — either way not a contradiction

    if not mismatches:
        return ""
    return (
        "\n\nPREDICTION MISMATCH — the action did not do what you predicted: "
        + "; ".join(mismatches)
        + ". Re-read the page before deciding again; do not simply retry the "
          "same action."
    )


@dataclass
class GuidedEvent:
    """
    Fired by GuidedAgent at every significant moment (mirrors OrchestratorEvent's
    shape so main.py can render this mode with a similar pattern).

    event_type:
      "goal"        — ThinkingAgent produced/updated its rolling current_goal
      "perceive"    — ThinkingAgent called a read-only tool to gather info
      "critique"    — CriticAgent reviewed the decision (passed or vetoed it)
      "act_select"  — ActorAgent resolved a decision to an exact tool call
      "act_execute" — tool executed, observation received
      "done"        — task complete, final answer ready
      "failed"      — max_iterations reached without completing

    usage/cumulative_usage carry token counts ({"input_tokens","output_tokens",
    "total_tokens"}) for events backed by an actual LLM call ("goal" [the
    ThinkingAgent decision], "critique", "act_select"); zeroed for purely
    mechanical events ("perceive" carries zero usage of its own since perception
    tools aren't LLM calls, but still reports the running cumulative_usage).
    """
    event_type: str
    iteration: int = 0
    current_goal: str = ""
    perceive_tool: str = ""
    perceive_args: dict = field(default_factory=dict)
    perceive_observation: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    reasoning: str = ""
    observation: str = ""
    final_answer: str = ""
    #: Deliberation surface — populated only when the matching layer is enabled.
    options_considered: list = field(default_factory=list)
    expected_outcome: dict = field(default_factory=dict)
    critique: str = ""
    veto: bool = False
    prediction_mismatch: str = ""
    #: Native model thinking behind the decision, when reasoning is enabled.
    model_reasoning: str = ""
    screenshot: Optional[bytes] = None  # viewport JPEG after this action ran — UI-only, never sent to the LLM
    usage: dict = field(default_factory=empty_usage)
    cumulative_usage: dict = field(default_factory=empty_usage)


# ===========================================================================
# ThinkingAgent — perceives (read-only) + decides ONE action at a time
# ===========================================================================

class ThinkingAgent:
    """
    Maintains a REAL growing Bedrock conversation (like ReAct's PlaywrightAgent —
    context is never lossily re-summarized between calls), but is only ever
    given READ-ONLY tools plus the special decide_action hand-off tool. It
    never executes a mutating action itself.
    """

    _PERCEPTION_TOOL_NAMES = ["search_elements", "expand_element", "get_page_structure", "get_current_url"]

    @staticmethod
    def _build_decide_action_tool(config: DeliberationConfig) -> dict:
        """
        decide_action's schema, grown to match the enabled deliberation layers.

        These extra fields ARE the mechanism for forced_schema and
        predict_verify. A small model will not weigh alternatives or commit to a
        prediction because the system prompt asked it to — but it cannot emit a
        decision at all without filling in fields the schema demands. Putting the
        requirement in the schema rather than the prose is the difference between
        a suggestion and a constraint.
        """
        properties: dict[str, Any] = {
            "current_goal": {
                "type": "string",
                "description": "Short rolling summary of what you're working on right now (shown to the user).",
            },
            "instruction": {
                "type": "string",
                "description": (
                    "Precise natural-language instruction, e.g. \"click the Sign In "
                    "button\" or \"type 'hello@x.com' into the email field\". Omit "
                    "when task_complete is true."
                ),
            },
            "target_selector": {
                "type": "string",
                "description": (
                    "Exact selector for the target element, copied from "
                    "search_elements/expand_element results. Omit when task_complete "
                    "is true or the instruction needs no element (e.g. navigate)."
                ),
            },
            "task_complete": {
                "type": "boolean",
                "description": "True if the ORIGINAL TASK is already fully achieved.",
            },
            "final_answer": {
                "type": "string",
                "description": "Required when task_complete is true.",
            },
        }

        if config.forced_schema:
            properties["options_considered"] = {
                "type": "array",
                "minItems": 2,
                "description": (
                    "REQUIRED whenever task_complete is false. At least TWO distinct "
                    "candidate elements or actions you weighed before choosing, drawn "
                    "from what you actually saw in a perception result. If only one "
                    "candidate looks right, the second entry should be the next-best "
                    "alternative and why you rejected it. Never invent candidates."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {
                            "type": "string",
                            "description": "Selector/ref of this candidate, e.g. 'aria-ref=e12'.",
                        },
                        "label": {
                            "type": "string",
                            "description": "Short human description, e.g. 'Sign in button in header'.",
                        },
                        "supports": {
                            "type": "string",
                            "description": "The case FOR this candidate being the right target.",
                        },
                        "against": {
                            "type": "string",
                            "description": "The case AGAINST it — the reason it might be wrong.",
                        },
                        "score": {
                            "type": "number",
                            "description": "Confidence 0.0-1.0 that this is the right target.",
                        },
                    },
                    "required": ["ref", "supports", "against", "score"],
                },
            }
            properties["chosen_ref"] = {
                "type": "string",
                "description": (
                    "Which options_considered ref you picked. Must match target_selector."
                ),
            }

        if config.predict_verify:
            properties["expected_outcome"] = {
                "type": "object",
                "description": (
                    "REQUIRED whenever task_complete is false. What you expect to be "
                    "true immediately AFTER this action runs. This is checked against "
                    "the real page automatically and you will be told if you were "
                    "wrong, so predict honestly rather than optimistically."
                ),
                "properties": {
                    "url_changes": {
                        "type": "boolean",
                        "description": "True if this action should navigate to a different URL.",
                    },
                    "expect_text": {
                        "type": "string",
                        "description": (
                            "A short distinctive string that should be visible on the page "
                            "afterwards, e.g. 'Welcome back'. Leave empty if you can't "
                            "predict one — do not guess."
                        ),
                    },
                    "target_should_vanish": {
                        "type": "boolean",
                        "description": (
                            "True if the element you're acting on should no longer be "
                            "visible afterwards (e.g. clicking a dialog's Close button)."
                        ),
                    },
                },
            }

        # options_considered/expected_outcome are intentionally NOT in `required`:
        # a task_complete decision legitimately has neither, and Bedrock's schema
        # dialect has no way to express "required only when task_complete is
        # false". next_decision() enforces the conditional part in code instead.
        return {
            "toolSpec": {
                "name": "decide_action",
                "description": (
                    "Hand off ONE concrete action to the Actor agent to execute. Call this "
                    "once you know exactly which element and what to do to it — do not call "
                    "it just to read or search."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": properties,
                        "required": ["current_goal", "task_complete"],
                    }
                },
            }
        }

    SYSTEM_PROMPT = (
        "You are the THINKING agent in a two-role browser automation system. You decide "
        "WHAT to do next; a separate ACTOR agent turns your decision into an exact tool "
        "call and executes it — you never execute a mutating action yourself.\n\n"
        "You have READ-ONLY tools to understand the page:\n"
        "  get_page_structure(...)  — CALL THIS FIRST on any new page. A compact outline of "
        "roles, names, and visible TEXT (view counts, prices, dates, titles — everything "
        "readable), tagged with short refs like [ref=e12]. Use that ref DIRECTLY as "
        "target_selector, formatted as \"aria-ref=e12\" — no need to construct a CSS selector.\n"
        "  search_elements(terms)   — filter candidate elements by guessed keywords, across "
        "the WHOLE page, when it's too large to read the whole outline at once\n"
        "  expand_element(selector) — full detail on one candidate before deciding\n"
        "  get_current_url()        — current URL/title\n\n"
        "Use these freely to gather information — each call executes immediately and its "
        "result is shown to you, so you can search/expand/inspect as many times as needed.\n\n"
        "When you know exactly what to do next, call decide_action ONCE with current_goal, "
        "instruction, and target_selector. Set task_complete=true with final_answer ONLY when "
        "the ORIGINAL TASK is fully done — don't set an instruction in that same call.\n\n"
        "RULES:\n"
        "1. Never invent a selector — always get it from get_page_structure/search_elements/"
        "expand_element first (prefer copying its aria-ref).\n"
        "2. One decide_action = ONE concrete action. Don't bundle multiple actions together.\n"
        "2a. For CLICK intents, choose the exact target and issue a direct click action (no exploratory "
        "reads in the same decision).\n"
        "2b. For TYPE intents, do two decisions when needed: first focus/click the target input, then "
        "type_text into that SAME target with the FULL intended string in one action. Do not type into an "
        "unfocused/uncertain field.\n"
        "2d. Do NOT spell multi-character text one key at a time with press_key. Use press_key only for "
        "control/navigation keys (Enter, Tab, Escape, Arrow keys, Backspace, Delete, Home, End, PageUp, "
        "PageDown).\n"
        "2c. If the user gives exact visible UI text, prefer that exact target first (exact label/button/text) "
        "before fuzzy keyword hunting. Use search_elements only as fallback when exact text is not directly "
        "available.\n"
        "3. If ACTION RESULT shows something unexpected (error, wrong page, 404 title), account "
        "   for it in your next decision — there is no separate evaluator, you decide what "
        "   happens next.\n"
        "4. Never invent a URL — only navigate to hrefs you have actually read.\n"
        "5. NEVER set task_complete=true unless final_answer is copied or directly derived from "
        "   content that literally appeared in a perception result or ACTION RESULT in this "
        "   conversation. Do not invent, estimate, or recall a plausible-sounding value from "
        "   general knowledge (a view count, price, date, name, etc. you have not actually seen). "
        "   If the needed data isn't visible yet, keep gathering info or hand off another action.\n"
        "6. After any typing action, verify from the next observation that the intended value appears on the "
        "same field/section before moving on."
    )

    #: Appended to SYSTEM_PROMPT per enabled layer. The schema already forces the
    #: fields to be present; this tells the model what a good answer in them
    #: looks like, so it fills them with real deliberation instead of filler.
    _DELIBERATION_PROMPT = {
        "forced_schema": (
            "\n\nBEFORE EVERY ACTION — WEIGH THE ALTERNATIVES:\n"
            "decide_action requires options_considered with at least TWO candidates. "
            "Fill it honestly from elements you actually saw in a perception result: "
            "for each, the case for it, the case against it, and a 0.0-1.0 score. Then "
            "set chosen_ref (and target_selector) to the one you picked. If your top "
            "candidate scores below ~0.5, prefer gathering more information over acting."
        ),
        "predict_verify": (
            "\n\nBEFORE EVERY ACTION — PREDICT THE RESULT:\n"
            "decide_action requires expected_outcome. State what should be true right "
            "after the action: does the URL change, what distinctive text should appear, "
            "should the target disappear. This is checked against the real page and you "
            "will be told when it does not match, so an honest prediction is far more "
            "useful to you than a flattering one. Leave expect_text empty rather than "
            "guessing a string you have no reason to expect."
        ),
    }

    def __init__(
        self,
        llm: BedrockClient,
        max_perception_calls: int = 6,
        deliberation: Optional[DeliberationConfig] = None,
    ):
        self.llm = llm
        self.config = deliberation or DeliberationConfig()
        self._max_perception_calls = max_perception_calls
        self._messages: list[dict] = []
        self._tools = [
            build_tool_schema(getattr(pt, name)) for name in self._PERCEPTION_TOOL_NAMES
        ] + [self._build_decide_action_tool(self.config)]
        self._valid_names = set(self._PERCEPTION_TOOL_NAMES) | {"decide_action"}

        system = self.SYSTEM_PROMPT
        if self.config.forced_schema:
            system += self._DELIBERATION_PROMPT["forced_schema"]
        if self.config.predict_verify:
            system += self._DELIBERATION_PROMPT["predict_verify"]
        self._system = system

    def reset(self) -> None:
        self._messages = []

    @staticmethod
    def _vote_key(decision: Decision) -> str:
        """
        What two sampled decisions have to agree on to count as the same choice.

        The target is the part that matters — two runs that pick the same element
        but word the instruction differently agree in every way we care about.
        """
        if decision.task_complete:
            return "\x00COMPLETE"
        return (decision.target_selector or decision.instruction).strip().lower()

    @staticmethod
    def _confidence(decision: Decision) -> float:
        """Best candidate score the run reported, for breaking vote ties."""
        scores = [
            o.get("score", 0.0) for o in decision.options_considered
            if isinstance(o, dict) and isinstance(o.get("score"), (int, float))
        ]
        return max(scores) if scores else 0.0

    async def next_decision(
        self, task: str, last_observation: Optional[str]
    ) -> tuple[Decision, list[tuple[str, dict, str]]]:
        """
        Decide the next action, optionally by self-consistency vote.

        With self_consistency_k == 1 this is exactly _decide_once. Above 1, the
        conversation is forked K ways, each fork decides independently, and the
        majority target wins; the winning fork's message history becomes the real
        one so the conversation continues from a single coherent branch.

        Forking is only safe because the ThinkingAgent is read-only. Its
        perception loop re-runs in every fork, which costs tokens but cannot
        touch the page — the same trick would corrupt state if this agent could
        click things.
        """
        k = self.config.self_consistency_k
        if k == 1:
            return await self._decide_once(task, last_observation)

        baseline = copy.deepcopy(self._messages)
        runs: list[tuple[Decision, list[tuple[str, dict, str]], list[dict]]] = []
        for i in range(k):
            self._messages = copy.deepcopy(baseline)
            decision, perceptions = await self._decide_once(task, last_observation)
            runs.append((decision, perceptions, self._messages))
            logger.info("Self-consistency %d/%d -> %r", i + 1, k, self._vote_key(decision))

        tally = Counter(self._vote_key(d) for d, _, _ in runs)
        winning_key, votes = tally.most_common(1)[0]

        # Among runs that chose the winning target, keep the most confident one —
        # they agree on the target but may differ in how well they justified it.
        winner = max(
            (r for r in runs if self._vote_key(r[0]) == winning_key),
            key=lambda r: self._confidence(r[0]),
        )
        decision, perceptions, messages = winner
        self._messages = messages

        # All K forks each called Bedrock at least once — every fork's tokens
        # were actually spent, even though only the winner's messages survive.
        decision.usage = add_usage(*(d.usage for d, _, _ in runs))

        if votes == 1:
            # Every fork picked something different — no consensus at all. Say so
            # in the goal so it surfaces in the UI rather than passing silently.
            logger.warning(
                "Self-consistency: %d runs produced %d distinct targets — no majority",
                k, len(tally),
            )
            decision.current_goal = f"{decision.current_goal} (low confidence: no majority across {k} samples)"
        else:
            logger.info("Self-consistency: %d/%d agreed on %r", votes, k, winning_key)
        return decision, perceptions

    async def _decide_once(
        self, task: str, last_observation: Optional[str]
    ) -> tuple[Decision, list[tuple[str, dict, str]]]:
        """
        Runs an internal perception loop (search/expand/structure/current_url —
        executed directly, no second LLM call) until the model calls
        decide_action, then returns that decision.

        Returns:
            (Decision, perceptions) where perceptions is the list of
            (tool_name, tool_args, observation) calls made along the way,
            for the caller to fire UI events for.
        """
        if not self._messages:
            self._messages = [_user_message(
                f"ORIGINAL TASK:\n{task}\n\nDecide the first action."
            )]
        elif last_observation is not None:
            self._messages.append(_user_message(
                f"ACTION RESULT:\n{last_observation}\n\nDecide the next action, or call "
                f"decide_action with task_complete=true if the original task is now fully done."
            ))

        perceptions: list[tuple[str, dict, str]] = []
        # Native thinking accumulates across the perception turns that lead to
        # one decision, so the UI can show the whole train of thought behind it
        # rather than only whatever the final turn happened to emit.
        reasoning_parts: list[str] = []
        decision_usage = empty_usage()  # summed across every perception-loop converse() call

        for _ in range(self._max_perception_calls):
            # Send the COMPACTED history, not the raw log. This conversation grows
            # for the whole run — every perception result and every full action
            # observation — and it is re-sent on each of the (up to
            # _max_perception_calls) turns behind a single decision, so untrimmed
            # growth is multiplied, not just accumulated. self._messages still
            # keeps everything; only what goes over the wire is reduced.
            response = await self.llm.converse(
                messages=compact_history(self._messages, keep_full_last_n=2),
                system=self._system, tools=self._tools,
            )
            decision_usage = add_usage(decision_usage, usage_from_response(response))
            blocks = response["output"]["message"].get("content", [])
            self._messages.append({"role": "assistant", "content": blocks})
            turn_reasoning = self.llm.extract_reasoning(blocks)
            if turn_reasoning:
                reasoning_parts.append(turn_reasoning)
            tool_uses = _extract_tool_uses(blocks)

            if not tool_uses:
                self._messages.append(_user_message(
                    "You must call decide_action or one of the read-only tools — no plain-text replies."
                ))
                continue

            # Resolve EVERY tool_use from this turn into ONE batched toolResult
            # message. Bedrock requires a toolResult for each toolUseId in the
            # message immediately following an assistant turn — models like
            # Claude Haiku routinely call more than one tool in parallel, and
            # only ever resolving tool_uses[0] leaves any sibling toolUseId
            # dangling, which makes the NEXT converse() call fail with
            # "Expected toolResult blocks ... for the following Ids".
            decision: Optional[Decision] = None
            result_blocks: list[dict] = []

            for tu in tool_uses:
                name = tu.get("name", "")
                tool_use_id = tu.get("toolUseId", "")
                inp = tu.get("input", {})

                if name not in self._valid_names:
                    prefix_match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", name)
                    prefix = prefix_match.group(0) if prefix_match else ""
                    resolved = prefix if prefix in self._valid_names else ""
                    if not resolved:
                        matches = difflib.get_close_matches(name, self._valid_names, n=1, cutoff=0.4)
                        resolved = matches[0] if matches else ""
                    name = resolved

                if not name:
                    logger.warning("ThinkingAgent: unrecognized tool '%s' — nudging", tu.get("name"))
                    result_blocks.append(_tool_result_block(
                        tool_use_id,
                        f"'{tu.get('name')}' is not available. Use one of: "
                        f"{', '.join(self._PERCEPTION_TOOL_NAMES)}, or decide_action.",
                        is_error=True,
                    ))
                    continue

                if name == "decide_action":
                    if decision is not None:
                        # Model called decide_action more than once in one turn — keep the first.
                        result_blocks.append(_tool_result_block(
                            tool_use_id, "Only one decide_action per turn is used — ignored."
                        ))
                        continue

                    task_complete = bool(inp.get("task_complete", False))
                    final_answer = inp.get("final_answer", "")

                    # Deterministic anti-hallucination guard: an empty final_answer on
                    # task_complete means the model declared success without grounding a
                    # value in anything actually seen — reject and keep looping instead
                    # of accepting a silent no-op success (see SYSTEM_PROMPT rule 5).
                    if task_complete and not final_answer.strip():
                        logger.warning("ThinkingAgent: task_complete with empty final_answer — rejecting")
                        result_blocks.append(_tool_result_block(
                            tool_use_id,
                            "Rejected: task_complete was true but final_answer was empty — you "
                            "must ground the answer in something actually observed before "
                            "completing. Keep gathering info or hand off another action.",
                            is_error=True,
                        ))
                        continue

                    # Deliberation guards. The schema cannot express "required
                    # only when task_complete is false", so the conditional half
                    # is enforced here — same reject-and-loop shape as the guard
                    # above, which is what makes these layers a constraint the
                    # model must satisfy rather than advice it can ignore.
                    options = inp.get("options_considered") or []
                    expected = inp.get("expected_outcome") or {}

                    if not task_complete and self.config.forced_schema and len(options) < 2:
                        logger.warning(
                            "ThinkingAgent: decision with %d option(s) considered — rejecting",
                            len(options),
                        )
                        result_blocks.append(_tool_result_block(
                            tool_use_id,
                            f"Rejected: you listed {len(options)} candidate(s) in "
                            "options_considered, but at least 2 are required before "
                            "acting. Name the alternative you are NOT picking and why "
                            "it loses. If you genuinely cannot see a second candidate, "
                            "search or expand the page first.",
                            is_error=True,
                        ))
                        continue

                    if not task_complete and self.config.predict_verify and not isinstance(expected, dict):
                        logger.warning("ThinkingAgent: decision without expected_outcome — rejecting")
                        result_blocks.append(_tool_result_block(
                            tool_use_id,
                            "Rejected: expected_outcome is required before acting. State "
                            "what should be true right after this action runs.",
                            is_error=True,
                        ))
                        continue

                    result_blocks.append(_tool_result_block(tool_use_id, "Decision received."))
                    decision = Decision(
                        current_goal=inp.get("current_goal", ""),
                        instruction=inp.get("instruction", ""),
                        target_selector=inp.get("target_selector", ""),
                        task_complete=task_complete,
                        final_answer=final_answer,
                        options_considered=options,
                        expected_outcome=expected if isinstance(expected, dict) else {},
                        model_reasoning="\n\n".join(reasoning_parts),
                        usage=decision_usage,
                    )
                    continue

                # Perception tool — execute directly, no second LLM, feed result back
                observation = await execute_tool_call(name, inp)
                perceptions.append((name, inp, observation))
                result_blocks.append(_tool_result_block(tool_use_id, observation))

            # Every toolUseId from this turn now has a matching toolResult, all in
            # one message — safe to send back to Bedrock on the next converse() call.
            self._messages.append({"role": "user", "content": result_blocks})

            if decision is not None:
                return decision, perceptions
            # else: perception-only turn (or a rejected decide_action) — loop again

        logger.warning("ThinkingAgent: perception budget (%d) exhausted without a decision",
                        self._max_perception_calls)
        return Decision(
            current_goal="(perception budget exhausted)",
            task_complete=True,
            final_answer="Could not decide on a next action after extensive searching.",
            usage=decision_usage,
        ), perceptions


# ===========================================================================
# ActorAgent — translates ONE decision into an exact tool call (no grouping)
# ===========================================================================

class ActorAgent:
    """
    Stateless — one call per action (like PlanningAgent/SpecialistAgent).
    Sees the FULL flat tool list (no grouping — the instruction is already
    narrow enough that routing isn't needed) minus session-management tools
    and ThinkingAgent's own perception tools. Always an LLM call, even for
    seemingly mechanical mappings (explicit design choice).
    """

    _ACT_TOOL = {
        "toolSpec": {
            "name": "resolve_action",
            "description": "Translate the given instruction into the exact tool call that fulfils it.",
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
        "You are the ACTOR in a two-role browser automation system. A separate THINKING "
        "agent has already decided exactly which element and what to do to it — your only "
        "job is to translate that into the correct tool call.\n\n"
        "AVAILABLE TOOLS:\n{tool_list}\n\n"
        "RULES:\n"
        "1. Only use tool names exactly as listed above.\n"
        "2. Use TARGET SELECTOR exactly as given — do not modify it.\n"
        "3. Never pass a 'timeout' parameter.\n"
        "4. Map intent literally: click/tap/press button -> click; type/enter/input text -> type_text; "
        "focus cursor/caret -> focus; keypress intent -> press_key.\n"
        "5. For text-entry instructions, do not switch to fill unless explicitly asked. Prefer type_text "
        "to simulate keyboard input, and pass the complete string in tool_args.text.\n"
        "6. Use press_key ONLY for special keys or shortcuts (e.g., Enter/Tab/Escape/ArrowDown/Ctrl+A). "
        "Never pass phrases/sentences to press_key and never decompose a sentence into per-character "
        "press_key calls.\n"
        "7. Always call resolve_action."
    )

    # Tools that operate on one concrete element selector. The ThinkingAgent is
    # responsible for picking that element; the Actor should not drift to a
    # different selector during tool-arg synthesis.
    _TARGET_SELECTOR_ARG_BY_TOOL: dict[str, str] = {
        "click": "selector",
        "hover": "selector",
        "focus": "selector",
        "fill": "selector",
        "type_text": "selector",
        "press_key": "selector",
        "select_option": "selector",
        "check_checkbox": "selector",
        "uncheck_checkbox": "selector",
        "upload_file": "selector",
        "get_text": "selector",
        "get_all_text": "selector",
        "get_attribute": "selector",
        "get_input_value": "selector",
        "evaluate_js_on_element": "selector",
        "wait_for_selector": "selector",
        "assert_visible": "selector",
        "assert_text": "selector",
        "drag_and_drop": "target_selector",
    }

    _RAW_SELECTOR_PREFIXES = (
        "aria-ref=",
        "text=",
        "xpath=",
        "css=",
    )

    _CONTROL_KEYS = {
        "Enter", "Tab", "Escape", "Esc", "Backspace", "Delete",
        "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
        "Home", "End", "PageUp", "PageDown", "Insert",
    }

    # Session mgmt is handled directly by GuidedAgent; perception tools belong to ThinkingAgent.
    _EXCLUDED_TOOLS = {
        "start_browser", "stop_browser",
        "search_elements", "expand_element", "get_page_structure", "get_current_url",
    }

    def __init__(self, llm: BedrockClient, all_tool_schemas: list[dict]):
        self.llm = llm
        self._schemas = [s for s in all_tool_schemas if s["toolSpec"]["name"] not in self._EXCLUDED_TOOLS]
        self._valid_tools = {s["toolSpec"]["name"] for s in self._schemas}

        # Condensed (name + params + short description) rather than full
        # docstrings — with ~50 flat tools here (vs. ~2-10 per SpecialistAgent
        # group), full docstrings for all of them would bloat the prompt.
        lines = []
        for s in self._schemas:
            spec = s["toolSpec"]
            props = spec["inputSchema"]["json"].get("properties", {})
            required = spec["inputSchema"]["json"].get("required", [])
            param_str = ", ".join(props.keys())
            req_note = f" [required: {', '.join(required)}]" if required else ""
            lines.append(f"  {spec['name']}({param_str}){req_note} — {spec['description'][:100]}")
        self._system = self._SYSTEM_TEMPLATE.format(tool_list="\n".join(lines))

    @staticmethod
    def _extract_long_quoted_phrase(text: str) -> str:
        """Return the longest quoted phrase from instruction text, if any."""
        if not text:
            return ""
        matches = re.findall(r"['\"]([^'\"]+)['\"]", text)
        cleaned = [m.strip() for m in matches if m and m.strip()]
        return max(cleaned, key=len) if cleaned else ""

    @classmethod
    def _enforce_text_entry_policy(
        cls,
        instruction: str,
        tool_name: str,
        tool_args: dict,
    ) -> tuple[str, dict, str]:
        """
        Guardrail: reserve press_key for control keys and convert text-entry
        key presses into one-shot type_text when possible.
        """
        if tool_name != "press_key":
            return tool_name, tool_args, ""

        key = str((tool_args or {}).get("key", "")).strip()
        if not key:
            return tool_name, tool_args, ""

        # Keep genuine control keys and shortcuts as press_key.
        if key in cls._CONTROL_KEYS or re.match(r"^(Ctrl|Alt|Shift|Meta)\+", key):
            return tool_name, tool_args, ""

        phrase = cls._extract_long_quoted_phrase(instruction)
        if phrase and len(phrase) > 1 and phrase != key:
            patched = dict(tool_args or {})
            patched.pop("key", None)
            patched["text"] = phrase
            return (
                "type_text",
                patched,
                (
                    "Actor typing guard: converted press_key to type_text with "
                    f"full phrase {phrase!r} to avoid per-character typing."
                ),
            )

        # If actor produced a non-control key payload that is longer than one
        # key token, treat it as text content.
        if len(key) > 1:
            patched = dict(tool_args or {})
            patched.pop("key", None)
            patched["text"] = key
            return (
                "type_text",
                patched,
                "Actor typing guard: converted non-control press_key payload to type_text.",
            )

        # Single printable character with typing intent: keep progress while
        # still using text-entry API rather than keypress API.
        lowered = instruction.lower()
        if any(token in lowered for token in ("type", "enter", "input")) and len(key) == 1:
            patched = dict(tool_args or {})
            patched.pop("key", None)
            patched["text"] = key
            return (
                "type_text",
                patched,
                "Actor typing guard: converted character press_key to type_text.",
            )

        return tool_name, tool_args, ""

    @classmethod
    def _normalize_target_selector(cls, target_selector: str) -> str:
        """
        Keep real selectors as-is, but treat plain human text as an exact text
        selector. This lets frontend prompts like "click 'Continue to checkout'"
        become deterministic selector inputs.
        """
        target = (target_selector or "").strip()
        if not target:
            return ""

        if target.startswith(cls._RAW_SELECTOR_PREFIXES):
            return target

        # Already looks like CSS/XPath/locator syntax.
        if (
            target.startswith("//")
            or target.startswith("#")
            or target.startswith(".")
            or target.startswith("[")
            or ">>" in target
            or re.search(r"[#.\[\]():>/]", target)
            or re.match(r"^[a-zA-Z_][a-zA-Z0-9_-]*=", target)
        ):
            return target

        escaped = target.replace("\\", "\\\\").replace('"', '\\"')
        return f'text="{escaped}"'

    @classmethod
    def _bind_target_selector(
        cls,
        tool_name: str,
        tool_args: dict,
        target_selector: str,
    ) -> tuple[dict, str]:
        """
        Enforce that element-targeting tools use the ThinkingAgent's target.
        Returns (patched_args, note).
        """
        arg_name = cls._TARGET_SELECTOR_ARG_BY_TOOL.get(tool_name)
        if not arg_name:
            return tool_args, ""

        normalized_target = cls._normalize_target_selector(target_selector)
        if not normalized_target:
            return tool_args, ""

        patched = dict(tool_args or {})
        current = (patched.get(arg_name) or "").strip()
        if current == normalized_target:
            return patched, ""

        patched[arg_name] = normalized_target
        if current:
            return patched, (
                f"Actor selector override: replaced {arg_name}={current!r} "
                f"with ThinkingAgent target {normalized_target!r}."
            )
        return patched, (
            f"Actor selector binding: set {arg_name} to ThinkingAgent target "
            f"{normalized_target!r}."
        )

    async def resolve(self, instruction: str, target_selector: str, current_goal: str) -> tuple[str, dict, str, dict]:
        """
        Returns (tool_name, tool_args, reasoning, usage). An empty tool_name
        means the actor could not resolve a tool — reasoning explains why.
        `usage` is the token usage of this one converse() call.
        """
        prompt = (
            f"CURRENT GOAL:\n{current_goal}\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"TARGET SELECTOR:\n{target_selector or '(none needed for this instruction)'}\n\n"
            "Which tool call fulfils this instruction?"
        )
        response = await self.llm.converse(
            messages=[_user_message(prompt)], system=self._system, tools=[self._ACT_TOOL],
        )
        usage = usage_from_response(response)
        blocks = response["output"]["message"].get("content", [])
        tool_uses = _extract_tool_uses(blocks)
        if not tool_uses:
            logger.warning("ActorAgent: no tool call for instruction %r", instruction)
            return "", {}, "No tool call from actor.", usage

        inp = tool_uses[0].get("input", {})
        raw_name = inp.get("tool_name", "")
        tool_args = inp.get("tool_args", {})
        reasoning = inp.get("reasoning", "")

        if raw_name not in self._valid_tools:
            prefix_match = re.match(r"[a-zA-Z_][a-zA-Z0-9_]*", raw_name)
            prefix = prefix_match.group(0) if prefix_match else ""
            if prefix in self._valid_tools:
                logger.warning("ActorAgent: recovered '%s' from garbled tool_name '%s'", prefix, raw_name[:80])
                raw_name = prefix
            else:
                matches = difflib.get_close_matches(raw_name, self._valid_tools, n=1, cutoff=0.4)
                if matches:
                    logger.warning("ActorAgent: '%s' → corrected to '%s'", raw_name, matches[0])
                    raw_name = matches[0]
                else:
                    return "", {}, f"Could not resolve a tool for instruction: {instruction!r}", usage

        raw_name, tool_args, policy_note = self._enforce_text_entry_policy(
            instruction, raw_name, tool_args,
        )

        patched_args, note = self._bind_target_selector(raw_name, tool_args, target_selector)
        notes = [n for n in (policy_note, note) if n]
        if notes:
            reasoning = f"{reasoning}\n\n" + "\n".join(notes)
            reasoning = reasoning.strip()

        return raw_name, patched_args, reasoning, usage


# ===========================================================================
# CriticAgent — argues the ThinkingAgent's choice is wrong, before it runs
# ===========================================================================

class CriticAgent:
    """
    Stateless devil's advocate, one call per action, enabled by
    DeliberationConfig.critic.

    Sees only the task and the proposed decision — deliberately NOT the
    conversation that produced it, so it cannot inherit the reasoning it is
    supposed to be attacking. It is prompted to argue for a veto, because a
    critic asked neutrally "is this fine?" agrees with almost anything.

    The bias has a cost: an over-eager critic vetoes correct actions and causes
    thrash, which is why GuidedAgent allows at most one veto in a row and why
    the veto bar below is explicitly "concretely wrong", not "imperfect".
    """

    _CRITIC_TOOL = {
        "toolSpec": {
            "name": "judge_action",
            "description": "Return your verdict on the proposed action.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "veto": {
                            "type": "boolean",
                            "description": (
                                "True ONLY if the action is concretely wrong — wrong element, "
                                "wrong page, unsupported by the evidence, or actively harmful "
                                "to the task. Not merely suboptimal."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": (
                                "One or two sentences. On a veto, say what is wrong and what "
                                "to check instead. On a pass, say briefly why it holds up."
                            ),
                        },
                    },
                    "required": ["veto", "reason"],
                }
            },
        }
    }

    SYSTEM_PROMPT = (
        "You are the CRITIC in a browser automation system. Another agent has chosen an "
        "action but has NOT executed it yet. Your job is to try to find a concrete reason "
        "it is the wrong action, and say so before it runs.\n\n"
        "Argue for a veto where you can. But the bar is real error, not imperfection:\n"
        "  VETO if the target contradicts the stated goal, the evidence cited doesn't "
        "support the choice, the agent is acting on a page that can't have what it wants, "
        "or a rejected candidate was clearly the better one.\n"
        "  DO NOT veto because you would have phrased it differently, because a different "
        "element might also work, or because you would like more information first. A "
        "wrongly vetoed action wastes a whole turn and teaches the agent nothing.\n\n"
        "If you cannot name something concretely wrong, pass it. Always call judge_action."
    )

    def __init__(self, llm: BedrockClient):
        self.llm = llm

    async def review(self, task: str, decision: Decision) -> tuple[bool, str, dict]:
        """Returns (veto, reason, usage). Any failure passes the action — a
        broken critic must not be able to block the whole run. `usage` is
        empty_usage() when the call never happened (exception before/around it)."""
        options = "\n".join(
            f"  - {o.get('ref', '?')} ({o.get('label', '')}) score={o.get('score', '?')}\n"
            f"      for: {o.get('supports', '')}\n"
            f"      against: {o.get('against', '')}"
            for o in decision.options_considered
        ) or "  (none recorded)"

        prompt = (
            f"ORIGINAL TASK:\n{task}\n\n"
            f"CURRENT GOAL:\n{decision.current_goal}\n\n"
            f"PROPOSED ACTION:\n{decision.instruction}\n"
            f"TARGET: {decision.target_selector or '(none)'}\n\n"
            f"CANDIDATES IT WEIGHED:\n{options}\n\n"
            f"PREDICTED RESULT:\n{json.dumps(decision.expected_outcome) or '(none)'}\n\n"
            "Is this action concretely wrong?"
        )
        try:
            response = await self.llm.converse(
                messages=[_user_message(prompt)],
                system=self.SYSTEM_PROMPT,
                tools=[self._CRITIC_TOOL],
            )
            usage = usage_from_response(response)
            tool_uses = _extract_tool_uses(response["output"]["message"].get("content", []))
            if not tool_uses:
                return False, "", usage
            inp = tool_uses[0].get("input", {})
            return bool(inp.get("veto", False)), inp.get("reason", ""), usage
        except Exception as exc:
            logger.warning("CriticAgent failed (%s) — passing the action through", exc)
            return False, "", empty_usage()


# ===========================================================================
# GuidedAgent — drives the ThinkingAgent <-> ActorAgent <-> Executor loop
# ===========================================================================

class GuidedAgent:
    """
    Third mode, alongside PlaywrightAgent (ReAct) and OrchestratorAgent (plan
    + per-group specialist). Continuously: ThinkingAgent perceives + decides
    ONE action -> ActorAgent resolves it to an exact tool call -> executed ->
    result feeds back into ThinkingAgent's next decision. No upfront plan;
    "current_goal" gives a lightweight rolling view of intent instead.
    """

    #: Temperature the ThinkingAgent samples at when self-consistency voting is
    #: on. High enough to produce genuinely different candidate decisions,
    #: low enough not to produce nonsense ones.
    _VOTE_TEMPERATURE = 0.7

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_iterations: int = 40,
        on_event: Optional[Callable[[GuidedEvent], Any]] = None,
        reasoning_budget_tokens: Optional[int] = None,
        deliberation: Optional["DeliberationConfig"] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Args:
            on_delta: Callback(kind, text) fired per chunk as the Thinking or
                Actor agent produces it, so their output appears live instead of
                only when a whole turn lands. See BedrockClient.converse_stream().
        """
        self._llm = BedrockClient.create(
            model_id=model_id, region=region, profile=profile,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_budget_tokens=reasoning_budget_tokens,
            on_delta=on_delta,
        )
        self._deliberation = deliberation or DeliberationConfig()

        # Self-consistency needs the K samples to actually differ. At
        # temperature 0 every fork returns the same decision and the vote is an
        # expensive no-op, so the ThinkingAgent gets its own client with
        # sampling turned up. (Claude with thinking enabled is already pinned to
        # temperature 1, so this is a no-op in that combination.)
        thinking_llm = self._llm
        if self._deliberation.self_consistency_k > 1 and temperature < self._VOTE_TEMPERATURE:
            logger.info(
                "Self-consistency k=%d: raising the ThinkingAgent's temperature "
                "%.2f -> %.2f so the samples differ. Actor and Critic stay at %.2f.",
                self._deliberation.self_consistency_k, temperature,
                self._VOTE_TEMPERATURE, temperature,
            )
            thinking_llm = BedrockClient.create(
                model_id=model_id, region=region, profile=profile,
                max_tokens=max_tokens, temperature=self._VOTE_TEMPERATURE,
                reasoning_budget_tokens=reasoning_budget_tokens,
                on_delta=on_delta,
            )

        self._thinking = ThinkingAgent(thinking_llm, deliberation=self._deliberation)
        self._actor = ActorAgent(self._llm, get_all_tool_schemas())
        self._critic = CriticAgent(self._llm) if self._deliberation.critic else None
        self._executor = ToolExecutorAgent()
        self._max_iterations = max_iterations
        self.on_event = on_event
        self._total_usage: dict = empty_usage()

    def _track(self, usage: dict) -> dict:
        """Add `usage` to the running total and return the new cumulative total."""
        self._total_usage = add_usage(self._total_usage, usage)
        return self._total_usage

    async def _fire(self, event: GuidedEvent) -> None:
        if self.on_event is None:
            return
        result = self.on_event(event)
        if asyncio.iscoroutine(result):
            await result

    async def run(self, task: str) -> str:
        """
        Execute the guided loop for a task.

        Args:
            task: High-level task in plain English

        Returns:
            Final answer string.
        """
        reset_loop_detection()
        browser_start_obs = await execute_tool_call("start_browser", {})
        logger.info("Browser started: %s", browser_start_obs)
        browser_result = json.loads(browser_start_obs)
        if "error" in browser_result:
            raise RuntimeError(f"Failed to start browser: {browser_result['error']}")

        last_observation: Optional[str] = None
        vetoed_previous = False

        for i in range(1, self._max_iterations + 1):
            decision, perceptions = await self._thinking.next_decision(task, last_observation)
            decision_cumulative = self._track(decision.usage)

            for p_tool, p_args, p_obs in perceptions:
                await self._fire(GuidedEvent(
                    event_type="perceive", iteration=i,
                    perceive_tool=p_tool, perceive_args=p_args, perceive_observation=p_obs,
                    cumulative_usage=self._total_usage,
                ))

            await self._fire(GuidedEvent(
                event_type="goal", iteration=i, current_goal=decision.current_goal,
                options_considered=decision.options_considered,
                expected_outcome=decision.expected_outcome,
                model_reasoning=decision.model_reasoning,
                usage=decision.usage, cumulative_usage=decision_cumulative,
            ))

            if decision.task_complete:
                await self._fire(GuidedEvent(
                    event_type="done", iteration=i, final_answer=decision.final_answer,
                    cumulative_usage=self._total_usage,
                ))
                if pt._state["browser"] is not None:
                    await pt.stop_browser()
                return decision.final_answer

            # -- critic ---------------------------------------------------
            # Skipped immediately after a veto: the critic is prompted to look
            # for problems, so letting it judge the replacement it just forced
            # invites an endless veto-redecide ping-pong. One objection per
            # decision, then the ThinkingAgent's next call stands.
            if self._critic is not None and not vetoed_previous:
                veto, critique, critic_usage = await self._critic.review(task, decision)
                critic_cumulative = self._track(critic_usage)
                await self._fire(GuidedEvent(
                    event_type="critique", iteration=i, critique=critique, veto=veto,
                    current_goal=decision.current_goal,
                    usage=critic_usage, cumulative_usage=critic_cumulative,
                ))
                if veto:
                    vetoed_previous = True
                    last_observation = (
                        f"NO ACTION WAS TAKEN. A reviewing agent rejected your proposed "
                        f"action ({decision.instruction!r} on "
                        f"{decision.target_selector or '(no target)'}): {critique} "
                        f"Choose a different action or gather more information first."
                    )
                    continue
            vetoed_previous = False

            # -- act ------------------------------------------------------
            before_state = (
                await _capture_page_state() if self._deliberation.predict_verify else {}
            )

            tool_name, tool_args, reasoning, actor_usage = await self._actor.resolve(
                decision.instruction, decision.target_selector, decision.current_goal,
            )
            actor_cumulative = self._track(actor_usage)
            await self._fire(GuidedEvent(
                event_type="act_select", iteration=i,
                tool_name=tool_name, tool_args=tool_args, reasoning=reasoning,
                usage=actor_usage, cumulative_usage=actor_cumulative,
            ))

            mismatch = ""
            if not tool_name:
                last_observation = json.dumps({"error": reasoning or "Actor could not resolve a tool."})
            else:
                last_observation = await self._executor.execute(tool_name, tool_args)
                if self._deliberation.predict_verify:
                    mismatch = await _verify_prediction(
                        decision.expected_outcome,
                        before_state,
                        decision.target_selector,
                        last_observation,
                    )
                    if mismatch:
                        logger.info("Prediction mismatch on iteration %d: %s", i, mismatch.strip())
                        last_observation += mismatch

            await self._fire(GuidedEvent(
                event_type="act_execute", iteration=i, observation=last_observation,
                prediction_mismatch=mismatch,
                screenshot=pt._state.get("last_screenshot"),
                cumulative_usage=self._total_usage,
            ))

        final_answer = f"Stopped after {self._max_iterations} iterations without completing the task."
        await self._fire(GuidedEvent(
            event_type="failed", iteration=self._max_iterations, final_answer=final_answer,
            cumulative_usage=self._total_usage,
        ))
        if pt._state["browser"] is not None:
            await pt.stop_browser()
        return final_answer


# ===========================================================================
# Sync convenience wrapper
# ===========================================================================

def run_guided(
    task: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    region: str = "us-east-1",
    profile: Optional[str] = None,
    reasoning_budget_tokens: Optional[int] = None,
    deliberation: Optional[DeliberationConfig] = None,
) -> str:
    """
    Synchronous entry point for the guided (thinking + actor) run.

    Example:
        from guided_agent import run_guided, DeliberationConfig
        print(run_guided("Go to HN and return the top 3 story titles."))
        print(run_guided("...", deliberation=DeliberationConfig(critic=True)))
    """
    async def _run():
        agent = GuidedAgent(
            model_id=model_id, region=region, profile=profile,
            reasoning_budget_tokens=reasoning_budget_tokens,
            deliberation=deliberation,
        )
        return await agent.run(task)

    return asyncio.run(_run())
