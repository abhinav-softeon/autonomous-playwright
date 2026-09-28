"""
LLM Agent with Playwright Tools (AWS Bedrock)
----------------------------------------------
Abstractions for:
  1. Building a Bedrock Converse API tool schema from playwright_tools.py
  2. Executing tool calls returned by the LLM
  3. Running a full agentic loop: prompt → LLM → tool calls → results → repeat

Dependencies:
    pip install boto3 playwright

AWS credentials must be configured via one of:
  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  - AWS profile: ~/.aws/credentials
  - IAM role (if running on EC2/ECS/Lambda)

Usage:
    import asyncio
    from llm_agent import PlaywrightAgent

    async def main():
        agent = PlaywrightAgent(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        result = await agent.run("Go to https://example.com and tell me the page title.")
        print(result)

    asyncio.run(main())

Supported model IDs (examples):
    anthropic.claude-3-5-sonnet-20241022-v2:0   ← recommended
    anthropic.claude-3-haiku-20240307-v1:0
    amazon.nova-pro-v1:0
    amazon.nova-lite-v1:0
    us.amazon.nova-pro-v1:0                      ← cross-region inference prefix
"""

import asyncio
import inspect
import json
import logging
import re
from collections import Counter
from typing import Any, Callable, Optional

import boto3

import playwright_tools as pt
import selector_cache as sc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Tool registry — auto-built from playwright_tools module
# ---------------------------------------------------------------------------

# All async public functions in playwright_tools are registered as LLM tools.
_TOOL_REGISTRY: dict[str, Any] = {
    name: func
    for name, func in inspect.getmembers(pt, inspect.iscoroutinefunction)
    if not name.startswith("_")
}


# ---------------------------------------------------------------------------
# Schema builder — converts Python function signatures → Bedrock Converse tool schema
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
    "NoneType": "null",
}


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Extract param descriptions from a Google-style docstring."""
    descriptions: dict[str, str] = {}
    if not docstring:
        return descriptions
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped.lower() in ("args:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(" ") and stripped.endswith(":") and " " not in stripped:
                break
            match = re.match(r"(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


def _resolve_json_type(annotation: Any) -> str:
    """Resolve a Python type annotation to a JSON Schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    # Optional[T] → unwrap T
    args = getattr(annotation, "__args__", None)
    if args and type(None) in args:
        inner = [a for a in args if a is not type(None)][0]
        return _PY_TO_JSON_TYPE.get(getattr(inner, "__name__", str(inner)), "string")
    return _PY_TO_JSON_TYPE.get(getattr(annotation, "__name__", str(annotation)), "string")


#: Google-style section headers that end the prose part of a docstring.
_DOC_SECTION_RE = re.compile(
    r"^\s*(args|arguments|parameters|returns|raises|yields|example|examples):\s*$",
    re.IGNORECASE,
)

#: Upper bound on a single tool description. ReAct sends every tool's schema on
#: every turn, so unbounded descriptions across ~60 tools would dominate the
#: prompt. Generous enough that no current docstring is actually clipped.
_MAX_TOOL_DESCRIPTION_CHARS = 1600


def _docstring_description(docstring: str) -> str:
    """
    Everything before the first Google-style section header (Args:/Returns:/...).

    Deliberately NOT just the first paragraph: the paragraphs after it are where
    tool docstrings put the guidance that actually prevents misuse — when to
    prefer this tool over another, how long a returned ref stays valid, which
    tools are fallbacks. Truncating to the first paragraph silently drops all of
    it, and the model then has nothing to go on but the one-line summary.
    """
    kept: list[str] = []
    for line in docstring.splitlines():
        if _DOC_SECTION_RE.match(line):
            break
        kept.append(line)
    text = "\n".join(kept).strip()
    if len(text) > _MAX_TOOL_DESCRIPTION_CHARS:
        text = text[:_MAX_TOOL_DESCRIPTION_CHARS].rstrip() + " …(truncated)"
    return text


def build_tool_schema(func) -> dict:
    """
    Build an AWS Bedrock Converse API tool schema from a Python async function.

    Bedrock format:
    {
      "toolSpec": {
        "name": ...,
        "description": ...,
        "inputSchema": {
          "json": {
            "type": "object",
            "properties": { ... },
            "required": [ ... ]
          }
        }
      }
    }
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    description = _docstring_description(doc)
    param_docs = _parse_docstring_params(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        json_type = _resolve_json_type(param.annotation)
        prop: dict[str, Any] = {"type": json_type}
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]
        properties[param_name] = prop

        # Required = no default and not Optional
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "toolSpec": {
            "name": func.__name__,
            "description": description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            },
        }
    }


def get_all_tool_schemas() -> list[dict]:
    """Return Bedrock Converse tool schemas for all registered playwright tools."""
    return [build_tool_schema(func) for func in _TOOL_REGISTRY.values()]


# ---------------------------------------------------------------------------
# Tool executor — runs a single tool call returned by Bedrock
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

# Tools that change page state — cleaned HTML is appended after these
_PAGE_MUTATING_TOOLS = {
    "fill", "type_text", "press_key", "click", "select_option",
    "check_checkbox", "uncheck_checkbox", "upload_file", "drag_and_drop",
    "go_back", "go_forward", "reload",
}

# Tools that take a 'selector' argument and need pre-flight validation.
# wait_for_selector and assert_* are intentionally excluded — they handle
# elements that may not be visible yet; pre-flight would defeat their purpose.
_SELECTOR_TOOLS = {
    "fill", "type_text", "press_key", "click", "hover", "focus",
    "select_option", "check_checkbox", "uncheck_checkbox", "upload_file",
    "drag_and_drop", "get_text", "get_all_text", "get_attribute", "get_input_value",
    "evaluate_js_on_element",
}

# Tools that act on one specific element — the fake mouse pointer is moved to
# it before the action so the post-action screenshot shows WHERE the agent
# acted. Value is the argument holding the selector to point at.
_CURSOR_TOOLS: dict[str, str] = {
    "click": "selector", "hover": "selector", "focus": "selector",
    "fill": "selector", "type_text": "selector", "press_key": "selector",
    "select_option": "selector", "check_checkbox": "selector",
    "uncheck_checkbox": "selector", "upload_file": "selector",
    "drag_and_drop": "target_selector",   # point at where it lands, not where it started
}

# Of those, the ones that are a real click — these also get the click ring.
_CURSOR_CLICK_TOOLS = {
    "click", "select_option", "check_checkbox", "uncheck_checkbox", "drag_and_drop",
}

# Tools whose empty result is a sign the selector matched the wrong page
_EMPTY_RESULT_CHECKS: dict = {
    "get_all_text": lambda r: not r.get("texts") or not any(t.strip() for t in r["texts"]),
    "get_text":     lambda r: not (r.get("text") or "").strip(),
}


#: Page outline attached to a selector error. Deliberately smaller than a normal
#: page view: a failing run produces these in bursts, and they are never trimmed.
_SELECTOR_ERROR_VIEW_CHARS = 3000


async def _validate_selector(selector: str, tool_name: str = "") -> str | None:
    """
    Check whether the selector matches at least one VISIBLE element on the page.
    A hidden element that exists in the DOM but is not visible will also fail —
    Playwright cannot interact with hidden elements.
    Returns None if valid, or an error string with real selectors.
    """
    page = pt._state.get("page")
    if page is None:
        return None

    # Step 1: count — if this fails it's a bad selector, let Playwright handle it
    try:
        locator = page.locator(selector)
        count = await locator.count()
    except Exception:
        return None

    valid = True
    reason = ""

    if count == 0:
        valid = False
        reason = "does not exist"
    elif count > 1 and tool_name not in ("get_all_text", "get_text"):
        # Playwright's selector-based action methods (click, fill, ...) are
        # strict-mode: they raise a "strict mode violation" if a selector
        # resolves to more than one element. Catch that here with a clear,
        # actionable message instead of letting it surface as a cryptic
        # Playwright error. get_text/get_all_text are excluded — they're
        # read-only and adapt to multiple matches themselves, since widening
        # a READ is safe but widening an ACTION (e.g. "click all matches")
        # could have real, hard-to-undo side effects.
        valid = False
        reason = (
            f"matches {count} elements, not one — {tool_name or 'this tool'} can only target "
            f"a single element. Narrow the selector (e.g. add an id, >> nth=N, or a more "
            f"specific attribute)"
        )
    else:
        first = locator.first
        try:
            is_visible = await first.is_visible(timeout=1000)
            is_enabled = await first.is_enabled(timeout=1000)
            if not is_visible:
                valid = False
                reason = "exists but is hidden"
            elif not is_enabled:
                valid = False
                reason = "exists and visible but is disabled"
        except Exception:
            valid = False
            reason = "could not determine visibility/enabled state"

        # For fill/type_text, must be an actual form field
        if valid and tool_name in ("fill", "type_text"):
            try:
                tag = await first.evaluate("el => el.tagName.toLowerCase()")
                if tag not in ("input", "textarea", "select"):
                    valid = False
                    reason = (
                        f"resolved to a <{tag}> element — not a form field. "
                        f"fill/type_text only work on input, textarea, or select"
                    )
            except Exception:
                pass

        # For click, verify target contains the hit-test point (bubbling is fine;
        # only reject if a completely different overlay element is on top).
        # Skipped for aria-ref=eN — it's a Playwright-only selector engine, not
        # real CSS, so document.querySelector(sel) below can never resolve it.
        if valid and tool_name == "click" and not selector.startswith("aria-ref="):
            try:
                box = await first.bounding_box()
                if box:
                    cx = box["x"] + box["width"] / 2
                    cy = box["y"] + box["height"] / 2
                    target_contains_hit = await page.evaluate(
                        """([sel, x, y]) => {
                            const target = document.querySelector(sel);
                            const hit = document.elementFromPoint(x, y);
                            if (!target || !hit) return true;
                            return target === hit || target.contains(hit);
                        }""",
                        [selector, cx, cy]
                    )
                    if not target_contains_hit:
                        valid = False
                        reason = (
                            "a different element is covering the click target. "
                            "Try press_key(selector, 'Enter') instead of clicking."
                        )
            except Exception:
                pass

    if not valid:
        # Only persist truly absent selectors — not timing/hidden failures.
        # aria-ref=eN selectors are scoped to one snapshot generation, not
        # stable across future page loads/runs — never persist those.
        if reason == "does not exist" and not selector.startswith("aria-ref="):
            try:
                sc.save_failed_selector(pt._state["page"].url, selector)
            except Exception:
                pass

        # Keep this SMALL. It used to attach 8,000 chars of raw HTML plus the
        # full element list — ~3,858 tok measured, stored under the "error" key
        # where history trimming never reached it. A handful of concrete
        # alternatives is what's actionable; a page dump is not.
        parts = [
            f"SELECTOR ERROR: '{selector}' cannot be interacted with ({reason}).",
            "Do NOT use this selector again.",
        ]

        try:
            view = await pt._page_view(max_chars=_SELECTOR_ERROR_VIEW_CHARS)
            parts.append(f"\nCURRENT PAGE: {view.get('url')}  |  Title: {view.get('title')}")
            if view.get("structure"):
                parts.append(
                    "\nPAGE OUTLINE (pass any [ref=eN] straight through as "
                    '"aria-ref=eN"):\n' + view["structure"]
                )
        except Exception as e:
            parts.append(f"\n(page outline unavailable: {e})")

        return "\n".join(parts)

    return None


async def _do_execute_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Execute a playwright tool by name with the input dict from Bedrock.

    Before running selector-based tools, validates the selector exists on the
    current page. If it doesn't, returns an error with the current page HTML
    so the agent can immediately pick a correct selector.

    For page-mutating tools, automatically appends the page view (see
    playwright_tools._page_view) so the agent always sees the updated page.
    """
    func = _TOOL_REGISTRY.get(tool_name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # Pre-flight selector validation
    if tool_name in _SELECTOR_TOOLS and pt._state.get("page") is not None:
        selector = tool_input.get("selector") or tool_input.get("source_selector", "")
        if selector:
            err = await _validate_selector(selector, tool_name=tool_name)
            if err:
                logger.warning("Selector validation failed for %s: %s", tool_name, selector)
                return json.dumps({"error": err})

    try:
        logger.info("Executing tool: %s(%s)", tool_name, tool_input)
        result = await func(**tool_input)
        logger.info("Tool result: %s", result)

        if tool_name in _PAGE_MUTATING_TOOLS:
            pt._invalidate_index()

        # Auto-append the ONE page view for mutating tools that don't already
        # carry it (see playwright_tools._page_view — structure, not raw HTML).
        if (
            tool_name in _PAGE_MUTATING_TOOLS
            and isinstance(result, dict)
            and "error" not in result
            and "structure" not in result
            and pt._state["page"] is not None
        ):
            try:
                result.update(await pt._page_view())
            except Exception:
                pass

        # Warn when a reading tool returns empty — likely wrong page or wrong selector
        empty_check = _EMPTY_RESULT_CHECKS.get(tool_name)
        if empty_check and isinstance(result, dict) and "error" not in result and empty_check(result):
            try:
                view = await pt._page_view()
                result["empty_result_warning"] = (
                    f"Selector '{tool_input.get('selector')}' returned nothing on "
                    f"{view.get('url')} (title: {view.get('title')}). "
                    f"The element may not exist on this page."
                )
                result.update(view)
            except Exception:
                pass

        # For fill/type_text: verify the value was actually written and save to cache
        if (
            tool_name in ("fill", "type_text")
            and isinstance(result, dict)
            and "error" not in result
        ):
            selector = tool_input.get("selector", "")
            expected = tool_input.get("value") or tool_input.get("text", "")
            if selector and expected and pt._state.get("page"):
                try:
                    actual = await pt._state["page"].input_value(selector, timeout=2000)
                    result["verified_value"] = actual
                    if expected.lower() in actual.lower():
                        # Save as a working selector for this domain — skip
                        # aria-ref=eN, which is only valid for the current
                        # snapshot generation, not stable across future runs.
                        if not selector.startswith("aria-ref="):
                            try:
                                sc.save_working_selector(
                                    pt._state["page"].url, "input_field", selector
                                )
                            except Exception:
                                pass
                    else:
                        result["verification_warning"] = (
                            f"Expected '{expected}' but input contains '{actual}'. "
                            "The text may not have been entered correctly."
                        )
                except Exception:
                    pass  # input_value not supported for this element type

        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error("Tool %s raised: %s", tool_name, e, exc_info=True)
        return json.dumps({"error": str(e)})


#: (tool, args) signatures seen this run, and how often. Loop detection lived
#: only in orchestrator._check_hard_stop, scoped to one subtask — so ReAct and
#: Guided had none at all, and even the orchestrator's counter reset every step,
#: letting the same dead selector be retried 3x per step forever. This is
#: run-scoped and mode-agnostic: whatever the architecture above, a tool call
#: that has already failed identically twice is not worth a third round trip.
_call_signatures: Counter = Counter()

#: Identical attempts allowed before the call is refused instead of executed.
_MAX_IDENTICAL_CALLS = 3


def reset_loop_detection() -> None:
    """Clear the per-run call history. Called at the start of every mode's run()."""
    _call_signatures.clear()


def _loop_signature(tool_name: str, tool_input: dict, page_fingerprint: str = "") -> str:
    try:
        return (
            f"{tool_name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"
            f":{page_fingerprint}"
        )
    except (TypeError, ValueError):
        return f"{tool_name}:{tool_input!r}:{page_fingerprint}"


async def execute_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Public entry point every mode calls through. Runs the tool via
    _do_execute_tool_call, then stashes a fresh viewport screenshot in
    pt._state["last_screenshot"] for the UI's step-by-step trace — a
    side channel only, never part of the JSON returned here, so it never
    reaches the LLM's context (screenshots are large; putting them in the
    tool result would blow up token usage on every single call).

    A screenshot of a page nobody is pointing at doesn't show much, so for
    element-targeting tools the fake mouse pointer is glided onto the target
    first and re-asserted (with a click ring, where it was a click) after — the
    action itself may have navigated and dropped the overlay. Both halves are
    best-effort and cannot fail the tool call.

    Identical repeated calls are refused rather than executed — see
    _call_signatures. The refusal is returned as a normal observation so the
    model reads it as feedback and changes approach, instead of the run dying.
    """
    page_fingerprint = ""
    page = pt._state.get("page")
    if page is not None:
        try:
            page_fingerprint = f"{page.url}|{await page.title()}"
        except Exception:
            page_fingerprint = page.url or ""

    signature = _loop_signature(tool_name, tool_input, page_fingerprint)
    _call_signatures[signature] += 1
    attempt = _call_signatures[signature]
    if attempt > _MAX_IDENTICAL_CALLS:
        logger.warning("Loop guard: refusing %s (attempt %d, identical args)",
                       tool_name, attempt)
        return json.dumps({
            "error": (
                f"LOOP DETECTED: {tool_name}() has already been called "
                f"{_MAX_IDENTICAL_CALLS} times with exactly these arguments and "
                f"did not get you further. It was NOT run again. Do something "
                f"different: pick a different element, read the page outline "
                f"again with get_page_structure(), navigate straight to the "
                f"target URL, or finish with what you already have."
            ),
            "attempts": attempt,
        })

    selector_arg = _CURSOR_TOOLS.get(tool_name)
    selector = tool_input.get(selector_arg) if selector_arg else None
    point = await pt._cursor_to_selector(selector) if selector else None

    observation = await _do_execute_tool_call(tool_name, tool_input)

    if point is not None:
        await pt._mark_cursor_action(point, click=tool_name in _CURSOR_CLICK_TOOLS)
    else:
        # No target of its own (a read, a navigate): keep the pointer where the
        # last action left it so it doesn't blink out of the trace.
        await pt._restore_cursor()
    pt._state["last_screenshot"] = await pt._capture_step_screenshot(tool_name)
    return observation


# ---------------------------------------------------------------------------
# Bedrock Converse message helpers
# ---------------------------------------------------------------------------

def _user_message(text: str) -> dict:
    """Wrap plain text as a Bedrock user message."""
    return {"role": "user", "content": [{"text": text}]}


def _tool_result_message(tool_use_id: str, result_text: str, is_error: bool = False) -> dict:
    """
    Wrap a tool result as a Bedrock user message containing a toolResult block.
    Tool results must be sent back as a 'user' role message in Bedrock.
    """
    return {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": result_text}],
                    **({"status": "error"} if is_error else {}),
                }
            }
        ],
    }


# ---------------------------------------------------------------------------
# History compaction — shared by every stateful agent
# ---------------------------------------------------------------------------
# Old approach stripped two named keys ("html", "page_html") and nothing else,
# which meant it reclaimed 29% on a clean run and 0% on a failing one — every
# byte of bloat on a bad run lives under "error", "elements" or "structure".
# This is key-agnostic instead: past a small window, an observation keeps only
# what a later decision can act on (did it work, where am I, what broke) and
# drops every page dump, no matter which key it arrived under.

#: Payload fields that describe the page rather than the outcome. Any of these
#: in an older observation is dead weight — the page has moved on since.
_BULKY_RESULT_KEYS = {
    "html", "page_html", "structure", "structure_note", "elements",
    "matches", "blocks", "texts", "snapshot",
}

#: Chars kept from a long free-text field (e.g. an error) in an older observation.
_STALE_TEXT_MAX_CHARS = 300


def compact_tool_result_text(text: str, max_chars: int = _STALE_TEXT_MAX_CHARS) -> str:
    """
    Shrink one past observation to its decision-relevant core.

    Handles both shapes an observation takes: a JSON tool result (drop the page
    views, keep status/url/verification flags, truncate long errors) and plain
    text (truncate). Never raises — an un-shrinkable payload is returned as-is,
    since a compaction failure must not corrupt the conversation.
    """
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        if len(text) > max_chars:
            return text[:max_chars] + f"  […{len(text) - max_chars:,} chars dropped]"
        return text

    if not isinstance(parsed, dict):
        return text

    cleaned: dict = {}
    dropped = 0
    for key, value in parsed.items():
        if key in _BULKY_RESULT_KEYS:
            dropped += len(json.dumps(value, ensure_ascii=False))
            continue
        if isinstance(value, str) and len(value) > max_chars:
            # Long free text is nearly always an error message whose first line
            # is the actionable part and whose tail is an attached page outline.
            dropped += len(value) - max_chars
            cleaned[key] = value[:max_chars] + "  […truncated]"
        else:
            cleaned[key] = value
    if dropped:
        cleaned["_compacted"] = f"stale page detail dropped ({dropped:,} chars)"
    return json.dumps(cleaned, ensure_ascii=False)


def _is_observation_message(msg: dict) -> bool:
    """
    True for a user message carrying a tool observation.

    Covers both conventions in use: Bedrock toolResult blocks (ReAct,
    ThinkingAgent's perception turns) and the plain-text "ACTION RESULT:"
    messages GuidedAgent feeds back after an action.
    """
    if msg.get("role") != "user":
        return False
    for block in msg.get("content", []):
        if "toolResult" in block:
            return True
        text = block.get("text", "")
        if text.startswith("ACTION RESULT:"):
            return True
    return False


def compact_history(messages: list[dict], keep_full_last_n: int = 2) -> list[dict]:
    """
    Return `messages` with every observation older than the last N compacted.

    The originals are left untouched — the caller keeps its full log and sends
    this reduced copy, so nothing is destroyed, it just stops being re-sent on
    every single turn.
    """
    indices = [i for i, m in enumerate(messages) if _is_observation_message(m)]
    if len(indices) <= keep_full_last_n:
        return messages
    compact_at = set(indices[:-keep_full_last_n] if keep_full_last_n else indices)

    out = []
    for i, msg in enumerate(messages):
        if i not in compact_at:
            out.append(msg)
            continue
        new_content = []
        for block in msg.get("content", []):
            if "toolResult" in block:
                tr = block["toolResult"]
                new_items = []
                for item in tr.get("content", []):
                    if "text" in item:
                        new_items.append({"text": compact_tool_result_text(item["text"])})
                    else:
                        new_items.append(item)
                new_content.append({"toolResult": {**tr, "content": new_items}})
            elif "text" in block:
                new_content.append({"text": compact_tool_result_text(block["text"])})
            else:
                new_content.append(block)
        out.append({**msg, "content": new_content})
    return out


def _extract_text(content_blocks: list[dict]) -> str:
    """Extract all text from a Bedrock assistant content block list."""
    return " ".join(
        block["text"] for block in content_blocks if "text" in block
    ).strip()


def _extract_tool_uses(content_blocks: list[dict]) -> list[dict]:
    """Extract all toolUse blocks from a Bedrock assistant content block list."""
    return [block["toolUse"] for block in content_blocks if "toolUse" in block]


def _extract_thinking(content_blocks: list[dict]) -> str:
    """
    Extract model reasoning/"thinking" text from a Bedrock assistant content
    block list, if the model returned any `reasoningContent` blocks.

    reasoningContent looks like:
        {"reasoningContent": {"reasoningText": {"text": "...", "signature": "..."}}}
    (a model may also return `redactedContent` instead of `reasoningText` when
    the provider encrypts part of the reasoning — that case is skipped here
    since there's no human-readable text to show.)

    Only reasoning-capable models (e.g. Claude extended-thinking, Amazon Nova
    reasoning models) emit this block, and only when reasoning is enabled for
    that model — see BedrockClient's additional_model_request_fields. For any
    other model this always returns "".
    """
    parts = []
    for block in content_blocks:
        rc = block.get("reasoningContent")
        if not rc:
            continue
        text = (rc.get("reasoningText") or {}).get("text")
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# LLM clients — AWS Bedrock Converse API, one subclass per model family
# ---------------------------------------------------------------------------

# Signatures of "this model/account can't do ConverseStream", as opposed to a
# real failure. Streaming is gated by its own IAM action
# (bedrock:InvokeModelWithResponseStream) which a policy may grant for some
# inference profiles and not others, so this is a per-model runtime discovery,
# not something that can be checked once up front.
_STREAM_UNAVAILABLE_MARKERS = (
    "invokemodelwithresponsestream",
    "does not support streaming",
    "streaming is not supported",
    "unsupported operation: converse_stream",
)


def _is_streaming_unavailable(exc: Exception) -> bool:
    """True if `exc` says streaming isn't permitted/supported, not that the call failed."""
    text = str(exc).lower()
    return any(marker in text for marker in _STREAM_UNAVAILABLE_MARKERS)

# ---------------------------------------------------------------------------
# Token usage helpers — shared by every agent mode (ReAct, Orchestrator,
# Guided) so per-prompt and running-total token counts are computed the same
# way everywhere instead of each caller reading response["usage"] by hand.
# ---------------------------------------------------------------------------

def empty_usage() -> dict:
    """A zeroed usage dict, safe to accumulate into or display before any
    LLM call has happened yet."""
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def usage_from_response(response: dict) -> dict:
    """
    Normalize a converse()/converse_stream() response's `usage` dict to
    {"input_tokens", "output_tokens", "total_tokens"}.

    Bedrock's own keys are "inputTokens"/"outputTokens"/"totalTokens"; this
    also tolerates a missing or partially-populated usage dict (e.g. a
    streamed response whose metadata event never arrived) by defaulting
    anything absent to 0.
    """
    usage = (response or {}).get("usage") or {}
    input_tokens = usage.get("inputTokens", 0) or 0
    output_tokens = usage.get("outputTokens", 0) or 0
    total_tokens = usage.get("totalTokens", input_tokens + output_tokens) or 0
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def add_usage(*usages: dict) -> dict:
    """Sum any number of usage dicts (as returned by usage_from_response)
    into one combined dict. Missing keys are treated as 0."""
    total = empty_usage()
    for u in usages:
        if not u:
            continue
        total["input_tokens"] += u.get("input_tokens", 0) or 0
        total["output_tokens"] += u.get("output_tokens", 0) or 0
        total["total_tokens"] += u.get("total_tokens", 0) or 0
    return total


class BedrockClient:
    """
    Async-compatible wrapper around the AWS Bedrock Converse API (boto3).

    boto3 is synchronous, so calls are offloaded to a thread pool via
    asyncio.get_event_loop().run_in_executor() to avoid blocking the event loop.

    The Converse API is uniform across model families, but turning on model
    reasoning/"thinking" is NOT — each family uses a different
    additionalModelRequestFields key, a different budget shape, and imposes
    different constraints on inferenceConfig while reasoning is active. All of
    that lives in a subclass; everything else lives here.

    Construct via BedrockClient.create(model_id, ...) — it dispatches on the
    model ID. Instantiating this base class directly is legal and gives you the
    generic fallback: additional_model_request_fields is forwarded verbatim and
    reasoning is reported as unsupported.
    """

    #: Substrings identifying this family within a Bedrock model ID. Matched
    #: after any cross-region inference prefix ("us.", "eu.", "apac.") is
    #: stripped. Empty on the base class — it is the fallback, not a match.
    _MODEL_MARKERS: tuple[str, ...] = ()

    #: Human-readable family name, used in log and error messages.
    _FAMILY = "generic"

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        additional_model_request_fields: Optional[dict] = None,
        reasoning_budget_tokens: Optional[int] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Args:
            model_id: Bedrock model ID.
                      Examples:
                        "anthropic.claude-haiku-4-5-20251001-v1:0"
                        "anthropic.claude-3-5-sonnet-20241022-v2:0"
                        "amazon.nova-lite-v1:0"
                        "us.amazon.nova-pro-v1:0"  ← cross-region inference prefix
            region: AWS region where Bedrock is enabled (e.g. "us-east-1")
            profile: AWS CLI profile name (None = use default/env vars)
            max_tokens: Max tokens in the response
            temperature: Sampling temperature (0 = deterministic). A subclass may
                override this while reasoning is on if the family requires it —
                Claude, for one, rejects any temperature but 1 alongside thinking.
            additional_model_request_fields: Raw escape hatch, forwarded as
                Bedrock's additionalModelRequestFields on every call and merged
                UNDER anything the subclass generates for reasoning. Only reach
                for this to set a field this class doesn't model; prefer
                reasoning_budget_tokens for reasoning.
            reasoning_budget_tokens: Provider-neutral way to ask for model
                reasoning/"thinking". The subclass translates it into whatever
                key and shape its family actually accepts, and clamps it to that
                family's legal range. None (the default) leaves reasoning off.
                Passing it to a family that doesn't support reasoning logs a
                warning and is otherwise ignored — it never reaches Bedrock.
            on_delta: Callback(kind, text) invoked for every chunk as the model
                produces it — see converse_stream(). Setting it switches every
                converse() call on this client to the streaming API, so all
                three agent modes stream simply by passing this down. Called
                from a worker thread, so it must be thread-safe (a
                queue.Queue.put or a print is; touching Streamlit state is not).
        """
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.additional_model_request_fields = additional_model_request_fields
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.on_delta = on_delta
        #: Set once if Bedrock refuses ConverseStream for this model/account, so
        #: the fallback is decided one time rather than on every single call.
        self._streaming_disabled = False

        if reasoning_budget_tokens is not None and not self.supports_reasoning():
            logger.warning(
                "reasoning_budget_tokens=%s ignored: model '%s' resolved to the "
                "'%s' family, which this client does not know how to enable "
                "reasoning on.",
                reasoning_budget_tokens, model_id, self._FAMILY,
            )

    # -- family dispatch ----------------------------------------------------

    @staticmethod
    def _strip_region_prefix(model_id: str) -> str:
        """Drop a cross-region inference prefix ('us.', 'eu.', 'apac.')."""
        for prefix in ("us.", "eu.", "apac."):
            if model_id.startswith(prefix):
                return model_id[len(prefix):]
        return model_id

    @classmethod
    def _handles(cls, model_id: str) -> bool:
        bare = cls._strip_region_prefix(model_id).lower()
        return any(marker in bare for marker in cls._MODEL_MARKERS)

    @staticmethod
    def create(model_id: str, **kwargs) -> "BedrockClient":
        """
        Build the right client subclass for a Bedrock model ID.

        Falls back to the generic BedrockClient (no reasoning support, raw
        passthrough of additional_model_request_fields) when no family claims
        the ID, so an unrecognised model still works for plain Converse calls.
        """
        for family in (ClaudeBedrockClient, NovaBedrockClient):
            if family._handles(model_id):
                return family(model_id=model_id, **kwargs)
        logger.info(
            "No model family matched '%s' — using the generic Bedrock client "
            "(reasoning unavailable).", model_id,
        )
        return BedrockClient(model_id=model_id, **kwargs)

    # -- family hooks (overridden per subclass) -----------------------------

    def supports_reasoning(self) -> bool:
        """Whether this family can have reasoning switched on by this client."""
        return False

    def _reasoning_fields(self) -> Optional[dict]:
        """
        The additionalModelRequestFields fragment that enables reasoning, or
        None when reasoning is off or unsupported.
        """
        return None

    def _reasoning_active(self) -> bool:
        return self.reasoning_budget_tokens is not None and self.supports_reasoning()

    def _inference_config(self) -> dict:
        """inferenceConfig for this call, after any family-specific overrides."""
        return {"maxTokens": self.max_tokens, "temperature": self.temperature}

    def extract_reasoning(self, content_blocks: list[dict]) -> str:
        """
        Pull reasoning text out of an assistant content block list. The Converse
        `reasoningContent` shape is uniform across families, so the shared
        implementation covers everything; a family with a bespoke shape can
        override.
        """
        return _extract_thinking(content_blocks)

    def _request_kwargs(
        self,
        messages: list[dict],
        system: Optional[str],
        tools: Optional[list[dict]],
        additional_model_request_fields: Optional[dict],
    ) -> tuple[dict, dict]:
        """
        Build the Bedrock request body shared by converse() and converse_stream().

        Returns (kwargs, extra_fields) — extra_fields is handed back so the
        callers can name it in the error message when Bedrock rejects it.
        """
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": self._inference_config(),
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {
                "tools": tools,
                "toolChoice": {"auto": {}},
            }

        # Raw passthrough first, then the family's reasoning fragment on top —
        # so a hand-written additional_model_request_fields can add fields this
        # class doesn't model without being able to silently corrupt the
        # reasoning config the subclass just validated.
        extra_fields: dict = dict(
            additional_model_request_fields or self.additional_model_request_fields or {}
        )
        reasoning_fields = self._reasoning_fields()
        if reasoning_fields:
            extra_fields.update(reasoning_fields)
        if extra_fields:
            kwargs["additionalModelRequestFields"] = extra_fields

        return kwargs, extra_fields

    def _extra_fields_error(self, extra_fields: dict, exc: Exception) -> Optional[RuntimeError]:
        """Turn an additionalModelRequestFields rejection into an actionable error."""
        if extra_fields and "additionalModelRequestFields" in str(exc):
            return RuntimeError(
                f"Bedrock rejected additionalModelRequestFields={extra_fields!r} "
                f"for model '{self.model_id}' (resolved to the '{self._FAMILY}' "
                f"family via {type(self).__name__}). Either the model ID matched "
                f"the wrong family, or this family's reasoning key/shape has "
                f"changed — check the model's Bedrock model card. "
                f"Original error: {exc}"
            )
        return None

    async def converse(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        additional_model_request_fields: Optional[dict] = None,
    ) -> dict:
        """
        Call the Bedrock Converse API asynchronously.

        If this client was given an on_delta callback, the call is routed
        through converse_stream() instead so the caller sees the response
        being produced. Either way the return shape is identical, which is
        what lets all three agent modes stream without touching their loops.

        Args:
            messages: List of Bedrock-format message dicts
            system: Optional system prompt string
            tools: Optional list of Bedrock toolSpec dicts
            additional_model_request_fields: Per-call override of the constructor's
                default of the same name (see __init__ docstring).

        Returns:
            The raw Bedrock converse() response dict.
        """
        if self.on_delta is not None and not self._streaming_disabled:
            try:
                return await self.converse_stream(
                    messages, system, tools, additional_model_request_fields
                )
            except Exception as exc:
                if not _is_streaming_unavailable(exc):
                    raise
                # Streaming needs the separate bedrock:InvokeModelWithResponseStream
                # IAM action, which an account may hold for some models and not
                # others, and a few models don't support it at all. Neither is a
                # reason to fail the task — drop to non-streaming for the rest of
                # this run and tell the caller why the output stopped being live.
                self._streaming_disabled = True
                message = (
                    f"Streaming unavailable for '{self.model_id}' — falling back to "
                    f"non-streaming for the rest of this run, so output arrives per "
                    f"step instead of per token. ({exc})"
                )
                logger.warning(message)
                try:
                    self.on_delta("notice", message)
                except Exception:
                    logger.debug("on_delta callback raised on notice", exc_info=True)

        kwargs, extra_fields = self._request_kwargs(
            messages, system, tools, additional_model_request_fields
        )
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._client.converse(**kwargs)
            )
        except Exception as exc:
            raise (self._extra_fields_error(extra_fields, exc) or exc) from exc

    async def converse_stream(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        additional_model_request_fields: Optional[dict] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ) -> dict:
        """
        Same call as converse(), but over Bedrock's ConverseStream API, invoking
        a callback for each chunk as it arrives instead of waiting for the whole
        response. Used so a run shows its reasoning and its tool call as they
        are produced, rather than going quiet for the length of every LLM turn.

        The assembled result is byte-for-byte the same shape converse() returns
        ({"output": {"message": ...}}, "stopReason", "usage"), including the
        `signature` on reasoning blocks — which Claude requires back verbatim on
        the next turn when extended thinking is combined with tool use. Callers
        therefore need no streaming-specific branch.

        Callback kinds:
            "reasoning"  — model thinking text
            "text"       — assistant-visible text
            "tool"       — a tool call started; text is the tool name
            "tool_input" — a fragment of that tool call's JSON arguments

        Args:
            on_delta: Per-call override of the constructor's callback.

        Returns:
            A converse()-shaped response dict.
        """
        callback = on_delta or self.on_delta
        kwargs, extra_fields = self._request_kwargs(
            messages, system, tools, additional_model_request_fields
        )
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: self._consume_stream(kwargs, callback)
            )
        except Exception as exc:
            raise (self._extra_fields_error(extra_fields, exc) or exc) from exc

    def _consume_stream(
        self, kwargs: dict, callback: Optional[Callable[[str, str], Any]]
    ) -> dict:
        """
        Drain a ConverseStream event stream into a converse()-shaped dict.

        Runs on a worker thread (boto3's event stream is a blocking iterator),
        so `callback` is invoked from that thread — see the on_delta note in
        __init__. Blocks arrive interleaved and are keyed by contentBlockIndex,
        so they're accumulated per index and emitted in index order to preserve
        the ordering Bedrock requires (reasoning before text/toolUse).
        """
        def emit(kind: str, text: str) -> None:
            if callback is None or not text:
                return
            try:
                callback(kind, text)
            except Exception:
                logger.debug("on_delta callback raised; ignoring", exc_info=True)

        blocks: dict[int, dict] = {}

        def block(index: int) -> dict:
            return blocks.setdefault(
                index, {"kind": "", "text": "", "signature": "", "tool": None, "redacted": None}
            )

        stop_reason = "end_turn"
        usage: dict = {}
        metrics: dict = {}

        response = self._client.converse_stream(**kwargs)
        for event in response["stream"]:
            if "contentBlockStart" in event:
                payload = event["contentBlockStart"]
                current = block(payload.get("contentBlockIndex", 0))
                tool_use = (payload.get("start") or {}).get("toolUse") or {}
                if tool_use:
                    current["kind"] = "toolUse"
                    current["tool"] = {
                        "toolUseId": tool_use.get("toolUseId", ""),
                        "name": tool_use.get("name", ""),
                    }
                    emit("tool", tool_use.get("name", ""))

            elif "contentBlockDelta" in event:
                payload = event["contentBlockDelta"]
                current = block(payload.get("contentBlockIndex", 0))
                delta = payload.get("delta") or {}
                if "text" in delta:
                    current["kind"] = current["kind"] or "text"
                    current["text"] += delta["text"]
                    emit("text", delta["text"])
                elif "toolUse" in delta:
                    # Tool arguments stream as a JSON string in fragments —
                    # only parseable once the block is complete.
                    current["kind"] = "toolUse"
                    fragment = delta["toolUse"].get("input", "")
                    current["text"] += fragment
                    emit("tool_input", fragment)
                elif "reasoningContent" in delta:
                    reasoning = delta["reasoningContent"]
                    current["kind"] = current["kind"] or "reasoning"
                    if "text" in reasoning:
                        current["text"] += reasoning["text"]
                        emit("reasoning", reasoning["text"])
                    elif "signature" in reasoning:
                        current["signature"] += reasoning["signature"]
                    elif "redactedContent" in reasoning:
                        current["kind"] = "redacted"
                        current["redacted"] = reasoning["redactedContent"]

            elif "messageStop" in event:
                stop_reason = event["messageStop"].get("stopReason", stop_reason)

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {}) or {}
                metrics = event["metadata"].get("metrics", {}) or {}

        content: list[dict] = []
        for index in sorted(blocks):
            current = blocks[index]
            kind = current["kind"]
            if kind == "toolUse":
                tool = current["tool"] or {"toolUseId": "", "name": ""}
                try:
                    tool_input = json.loads(current["text"]) if current["text"].strip() else {}
                except json.JSONDecodeError:
                    # A truncated response (hit maxTokens mid-arguments) can leave
                    # unparseable JSON. Pass {} through: the agents already handle a
                    # tool call with missing required args, and that path gives the
                    # model a usable error rather than crashing the whole run.
                    logger.warning(
                        "Streamed tool arguments for %s were not valid JSON: %.200s",
                        tool.get("name", "?"), current["text"],
                    )
                    tool_input = {}
                content.append({"toolUse": {**tool, "input": tool_input}})
            elif kind == "reasoning":
                reasoning_text: dict = {"text": current["text"]}
                if current["signature"]:
                    reasoning_text["signature"] = current["signature"]
                content.append({"reasoningContent": {"reasoningText": reasoning_text}})
            elif kind == "redacted":
                content.append({"reasoningContent": {"redactedContent": current["redacted"]}})
            else:
                content.append({"text": current["text"]})

        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": stop_reason,
            "usage": usage,
            "metrics": metrics,
        }


class ClaudeBedrockClient(BedrockClient):
    """
    Anthropic Claude on Bedrock.

    Reasoning is Claude "extended thinking", switched on through the `thinking`
    key — NOT `reasoning_config`, which is an Amazon convention and makes
    Bedrock raise a ValidationException on a Claude model. Two constraints come
    with it, both enforced here rather than left to blow up at call time:

      - budget_tokens must be at least 1024 and strictly less than maxTokens
      - temperature must be 1; any other value is rejected outright, so the
        agents' usual temperature=0.0 has to be overridden while thinking is on

    Claude 4.6 and newer replace the fixed budget with adaptive thinking
    (`{"type": "adaptive"}`) and an effort level. Haiku 4.5 — the model this is
    currently pointed at — predates that and takes the fixed budget below.
    """

    _MODEL_MARKERS = ("anthropic.", "claude")
    _FAMILY = "claude"

    #: Anthropic's floor for a thinking budget.
    _MIN_BUDGET_TOKENS = 1024

    #: The only temperature Claude accepts while thinking is enabled.
    _THINKING_TEMPERATURE = 1.0

    def supports_reasoning(self) -> bool:
        return True

    def _effective_budget(self) -> Optional[int]:
        """The requested budget clamped into Claude's legal range, or None."""
        requested = self.reasoning_budget_tokens
        if requested is None:
            return None
        ceiling = self.max_tokens - 1
        if ceiling < self._MIN_BUDGET_TOKENS:
            logger.warning(
                "Cannot enable Claude thinking: budget_tokens must be >= %d and "
                "< max_tokens, but max_tokens is only %d. Raise max_tokens to at "
                "least %d. Continuing without thinking.",
                self._MIN_BUDGET_TOKENS, self.max_tokens, self._MIN_BUDGET_TOKENS + 1,
            )
            return None
        budget = max(self._MIN_BUDGET_TOKENS, min(requested, ceiling))
        if budget != requested:
            logger.info(
                "Clamped Claude thinking budget %d -> %d (legal range %d..%d for "
                "max_tokens=%d).",
                requested, budget, self._MIN_BUDGET_TOKENS, ceiling, self.max_tokens,
            )
        return budget

    def _reasoning_active(self) -> bool:
        return self._effective_budget() is not None

    def _reasoning_fields(self) -> Optional[dict]:
        budget = self._effective_budget()
        if budget is None:
            return None
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}

    def _inference_config(self) -> dict:
        config = super()._inference_config()
        if self._reasoning_active() and config["temperature"] != self._THINKING_TEMPERATURE:
            logger.info(
                "Overriding temperature %.2f -> %.1f: Claude rejects any other "
                "temperature while extended thinking is enabled.",
                config["temperature"], self._THINKING_TEMPERATURE,
            )
            config["temperature"] = self._THINKING_TEMPERATURE
        return config


class NovaBedrockClient(BedrockClient):
    """
    Amazon Nova on Bedrock.

    WARNING — the reasoning shape below is UNVERIFIED. `reasoning_config` is the
    Amazon-side convention and is what this repo was already sending, but it has
    not been confirmed against a Nova model card, and reasoning is not offered
    across the whole Nova line. Verify before relying on it; if Bedrock rejects
    it, converse() raises a RuntimeError naming this class so the mismatch is
    obvious rather than silent.

    No temperature override is applied: whether Nova constrains sampling while
    reasoning is active is likewise unconfirmed, and inventing a constraint that
    doesn't exist would silently change sampling behaviour for every Nova call.
    """

    _MODEL_MARKERS = ("amazon.nova", "nova")
    _FAMILY = "nova"

    def supports_reasoning(self) -> bool:
        return True

    def _reasoning_fields(self) -> Optional[dict]:
        if self.reasoning_budget_tokens is None:
            return None
        return {
            "reasoning_config": {
                "type": "enabled",
                "budget_tokens": self.reasoning_budget_tokens,
            }
        }


# ---------------------------------------------------------------------------
# Agentic loop — Bedrock Converse
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# ReAct step event — passed to the on_step callback each iteration
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class ReActStep:
    """
    Represents one step of the ReAct loop.

    Fields:
        iteration:   1-based step counter
        thought:     Any plain text the model emitted before calling a tool
        reasoning:   The model's chain-of-thought, IF it returned a reasoningContent
                     block (only reasoning-capable models with reasoning enabled via
                     additional_model_request_fields ever populate this — otherwise "").
        tool_name:   Name of the tool being called (empty string if final answer)
        tool_input:  Arguments passed to the tool
        observation: JSON string result returned by the tool
        final:       True only on the last step that contains the final answer
        answer:      Populated on the final step with the model's answer text
        usage:            Token usage {"input_tokens","output_tokens","total_tokens"}
                          for the ONE converse() call that produced this step.
        cumulative_usage: Running total across every converse() call made so
                          far in this run (including this step's usage).
    """
    iteration: int
    thought: str = ""
    reasoning: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    observation: str = ""
    final: bool = False
    answer: str = ""
    prompt_sent: str = ""  # system prompt + last few messages sent to the model this turn
    screenshot: Optional[bytes] = None  # viewport JPEG after this tool ran — UI-only, never sent to the LLM
    usage: dict = field(default_factory=empty_usage)
    cumulative_usage: dict = field(default_factory=empty_usage)


class PlaywrightAgent:
    """
    A browser automation agent backed by AWS Bedrock running a ReAct loop.

    ReAct loop (one tool per step):
      1. Send messages to Bedrock.
      2. Model returns a thought (text) + ONE tool call.
      3. Execute that ONE tool, capture observation.
      4. Append observation to history and go to step 1.
      5. When model returns end_turn with no tool call → final answer.

    on_step callback:
      If provided, called after every step with a ReActStep dataclass so
      callers (e.g. main.py) can print/log each Think→Act→Observe cycle.

    The browser session is managed by playwright_tools._state.
    Call close() to ensure the browser is stopped on exit.
    """

    SYSTEM_PROMPT = (
        "You are a browser automation agent. You control a real web browser using tools. "
        "Always start by calling start_browser() before any page-level action. "
        "Think step by step and use the tools to accomplish the user's goal. "
        "Call get_page_structure() FIRST on any new page — it's a compact outline of roles, "
        "names, and visible TEXT (view counts, prices, dates, titles — everything readable) "
        "tagged with short refs like [ref=e12]. Pass that ref DIRECTLY as any tool's selector "
        "argument as \"aria-ref=e12\" — no need to construct or copy a CSS selector. "
        "Use search_elements(terms) to filter by guessed keywords (e.g. ['login', 'sign in']) "
        "when the page is too large to read the whole outline at once. "
        "Use expand_element(selector) to see full detail on one candidate before acting on it. "
        "get_page_structure() is the ONLY page-wide view — there is no raw-HTML tool. It is "
        "paginated via chunk_index, so nothing is ever permanently lost: page through "
        "chunk_index=1, 2, ... for more. Use get_text_blocks()/get_text() to read specific "
        "content, and expand_element(selector) for one element's full detail including its href. "
        "Prefer navigate(url) over click() when you can read an href from search/expand results. "
        "Use button:has-text('text') or text=Label when an element has no id or aria-label. "
        "Never invent URLs — only navigate to hrefs you have actually read. "
        "Call stop_browser() only when the task is fully complete, then give a final answer. "
        "NEVER give a final answer containing a value (a view count, price, date, name, etc.) "
        "you have not actually seen in a tool result in this conversation — do not invent, "
        "estimate, or recall a plausible-sounding value from general knowledge. If you can't "
        "find the data, keep searching/reading the page instead of guessing. "
        "Treat your first result with suspicion: if a search/filter result doesn't clearly "
        "match the user's intent (wrong category, wrong brand, sponsored/ad content, values "
        "outside a requested range), do not report it — refine the query or apply stricter "
        "filters/sorting and check again before finalizing. When a specific numeric claim "
        "(price, count, rating, 'cheapest'/'best') is central to the answer, cross-verify it "
        "a second way (e.g. sort by price ascending to confirm a 'cheapest' claim) before "
        "giving the final answer, and briefly note in the final answer how it was verified."
    )

    # HTML keys stripped from older message history to keep context window manageable.
    # Only the two most recent tool results keep their full HTML.
    _HTML_KEYS = {"html", "page_html"}

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_iterations: int = 30,
        system_prompt: Optional[str] = None,
        on_step: Optional[Any] = None,
        additional_model_request_fields: Optional[dict] = None,
        reasoning_budget_tokens: Optional[int] = None,
        on_delta: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Args:
            model_id: Bedrock model ID (e.g. "anthropic.claude-3-5-sonnet-20241022-v2:0")
            region: AWS region (must have Bedrock model access enabled)
            profile: AWS CLI profile name (None = use default credentials chain)
            max_tokens: Max response tokens per LLM call
            temperature: Sampling temperature
            max_iterations: Safety cap on agentic loop iterations (one tool = one iteration)
            system_prompt: Override the default system prompt
            on_step: Optional async or sync callable(ReActStep) called after every step.
                     Use this in main.py to print each Think→Act→Observe cycle.
            additional_model_request_fields: Raw Bedrock passthrough — see
                BedrockClient's docstring. Prefer reasoning_budget_tokens.
            reasoning_budget_tokens: Enable model reasoning/"thinking" (surfaced on
                each ReActStep.reasoning). Provider-neutral — the client subclass
                picked for model_id translates it into that family's actual key,
                shape, and legal range.
            on_delta: Optional callback(kind, text) fired for each chunk as the
                model produces it, so the caller can show thinking and tool
                calls live instead of only on step boundaries. Setting it turns
                on the streaming API — see BedrockClient.converse_stream().
        """
        self.llm = BedrockClient.create(
            model_id=model_id,
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
            on_delta=on_delta,
            additional_model_request_fields=additional_model_request_fields,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.tools = get_all_tool_schemas()
        self.on_step = on_step
        self._messages: list[dict] = []
        self._total_usage: dict = empty_usage()

    def reset(self):
        """Clear conversation history."""
        self._messages = []
        self._total_usage = empty_usage()

    async def _fire_step(self, step: ReActStep) -> None:
        """Call on_step callback (supports both sync and async callables)."""
        if self.on_step is None:
            return
        result = self.on_step(step)
        if asyncio.iscoroutine(result):
            await result

    def _trim_history(self, keep_full_last_n: int = 2) -> list[dict]:
        """
        The reduced history to send this turn. self._messages keeps everything;
        see compact_history() for what "reduced" means and why it is
        key-agnostic rather than a list of known page-view field names.
        """
        return compact_history(self._messages, keep_full_last_n=keep_full_last_n)

    async def run(self, user_message: str, reset: bool = True) -> str:
        """
        Run the ReAct loop on a natural-language browser task.

        Each iteration:
          - Ask the model what to do next (Think)
          - If it returns a tool call → execute ONE tool (Act) → send result back (Observe)
          - If Bedrock returns multiple tools in one response, they are each executed
            sequentially with individual Observe steps before the next Think
          - When the model returns end_turn with no tool call → final answer

        Args:
            user_message: Task description in plain English
            reset: Clear conversation history before starting

        Returns:
            Final plain-English answer from the model.
        """
        if reset:
            self.reset()

        reset_loop_detection()
        self._messages = [_user_message(user_message)]
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info("--- ReAct step %d ---", iteration)

            trimmed = self._trim_history()

            # Build a compact prompt snapshot for the UI (system + last 3 messages)
            try:
                sys_preview = (self.system_prompt or "")[:800]
                msg_preview = ""
                for m in trimmed[-3:]:
                    role = m.get("role", "")
                    for block in m.get("content", []):
                        if "text" in block:
                            msg_preview += f"\n[{role}] {block['text'][:300]}"
                        elif "toolUse" in block:
                            tu = block["toolUse"]
                            msg_preview += f"\n[{role}] TOOL {tu['name']}({json.dumps(tu.get('input',{}))[:120]})"
                        elif "toolResult" in block:
                            tr = block["toolResult"]
                            txt = (tr.get("content") or [{}])[0].get("text", "")[:200]
                            msg_preview += f"\n[{role}] RESULT {txt}"
                prompt_snapshot = f"SYSTEM:\n{sys_preview}\n\nMESSAGES (last 3):{msg_preview}"
            except Exception:
                prompt_snapshot = "(prompt capture failed)"

            response = await self.llm.converse(
                messages=trimmed,
                system=self.system_prompt,
                tools=self.tools,
            )

            output_message = response["output"]["message"]
            stop_reason = response["stopReason"]
            content_blocks: list[dict] = output_message.get("content", [])

            # One prompt = one converse() call. Track this call's tokens plus
            # the running total across the whole run.
            step_usage = usage_from_response(response)
            self._total_usage = add_usage(self._total_usage, step_usage)

            # Capture any reasoning text the model emitted alongside tool calls,
            # plus its structured chain-of-thought if reasoning is enabled (see
            # additional_model_request_fields) — kept separate from `thought` so
            # callers can render them differently (e.g. a dimmed "thinking" box).
            thought = _extract_text(content_blocks)
            reasoning = _extract_thinking(content_blocks)

            # Append full assistant turn to history
            self._messages.append({"role": "assistant", "content": content_blocks})

            # ── FINAL ANSWER ──────────────────────────────────────────────
            if stop_reason == "end_turn":
                await self._fire_step(ReActStep(
                    iteration=iteration,
                    thought=thought,
                    reasoning=reasoning,
                    final=True,
                    answer=thought,
                    prompt_sent=prompt_snapshot,
                    usage=step_usage,
                    cumulative_usage=self._total_usage,
                ))
                return thought

            # ── TOOL USE ─────────────────────────────────────────────────
            if stop_reason == "tool_use":
                tool_uses = _extract_tool_uses(content_blocks)

                # Execute tools ONE AT A TIME — core of the ReAct loop
                tool_result_blocks: list[dict] = []
                for tu in tool_uses:
                    observation = await execute_tool_call(tu["name"], tu["input"])

                    logger.info("TOOL %s(%s) -> %s",
                                tu["name"],
                                json.dumps(tu["input"])[:200],
                                observation[:300])

                    # Fire step callback so main.py can render Think→Act→Observe
                    await self._fire_step(ReActStep(
                        iteration=iteration,
                        thought=thought,
                        reasoning=reasoning,
                        tool_name=tu["name"],
                        tool_input=tu["input"],
                        observation=observation,
                        prompt_sent=prompt_snapshot,
                        screenshot=pt._state.get("last_screenshot"),
                        usage=step_usage,
                        cumulative_usage=self._total_usage,
                    ))

                    tool_result_blocks.append({
                        "toolResult": {
                            "toolUseId": tu["toolUseId"],
                            "content": [{"text": observation}],
                        }
                    })
                    # Increment iteration count per tool so the cap is per-action
                    if len(tool_uses) > 1:
                        iteration += 1

                # Send all results back in one user message (Bedrock requirement)
                self._messages.append({"role": "user", "content": tool_result_blocks})
                continue

            # ── UNEXPECTED ───────────────────────────────────────────────
            logger.warning("Unexpected stopReason: %s", stop_reason)
            return thought or f"Stopped unexpectedly: {stop_reason}"

        return "Max iterations reached without a final answer."

    async def close(self):
        """Stop the browser and clean up if still running."""
        if pt._state["browser"] is not None:
            await pt.stop_browser()


# ---------------------------------------------------------------------------
# Streaming variant — yields text from the final answer chunk by chunk
# Tool call rounds are executed normally (Bedrock streaming + tool_use is
# complex; we stream only the terminal text response for simplicity).
# ---------------------------------------------------------------------------

class StreamingPlaywrightAgent(PlaywrightAgent):
    """
    Same as PlaywrightAgent but yields the final text answer in chunks.

    Usage:
        agent = StreamingPlaywrightAgent(model_id="anthropic.claude-3-5-sonnet-20241022-v2:0")
        async for chunk in agent.stream("Go to example.com and return the h1 text."):
            print(chunk, end="", flush=True)
    """

    async def stream(self, user_message: str, reset: bool = True):
        """
        Async generator. Runs the full agentic loop silently, then yields the
        final answer as text chunks.

        Args:
            user_message: Natural language task description
            reset: Clear conversation history before starting
        """
        final_answer = await self.run(user_message, reset=reset)
        chunk_size = 20
        for i in range(0, len(final_answer), chunk_size):
            yield final_answer[i:i + chunk_size]


# ---------------------------------------------------------------------------
# Convenience: run a one-shot task from synchronous code
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    region: str = "us-east-1",
    profile: Optional[str] = None,
    headless: bool = True,
) -> str:
    """
    Synchronous entry point — run a single browser task and return the result.

    Args:
        task: Natural language task (e.g. "Go to github.com and get the page title")
        model_id: Bedrock model ID
        region: AWS region
        profile: AWS CLI profile name (None = default credentials chain)
        headless: Whether to run the browser without a visible window

    Returns:
        Final answer string from the model.

    Example:
        from llm_agent import run_task
        print(run_task("Go to https://example.com and return the h1 text."))
    """
    async def _run():
        agent = PlaywrightAgent(model_id=model_id, region=region, profile=profile)
        try:
            return await agent.run(task)
        finally:
            await agent.close()

    return asyncio.run(_run())
