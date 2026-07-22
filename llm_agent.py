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
from typing import Any, Optional

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
    description = doc.split("\n\n")[0].replace("\n", " ").strip()
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

# Tools whose empty result is a sign the selector matched the wrong page
_EMPTY_RESULT_CHECKS: dict = {
    "get_all_text": lambda r: not r.get("texts") or not any(t.strip() for t in r["texts"]),
    "get_text":     lambda r: not (r.get("text") or "").strip(),
}


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
        # only reject if a completely different overlay element is on top)
        if valid and tool_name == "click":
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
        # Only persist truly absent selectors — not timing/hidden failures
        if reason == "does not exist":
            try:
                sc.save_failed_selector(pt._state["page"].url, selector)
            except Exception:
                pass

        # Build error message with partial degradation — each section independent
        parts = [
            f"SELECTOR ERROR: '{selector}' cannot be interacted with ({reason}).",
            "Do NOT use this selector again.",
        ]

        try:
            page_html = await pt.get_page_html(max_chars=8000)
            parts.append(
                f"\nCURRENT PAGE: {page_html['url']}  |  Title: {page_html['title']}"
            )
            parts.append(f"\nCURRENT PAGE HTML:\n{page_html['html']}")
        except Exception as e:
            parts.append(f"\n(page HTML unavailable: {e})")

        try:
            snapshot = await pt.get_page_snapshot()
            sel_lines = []
            for el in snapshot.get("elements", []):
                s = el.get("selector", "")
                t = el.get("tag", "")
                lbl = (el.get("aria_label") or el.get("placeholder") or
                       el.get("name") or el.get("text") or "")
                if s:
                    sel_lines.append(f"  {s}  ({t} — {lbl[:60]})")
            if sel_lines:
                parts.append("\nVISIBLE ELEMENTS:\n" + "\n".join(sel_lines))
        except Exception as e:
            parts.append(f"\n(element snapshot unavailable: {e})")

        return "\n".join(parts)

    return None


async def execute_tool_call(tool_name: str, tool_input: dict) -> str:
    """
    Execute a playwright tool by name with the input dict from Bedrock.

    Before running selector-based tools, validates the selector exists on the
    current page. If it doesn't, returns an error with the current page HTML
    so the agent can immediately pick a correct selector.

    For page-mutating tools, automatically appends get_page_html() to the result
    so the agent always sees the updated page structure after every action.
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

        # Auto-append page state for mutating tools that don't already include it
        if (
            tool_name in _PAGE_MUTATING_TOOLS
            and isinstance(result, dict)
            and "error" not in result
            and "html" not in result
            and pt._state["page"] is not None
        ):
            try:
                page_html = await pt.get_page_html()
                result["url"]        = page_html["url"]
                result["title"]      = page_html["title"]
                result["page_html"]  = page_html["html"]
            except Exception:
                pass

        # Warn when a reading tool returns empty — likely wrong page or wrong selector
        empty_check = _EMPTY_RESULT_CHECKS.get(tool_name)
        if empty_check and isinstance(result, dict) and "error" not in result and empty_check(result):
            try:
                ph   = await pt.get_page_html()
                snap = await pt.get_page_snapshot()
                result["empty_result_warning"] = (
                    f"Selector '{tool_input.get('selector')}' returned nothing on "
                    f"{ph['url']} (title: {ph['title']}). "
                    f"The element may not exist on this page."
                )
                result["page_url"]    = ph["url"]
                result["page_title"]  = ph["title"]
                result["page_html"]   = ph["html"]
                result["elements"]    = snap.get("elements", [])
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
                        # Save as a working selector for this domain
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


def _extract_text(content_blocks: list[dict]) -> str:
    """Extract all text from a Bedrock assistant content block list."""
    return " ".join(
        block["text"] for block in content_blocks if "text" in block
    ).strip()


def _extract_tool_uses(content_blocks: list[dict]) -> list[dict]:
    """Extract all toolUse blocks from a Bedrock assistant content block list."""
    return [block["toolUse"] for block in content_blocks if "toolUse" in block]


# ---------------------------------------------------------------------------
# LLM client — AWS Bedrock Converse API
# ---------------------------------------------------------------------------

class BedrockClient:
    """
    Async-compatible wrapper around the AWS Bedrock Converse API (boto3).

    boto3 is synchronous, so calls are offloaded to a thread pool via
    asyncio.get_event_loop().run_in_executor() to avoid blocking the event loop.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        profile: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ):
        """
        Args:
            model_id: Bedrock model ID.
                      Examples:
                        "anthropic.claude-3-5-sonnet-20241022-v2:0"
                        "anthropic.claude-3-haiku-20240307-v1:0"
                        "amazon.nova-pro-v1:0"
                        "amazon.nova-lite-v1:0"
                        "us.amazon.nova-pro-v1:0"  ← cross-region inference prefix
            region: AWS region where Bedrock is enabled (e.g. "us-east-1")
            profile: AWS CLI profile name (None = use default/env vars)
            max_tokens: Max tokens in the response
            temperature: Sampling temperature (0 = deterministic)
        """
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self._client = session.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def converse(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """
        Call the Bedrock Converse API asynchronously.

        Args:
            messages: List of Bedrock-format message dicts
            system: Optional system prompt string
            tools: Optional list of Bedrock toolSpec dicts

        Returns:
            The raw Bedrock converse() response dict.
        """
        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if system:
            kwargs["system"] = [{"text": system}]
        if tools:
            kwargs["toolConfig"] = {
                "tools": tools,
                "toolChoice": {"auto": {}},
            }

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._client.converse(**kwargs)
        )


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
        tool_name:   Name of the tool being called (empty string if final answer)
        tool_input:  Arguments passed to the tool
        observation: JSON string result returned by the tool
        final:       True only on the last step that contains the final answer
        answer:      Populated on the final step with the model's answer text
    """
    iteration: int
    thought: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    observation: str = ""
    final: bool = False
    answer: str = ""
    prompt_sent: str = ""  # system prompt + last few messages sent to the model this turn


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
        "The cleaned page HTML is included in tool results — derive selectors from it directly. "
        "Prefer navigate(url) over click() when you can read an href from the HTML. "
        "Use button:has-text('text') or text=Label when an element has no id or aria-label. "
        "Never invent URLs — only navigate to hrefs you have read from the page HTML. "
        "Call stop_browser() only when the task is fully complete, then give a final answer."
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
        """
        self.llm = BedrockClient(
            model_id=model_id,
            region=region,
            profile=profile,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.tools = get_all_tool_schemas()
        self.on_step = on_step
        self._messages: list[dict] = []

    def reset(self):
        """Clear conversation history."""
        self._messages = []

    async def _fire_step(self, step: ReActStep) -> None:
        """Call on_step callback (supports both sync and async callables)."""
        if self.on_step is None:
            return
        result = self.on_step(step)
        if asyncio.iscoroutine(result):
            await result

    def _trim_history(self, keep_html_in_last_n: int = 2) -> list[dict]:
        # Collect indices of user messages that carry toolResult blocks
        tool_result_indices = [
            i for i, m in enumerate(self._messages)
            if m.get("role") == "user"
            and any("toolResult" in block for block in m.get("content", []))
        ]
        # Strip HTML from all but the last N of those
        strip_at = set(tool_result_indices[:-keep_html_in_last_n]) if len(tool_result_indices) > keep_html_in_last_n else set()

        if not strip_at:
            return self._messages  # nothing to trim yet

        trimmed = []
        for i, msg in enumerate(self._messages):
            if i not in strip_at:
                trimmed.append(msg)
                continue
            # Strip html keys from each toolResult content block
            new_content = []
            for block in msg.get("content", []):
                if "toolResult" not in block:
                    new_content.append(block)
                    continue
                tr = block["toolResult"]
                new_tr_content = []
                for item in tr.get("content", []):
                    if "text" not in item:
                        new_tr_content.append(item)
                        continue
                    try:
                        parsed = json.loads(item["text"])
                        cleaned = {k: v for k, v in parsed.items() if k not in self._HTML_KEYS}
                        new_tr_content.append({"text": json.dumps(cleaned)})
                    except (json.JSONDecodeError, TypeError):
                        new_tr_content.append(item)
                new_content.append({"toolResult": {**tr, "content": new_tr_content}})
            trimmed.append({**msg, "content": new_content})
        return trimmed

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

            # Capture any reasoning text the model emitted alongside tool calls
            thought = _extract_text(content_blocks)

            # Append full assistant turn to history
            self._messages.append({"role": "assistant", "content": content_blocks})

            # ── FINAL ANSWER ──────────────────────────────────────────────
            if stop_reason == "end_turn":
                await self._fire_step(ReActStep(
                    iteration=iteration,
                    thought=thought,
                    final=True,
                    answer=thought,
                    prompt_sent=prompt_snapshot,
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
                        tool_name=tu["name"],
                        tool_input=tu["input"],
                        observation=observation,
                        prompt_sent=prompt_snapshot,
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
