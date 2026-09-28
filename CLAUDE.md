# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A browser automation agent: natural-language tasks ("go to youtube.com, search X, return the view count") are executed against a real Playwright-controlled Chrome browser by an LLM running on AWS Bedrock. There are three interchangeable agent architectures (see below) sharing the same tool layer (`playwright_tools.py`). `main.py` is a CLI front end; `app.py` is a Streamlit UI. `test-frontend/` is an unrelated Vite+React scaffold and not part of the agent.

## Commands

```bash
# Install (venv already exists at ./venv)
./venv/Scripts/pip install boto3 playwright python-dotenv streamlit
./venv/Scripts/playwright install chromium   # or rely on channel="chrome" (system Chrome)

# Configure — copy .env.example to .env and set BEDROCK_MODEL_ID / AWS_DEFAULT_REGION / AWS_PROFILE

# CLI, single task
python main.py --task "Go to example.com and return the h1 text."
python main.py --orchestrate --task "..."   # multi-agent planner+specialist pipeline
python main.py --guided --task "..."        # ThinkingAgent+ActorAgent pipeline
python main.py                               # interactive REPL (type "tools" or "exit")
python main.py --list-tools                  # print every registered tool + required args
python main.py --no-headless --task "..."    # show the browser window
python main.py --model "amazon.nova-pro-v1:0" --region "us-west-2" --task "..."

# Streamlit UI
streamlit run app.py

# Quick Bedrock connectivity/model check (not part of the agent)
python aws_model_smoke_test.py --model-id anthropic.claude-3-5-sonnet-20241022-v2:0
python aws_model_smoke_test.py --all-models
```

There is no test suite, linter, or build step configured for the Python code (`_tmp_test_planner_search.py` is a manual scratch script, not part of a pytest suite). `test-frontend/` has its own independent `npm run lint` / `npm run build` (Vite/Oxlint) but is unrelated to the agent.

## Architecture

### The three agent modes share one tool layer

`playwright_tools.py` defines ~60 standalone `async def` functions (navigate, click, fill, search_elements, assert_visible, etc.) operating on a single module-level `_state` dict (`playwright`, `browser`, `context`, `page`, `page_index`). There is one global browser session — not per-agent-instance — so only one task should run at a time per process. Every public async function in this module is auto-registered as an LLM tool; adding a new `async def foo(...)` to `playwright_tools.py` with a Google-style docstring is enough to expose it to all three agents (schema is generated from the signature + docstring by `llm_agent.build_tool_schema`).

`llm_agent.execute_tool_call()` is the single chokepoint all three agents call through (directly, or via `orchestrator.ToolExecutorAgent`). It adds cross-cutting behavior no agent needs to know about:
- pre-flight selector validation (`_validate_selector`) for tools in `_SELECTOR_TOOLS` — checks the selector resolves to exactly one visible, enabled element before letting Playwright touch it, and returns page HTML + live element list in the error so the agent can self-correct instead of hitting a raw Playwright exception
- auto-appending fresh page HTML/URL/title after tools in `_PAGE_MUTATING_TOOLS`
- empty-result warnings for read tools (`get_text`/`get_all_text`) that likely hit the wrong page
- fill/type_text write verification, feeding `selector_cache.py` (a per-domain JSON cache of selectors known to work/fail, persisted across runs in `selector_cache.json` and injected into prompts as hints)

`search_elements`/`expand_element`/`_build_page_index` build an indexed, ranked view of the page (cached in `_state["page_index"]`, invalidated by `_invalidate_index()` after any mutating tool) so agents can find elements by fuzzy keyword instead of reading full HTML — this is the preferred discovery path over `get_page_html`/`get_page_snapshot` in every agent's prompt.

### Mode 1 — ReAct (`llm_agent.PlaywrightAgent`, default)
Single LLM role, one continuously growing Bedrock message history, all ~60 tools available every turn. Loop: send full history → model emits thought + one tool call → execute → append observation → repeat until `end_turn`. Simplest and most token-hungry mode; `_trim_history()` strips HTML from all but the last 2 tool results to bound context growth. `ReActStep` is fired via `on_step` after every step for rendering (CLI/Streamlit).

### Mode 2 — Orchestrator (`orchestrator.OrchestratorAgent`, `--orchestrate`)
Multi-agent pipeline, stateless per-call (no growing history — each LLM call gets a freshly built prompt):
1. **PlanningAgent** decomposes the task into an ordered list of subtasks, each tagged with a **tool group** (see `TOOL_GROUPS` — navigation, discovery, form_input, clicking, waiting, network, tabs_frames, storage, capture, assertions, debug). `discovery` merges finding a selector (search_elements/expand_element) with reading page content (get_page_html/get_page_structure/get_text/...) so a single step is never stranded needing both. Called once upfront and again after every subtask finishes/fails, given the live page state — this is a *replan-after-every-step* design, not a fixed upfront plan.
2. **SpecialistAgent.act()** — one class parameterized by group at call time. Every call does BOTH evaluate-the-previous-result AND pick-the-next-tool in a single LLM round trip (self-evaluates against fresh live URL/title, not the tool's own self-reported status, to reduce self-grading bias). Only sees tools in its assigned group.
3. **ToolExecutorAgent** — no LLM, just runs the chosen tool via `execute_tool_call`.
4. `_check_hard_stop()` — deterministic (non-LLM) loop and selector-error-storm detection that runs after every execution; a model can never argue its way past this. This is intentionally kept out of the LLM-judged evaluation in step 2.

Fires `OrchestratorEvent`s (`plan`, `step_start`, `loop_select`, `loop_execute`, `loop_eval`, `step_done`, `step_failed`, `done`) for rendering.

### Mode 3 — Guided (`guided_agent.GuidedAgent`, `--guided`)
Two LLM roles, tighter per-*action* granularity than Orchestrator's per-*step* planning (closer to ReAct), no upfront plan:
- **ThinkingAgent** — stateful, a real growing conversation like ReAct's, but restricted to READ-ONLY perception tools (`search_elements`, `expand_element`, `get_page_structure`, `get_current_url`) plus a special `decide_action` hand-off tool. Gathers info freely, then hands off exactly one instruction + target selector, or sets `task_complete`.
- **ActorAgent** — stateless, one call per action. Never decides *what* to do, only translates the ThinkingAgent's natural-language instruction into an exact `tool_name`/`tool_args` from the full flat tool list — always an LLM call even when the mapping looks mechanical (deliberate design choice, not an oversight).

Trade-off vs. Orchestrator: 2 LLM calls per action instead of 1, in exchange for a continuously-adaptive loop with no plan to go stale.

### Choosing a mode when changing code
When touching agent logic, check whether the change belongs in `playwright_tools.py` (a new capability, available to all three modes for free) vs. one specific agent file — the three modes are meant to stay architecturally distinct (single-agent vs. plan+specialist vs. think+act), so don't collapse shared logic across them beyond what already lives in `llm_agent.execute_tool_call`/`_validate_selector` and `orchestrator.ToolExecutorAgent`.

### Bedrock plumbing
`llm_agent.BedrockClient` wraps `boto3`'s synchronous `bedrock-runtime` client's `converse()` API, offloaded to a thread pool via `run_in_executor` since boto3 has no native asyncio support. All three agents build tool schemas via `get_all_tool_schemas()`/`build_tool_schema()`, which reflects Python function signatures + Google-style docstrings into Bedrock's `toolSpec` JSON schema format. Credentials resolve through the standard boto3 chain (env vars, `AWS_PROFILE`, or IAM role) — never hardcode keys.
