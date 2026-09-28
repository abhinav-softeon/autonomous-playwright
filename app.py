"""
Streamlit UI — Playwright Browser Agent
----------------------------------------
Default mode: single-agent ReAct loop (PlaywrightAgent) — full history,
              all tools, derives selectors from live HTML.
Orchestrate:  multi-agent pipeline via sidebar toggle.
"""

import asyncio
import html as html_lib
import json
import os
import queue
import threading
import time

from dotenv import load_dotenv
import streamlit as st

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)

def _env_bool(key: str, default: bool = True) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes")

def _env_int(key: str, default: int = 0) -> int:
    try: return int(_env(key, str(default)))
    except ValueError: return default

def _env_float(key: str, default: float = 0.0) -> float:
    try: return float(_env(key, str(default)))
    except ValueError: return default


from llm_agent import PlaywrightAgent, ReActStep, get_all_tool_schemas
import playwright_tools as pt
from orchestrator import (
    OrchestratorAgent, OrchestratorEvent,
    StepStatus, LoopStatus, LoopIteration, PlanStep,
)
from guided_agent import DeliberationConfig, GuidedAgent, GuidedEvent

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Playwright Agent", page_icon="🎭",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.obs-box { background:#1e1e1e; color:#d4d4d4; padding:0.5rem 0.8rem; border-radius:6px;
           font-family:monospace; font-size:0.8rem; white-space:pre-wrap;
           max-height:200px; overflow-y:auto; margin-top:0.3rem; }
.tag  { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.78rem; }
.tag-think  { background:#ede9fe; color:#6d28d9; }
.tag-act    { background:#dbeafe; color:#1d4ed8; }
.tag-obs    { background:#dcfce7; color:#166534; }
.tag-select { background:#dbeafe; color:#1d4ed8; }
.tag-eval   { background:#fef9c3; color:#854d0e; }
.tag-goal   { background:#ede9fe; color:#6d28d9; }
.tag-veto   { background:#fee2e2; color:#b91c1c; }
.tag-pass   { background:#dcfce7; color:#166534; }
.tag-warn   { background:#fef3c7; color:#92400e; }
.eval-done  { color:#27ae60; font-weight:600; }
.eval-retry { color:#e67e22; font-weight:600; }
.eval-failed{ color:#e74c3c; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for k, v in {
    "running": False, "steps": [], "final_answer": "",
    "q": None, "error": "", "mode": "react", "task_status": None,
    # Text streamed from the LLM turn currently in flight. Rendered below the
    # finished steps and cleared the moment that turn lands as a real step.
    "live": "",
    "notice": "",   # sticky infrastructure message, e.g. "streaming unavailable"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_output_state() -> None:
    """Clear prior run output so each new task starts from a clean UI."""
    st.session_state.steps = []
    st.session_state.final_answer = ""
    st.session_state.error = ""
    st.session_state.task_status = None
    st.session_state.live = ""
    st.session_state.notice = ""
    st.session_state.q = None

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎭 Playwright Agent")

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        st.success("`.env` loaded", icon="✅")
    else:
        st.warning("No `.env` — copy `.env.example` → `.env`", icon="⚠️")

    st.divider()
    st.subheader("Mode")
    mode = st.radio(
        "Agent mode",
        ["ReAct (recommended)", "Orchestrator", "Guided"],
        index=0,
        help=(
            "ReAct: single agent, full history. "
            "Orchestrator: plans steps then a per-group specialist executes each. "
            "Guided: a thinking agent (search/structure only) decides one action at a "
            "time, a separate actor agent executes it — no upfront plan."
        ),
    )
    use_orchestrate = mode == "Orchestrator"
    use_guided = mode == "Guided"

    st.divider()
    st.subheader("Model")
    _opts = [
        "amazon.nova-pro-v1:0",
        "us.amazon.nova-2-lite-v1:0",
        "us.anthropic.claude-haiku-4-5:20251001-v1:0",
    ]
    _default = _env("BEDROCK_MODEL_ID", _opts[0])
    _idx = _opts.index(_default) if _default in _opts else 0
    model_preset = st.selectbox("Quick select", options=["(custom)"] + _opts,
                                index=0 if _default not in _opts else _opts.index(_default) + 1)
    model_id = st.text_input(
        "Model ID (edit directly)",
        value=_default if model_preset == "(custom)" else model_preset,
        help="Paste any Bedrock model ID. Must be enabled in your region.",
    )
    region     = st.text_input("AWS region", value=_env("AWS_DEFAULT_REGION", "us-east-1"))
    profile    = st.text_input("AWS profile (optional)", value=_env("AWS_PROFILE", ""), placeholder="default")

    st.divider()
    st.subheader("Execution")
    max_iters  = st.slider("Max iterations", 5, 40, _env_int("MAX_RETRIES_PER_STEP", 20))
    max_tokens = st.slider("Max tokens", 512, 8192, _env_int("MAX_TOKENS", 4096), step=256)
    temperature = st.slider("Temperature", 0.0, 1.0, _env_float("TEMPERATURE", 0.0), step=0.05)
    enable_reasoning = st.checkbox(
        "Enable extended reasoning", value=False,
        help="Turns on model 'thinking' in every mode. The right Bedrock key and "
             "shape is chosen from the model ID (Claude uses 'thinking', Nova uses "
             "'reasoning_config'), so you don't have to match it by hand. Claude also "
             "requires temperature=1 while thinking, which is applied automatically. "
             "When it works, each step shows a THINK box above the action.",
    )
    reasoning_budget = st.slider(
        "Reasoning budget (tokens)", 1024, 8192, 2048, step=256,
        disabled=not enable_reasoning,
        help="Must be less than Max tokens — it is clamped into the model's legal "
             "range automatically, and reasoning is skipped with a warning if Max "
             "tokens leaves no room for it.",
    )
    # Guided-mode only: these layers live in the ThinkingAgent, which the other
    # two modes don't have. Defaults still defined for every mode so cfg is uniform.
    deliberate, vote_k, use_critic = True, 1, False
    if use_guided:
        st.divider()
        st.subheader("Deliberation")
        st.caption(
            "Rebuilds in code the weighing-up that large reasoning models do "
            "natively. The first is free; the other two multiply cost per action."
        )
        deliberate = st.checkbox(
            "Force candidate comparison + outcome prediction", value=True,
            help="Makes decide_action's schema demand at least two weighed candidates "
                 "and a prediction of what the action will do. The prediction is then "
                 "checked against the real page in code, and any mismatch is fed back. "
                 "No extra LLM calls.",
        )
        vote_k = st.slider(
            "Self-consistency samples", 1, 5, 1,
            help="Sample each decision this many times and take the majority target. "
                 "1 disables it. Costs roughly this many times the decision step, and "
                 "raises the thinking temperature to 0.7 so the samples actually differ.",
        )
        use_critic = st.checkbox(
            "Critic pass before each action", value=False,
            help="A second agent argues the chosen action is wrong before it runs; a "
                 "veto sends it back once. Adds one LLM call per action.",
        )

    st.divider()
    headless   = st.checkbox("Headless browser", value=_env_bool("HEADLESS", True))
    show_raw   = st.checkbox("Show full tool output (raw JSON)", value=False,
                              help="Off: each tool call shows a one-line summary, like a live "
                                   "console. On: shows the full JSON result under each call.")
    show_screenshots = st.checkbox("Show screenshots", value=True,
                                    help="Show a screenshot of the page after each tool call.")
    show_cursor = st.checkbox("Draw mouse pointer", value=True,
                              help="Draw a fake mouse pointer into the page and move it onto "
                                   "the element each action targets, so the screenshots show "
                                   "where the agent clicked or typed. Adds ~0.3s per action.")
    stream_output = st.checkbox("Stream model output", value=True,
                                help="Show thinking and tool calls token by token as the model "
                                     "produces them, instead of only when each turn completes.")

    st.divider()
    with st.expander("Registered tools"):
        for s in get_all_tool_schemas():
            spec = s["toolSpec"]
            req  = spec["inputSchema"]["json"].get("required", [])
            st.markdown(f"**`{spec['name']}`** — {', '.join(req) or '(none)'}")


# ── main area ─────────────────────────────────────────────────────────────────
st.header("Task")

examples = [
    "Custom task…",
    "Go to https://example.com and return the h1 text.",
    "Go to https://news.ycombinator.com and list the top 5 story titles.",
    "Go to https://www.youtube.com, search for 'github tutorial', return the first video name and view count.",
    "Go to https://httpbin.org/get and return the origin IP address.",
]
example = st.selectbox("Quick examples", examples, index=0)
task_input = st.text_area(
    "Describe what the agent should do",
    value="" if example == "Custom task…" else example,
    height=110,
    placeholder="e.g. Go to youtube.com, search 'github tutorial', return first video name and view count.",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_btn = st.button("▶  Run", type="primary",
                        disabled=st.session_state.running or not task_input.strip(),
                        use_container_width=True)
with col_clear:
    if st.button("✕  Clear", disabled=st.session_state.running):
        _reset_output_state()
        st.rerun()


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(raw: str) -> str:
    try:
        obj = json.loads(raw)
        display = {k: v for k, v in obj.items() if k not in ("html", "page_html", "elements")}
        text = json.dumps(display, indent=2, ensure_ascii=False)
    except Exception:
        text = raw[:1500]
    return html_lib.escape(text)

def _trunc(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + " [...]"

def _obs_summary(raw: str, n: int = 160) -> str:
    """Condense a tool observation into one line for the streaming log view."""
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return _trunc(str(obj), n)
        if obj.get("error"):
            return f"error: {_trunc(str(obj['error']), n)}"
        for key in ("answer", "text", "value", "title", "url", "checked", "selected", "status"):
            if obj.get(key):
                return _trunc(f"{key}: {obj[key]}", n)
        display = {k: v for k, v in obj.items()
                   if k not in ("html", "page_html", "elements", "structure")}
        return _trunc(json.dumps(display, ensure_ascii=False), n) if display else "ok"
    except Exception:
        return _trunc(raw, n)


def _fmt_usage(usage: dict) -> str:
    """Compact one-line token count for a single LLM call/running total."""
    u = usage or {}
    return (f"in={u.get('input_tokens', 0)} out={u.get('output_tokens', 0)} "
            f"total={u.get('total_tokens', 0)}")


def _usage_caption(usage: dict) -> None:
    """Small caption showing the token cost of the LLM call behind one line."""
    if usage:
        st.caption(f"🔢 {_fmt_usage(usage)}")


# ── background runners ────────────────────────────────────────────────────────

def _make_delta_pusher(cfg: dict, q: queue.Queue):
    """
    on_delta callback that forwards each streamed chunk to the render queue,
    or None when streaming is switched off.

    Bedrock's event stream is drained on a boto3 worker thread, so this is
    called off both the Streamlit thread and the agent's thread — queue.Queue
    is the safe hand-off (Streamlit session state is not thread-safe).
    """
    if not cfg.get("stream"):
        return None

    def on_delta(kind: str, text: str) -> None:
        q.put(("delta", (kind, text)))

    return on_delta


def _apply_browser_display_cfg(cfg: dict) -> None:
    """Push the UI's cursor/screenshot preferences onto the tool layer's state."""
    pt._state["show_cursor"] = bool(cfg.get("show_cursor", True))


def _run_react(task: str, cfg: dict, q: queue.Queue) -> None:
    """Run PlaywrightAgent in a background thread, push ReActStep to queue."""
    def on_step(step: ReActStep):
        q.put(("react", step))

    _apply_browser_display_cfg(cfg)

    async def _go():
        agent = PlaywrightAgent(
            model_id=cfg["model_id"], region=cfg["region"],
            profile=cfg["profile"] or None,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            max_iterations=cfg["max_iters"], on_step=on_step,
            reasoning_budget_tokens=cfg.get("reasoning_budget"),
            on_delta=_make_delta_pusher(cfg, q),
        )
        try:
            return await agent.run(task)
        finally:
            if pt._state["browser"]:
                await pt.stop_browser()

    try:
        asyncio.run(_go())
    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)


def _run_orchestrator(task: str, cfg: dict, q: queue.Queue) -> None:
    """Run OrchestratorAgent in a background thread, push OrchestratorEvent to queue."""
    def on_event(event: OrchestratorEvent):
        q.put(("orch", event))

    _apply_browser_display_cfg(cfg)

    async def _go():
        agent = OrchestratorAgent(
            model_id=cfg["model_id"], region=cfg["region"],
            profile=cfg["profile"] or None,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            max_retries_per_step=cfg["max_iters"], on_event=on_event,
            reasoning_budget_tokens=cfg.get("reasoning_budget"),
            on_delta=_make_delta_pusher(cfg, q),
        )
        return await agent.run(task)

    try:
        asyncio.run(_go())
    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)


def _run_guided(task: str, cfg: dict, q: queue.Queue) -> None:
    """Run GuidedAgent in a background thread, push GuidedEvent to queue."""
    def on_event(event: GuidedEvent):
        q.put(("guided", event))

    _apply_browser_display_cfg(cfg)

    async def _go():
        agent = GuidedAgent(
            model_id=cfg["model_id"], region=cfg["region"],
            profile=cfg["profile"] or None,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            max_iterations=cfg["max_iters"] * 5, on_event=on_event,
            reasoning_budget_tokens=cfg.get("reasoning_budget"),
            on_delta=_make_delta_pusher(cfg, q),
            deliberation=DeliberationConfig(
                forced_schema=cfg.get("deliberate", True),
                predict_verify=cfg.get("deliberate", True),
                self_consistency_k=cfg.get("vote_k", 1),
                critic=cfg.get("critic", False),
            ),
        )
        return await agent.run(task)

    try:
        asyncio.run(_go())
    except Exception as exc:
        q.put(exc)
    finally:
        q.put(None)


# ── trigger ───────────────────────────────────────────────────────────────────
if run_btn and task_input.strip():
    st.session_state.running = True
    _reset_output_state()
    st.session_state.mode    = "guided" if use_guided else ("orch" if use_orchestrate else "react")

    q: queue.Queue = queue.Queue()
    st.session_state.q = q
    cfg = dict(model_id=model_id, region=region, profile=profile,
               max_tokens=max_tokens, temperature=temperature, max_iters=max_iters,
               reasoning=enable_reasoning,
               reasoning_budget=(reasoning_budget if enable_reasoning else None),
               deliberate=deliberate, vote_k=vote_k, critic=use_critic,
               show_cursor=show_cursor, stream=stream_output)
    runner = _run_guided if use_guided else (_run_orchestrator if use_orchestrate else _run_react)
    threading.Thread(target=runner, args=(task_input, cfg, q), daemon=True).start()
    st.rerun()


# ── drain queue ───────────────────────────────────────────────────────────────
if st.session_state.running and st.session_state.q:
    q: queue.Queue = st.session_state.q
    done = False
    new_items = []

    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break

        if item is None:
            done = True
            # ReAct: done=True means final step was already captured
            if st.session_state.mode == "react" and st.session_state.final_answer:
                st.session_state.task_status = "pass"
            break
        if isinstance(item, Exception):
            st.session_state.error       = str(item)
            st.session_state.task_status = "error"
            st.session_state.running     = False
            done = True
            break

        kind, payload = item
        if kind == "delta":
            # A chunk of the turn in flight. Tool arguments arrive as JSON
            # fragments, so they're appended raw and only become a rendered
            # ACT line once the turn completes and lands as a step below.
            delta_kind, text = payload
            if delta_kind == "notice":
                # Infrastructure message (e.g. streaming not permitted for this
                # model), not model output — keep it after the run, unlike `live`.
                st.session_state.notice = text
            elif delta_kind == "tool":
                st.session_state.live += f"\n▸ {text}("
            else:
                st.session_state.live += text
            continue
        # Any structured item means the streamed turn is now rendered properly.
        st.session_state.live = ""
        if kind == "react":
            step: ReActStep = payload
            if step.final:
                st.session_state.final_answer = step.answer
                st.session_state.task_status  = "pass"
            new_items.append(("react", step))
        elif kind == "orch":
            event: OrchestratorEvent = payload
            if event.event_type == "done":
                st.session_state.final_answer = event.final_answer
                # Pass if at least one subtask succeeded
                failed = sum(1 for s in event.state.plan if s.status == StepStatus.FAILED)
                total  = len(event.state.plan)
                st.session_state.task_status = "pass" if failed < total else "fail"
                done = True
            new_items.append(("orch", event))
        elif kind == "guided":
            event: GuidedEvent = payload
            if event.event_type == "done":
                st.session_state.final_answer = event.final_answer
                st.session_state.task_status  = "pass"
                done = True
            elif event.event_type == "failed":
                st.session_state.final_answer = event.final_answer
                st.session_state.task_status  = "fail"
                done = True
            new_items.append(("guided", event))

    st.session_state.steps.extend(new_items)
    if done:
        st.session_state.running = False
        st.session_state.q = None
        st.session_state.live = ""
    # Deliberately NO st.rerun() here. It raises RerunException, which aborts
    # the script on the spot — rerunning from this point meant the render block
    # below was never reached while the agent was running, so the whole trace
    # appeared in one lump at the end instead of a step at a time. The refresh
    # now happens at the very bottom of the file, after rendering.


# ── render — live streaming tool-call feed ─────────────────────────────────────
steps = st.session_state.steps
_MODE_LABEL = {"react": "ReAct", "orch": "Orchestrator", "guided": "Guided"}

if steps or st.session_state.running:
    mode_label = _MODE_LABEL.get(st.session_state.mode, "Agent")

    if st.session_state.running:
        status_state, status_label = "running", f"🔄 {mode_label} agent running…"
    elif st.session_state.error:
        status_state, status_label = "error", "✗ Error"
    elif st.session_state.task_status == "pass":
        status_state, status_label = "complete", "✓ Task completed"
    elif st.session_state.task_status == "fail":
        status_state, status_label = "error", "✗ Task incomplete"
    else:
        status_state, status_label = "complete", f"{mode_label} agent"

    # Auto-collapse only on a clean success — stay open while running or on error/incomplete.
    expanded = st.session_state.running or status_state != "complete"

    # ── running token total, read off the last event's cumulative_usage ────
    running_usage = {}
    for _, item in reversed(steps):
        u = getattr(item, "cumulative_usage", None)
        if u:
            running_usage = u
            break
    tok_cols = st.columns(3)
    tok_cols[0].metric("Input tokens", running_usage.get("input_tokens", 0))
    tok_cols[1].metric("Output tokens", running_usage.get("output_tokens", 0))
    tok_cols[2].metric("Total tokens", running_usage.get("total_tokens", 0))

    def _act_line(tag_cls: str, label: str, tool_name: str, args: dict, n: int = 120) -> None:
        args_str = html_lib.escape(_trunc(json.dumps(args, ensure_ascii=False), n))
        st.markdown(
            f'<span class="tag {tag_cls}">{label}</span> <code>{tool_name}({args_str})</code>',
            unsafe_allow_html=True,
        )

    def _obs_line(observation: str, screenshot: bytes = None) -> None:
        if observation:
            if show_raw:
                st.markdown(f'<div class="obs-box">{_fmt(observation)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="tag tag-obs">→</span> {html_lib.escape(_obs_summary(observation))}',
                    unsafe_allow_html=True,
                )
        if show_screenshots and screenshot:
            st.image(screenshot, width=320)

    if st.session_state.notice:
        st.caption(f"ℹ️ {st.session_state.notice}")

    with st.status(status_label, state=status_state, expanded=expanded):
        if st.session_state.mode == "react":
            # ── ReAct: one tool call after another, in the order they happened ──
            for _, step in steps:
                if step.final:
                    continue
                if step.reasoning:
                    st.markdown(
                        f'<span class="tag tag-think">THINK</span> '
                        f'{html_lib.escape(_trunc(step.reasoning, 400))}',
                        unsafe_allow_html=True,
                    )
                if step.thought:
                    st.caption(_trunc(step.thought, 160))
                _act_line("tag-act", "ACT", step.tool_name, step.tool_input)
                _obs_line(step.observation, step.screenshot)
                _usage_caption(step.usage)

        elif st.session_state.mode == "orch":
            # ── Orchestrator: plan, then every select/execute/eval as it streams ──
            orch_events = [e for k, e in steps if k == "orch"]
            for e in orch_events:
                if e.event_type == "plan":
                    st.markdown(f"**📋 Plan — {len(e.state.plan)} step(s)**")
                    for s in e.state.plan:
                        st.caption(f"{s.index}. {s.description}  ({s.group})")
                    _usage_caption(e.usage)
                elif e.event_type == "step_start":
                    st.markdown(f"---\n**▶ Subtask {e.step.index}: {e.step.description}**")
                elif e.event_type == "loop_select":
                    it = e.iteration
                    _act_line("tag-select", "ACT", it.tool_name, it.tool_args)
                    if it.selection_reasoning:
                        st.caption(_trunc(it.selection_reasoning, 140))
                    _usage_caption(e.usage)
                elif e.event_type == "loop_execute":
                    _obs_line(e.iteration.observation, e.iteration.screenshot)
                elif e.event_type == "loop_eval":
                    it = e.iteration
                    ecls  = {"done": "eval-done", "retry": "eval-retry", "failed": "eval-failed"}.get(it.eval_status.value, "")
                    eicon = {"done": "✓", "retry": "↺", "failed": "✗"}.get(it.eval_status.value, "?")
                    st.markdown(
                        f'<span class="tag tag-eval">EVAL</span> <span class="{ecls}">{eicon} {it.eval_status.value.upper()}</span>'
                        f' — {html_lib.escape(_trunc(it.eval_reason, 120))}',
                        unsafe_allow_html=True,
                    )
                    _usage_caption(e.usage)
                elif e.event_type == "step_done":
                    st.markdown(f"✓ **Subtask {e.step.index} done** — {_trunc(e.step.result, 140)}")
                elif e.event_type == "step_failed":
                    st.markdown(f"✗ **Subtask {e.step.index} failed** — {_trunc(e.step.error, 140)}")

        else:
            # ── Guided: LOOK / ACT / OBSERVE as it streams ──────────────────
            guided_events = [e for k, e in steps if k == "guided"]
            for e in guided_events:
                if e.event_type == "perceive":
                    _act_line("tag-goal", "LOOK", e.perceive_tool, e.perceive_args, n=100)
                    _obs_line(e.perceive_observation)
                elif e.event_type == "goal":
                    st.markdown(f"---\n**🎯 {e.current_goal}**")
                    if e.model_reasoning:
                        st.markdown(
                            f'<span class="tag tag-think">THINK</span> '
                            f'{html_lib.escape(_trunc(e.model_reasoning, 400))}',
                            unsafe_allow_html=True,
                        )
                    # Deliberation surface — only populated when those layers are on.
                    if e.options_considered:
                        with st.expander(
                            f"Weighed {len(e.options_considered)} candidates", expanded=False
                        ):
                            for o in e.options_considered:
                                if not isinstance(o, dict):
                                    continue
                                st.markdown(
                                    f"**`{o.get('ref', '?')}`** · score "
                                    f"**{o.get('score', '?')}** — {o.get('label', '')}"
                                )
                                st.caption(
                                    f"for: {o.get('supports', '—')}\n\n"
                                    f"against: {o.get('against', '—')}"
                                )
                    if e.expected_outcome:
                        predicted = []
                        if e.expected_outcome.get("url_changes"):
                            predicted.append("the URL changes")
                        if e.expected_outcome.get("expect_text"):
                            predicted.append(f"{e.expected_outcome['expect_text']!r} appears")
                        if e.expected_outcome.get("target_should_vanish"):
                            predicted.append("the target disappears")
                        if predicted:
                            st.caption("Predicts: " + ", ".join(predicted))
                    _usage_caption(e.usage)
                elif e.event_type == "critique":
                    cls, label = ("tag-veto", "VETO") if e.veto else ("tag-pass", "PASS")
                    st.markdown(
                        f'<span class="tag {cls}">CRITIC {label}</span> '
                        f'{html_lib.escape(_trunc(e.critique, 220))}',
                        unsafe_allow_html=True,
                    )
                    _usage_caption(e.usage)
                elif e.event_type == "act_select":
                    _act_line("tag-select", "ACT", e.tool_name, e.tool_args)
                    if e.reasoning:
                        st.caption(_trunc(e.reasoning, 140))
                    _usage_caption(e.usage)
                elif e.event_type == "act_execute":
                    _obs_line(e.observation, e.screenshot)
                    if e.prediction_mismatch:
                        st.markdown(
                            f'<span class="tag tag-warn">PREDICTION</span> '
                            f'{html_lib.escape(_trunc(e.prediction_mismatch.strip(), 260))}',
                            unsafe_allow_html=True,
                        )

        # ── the turn currently in flight, streamed token by token ───────────
        # Shown after the finished steps and replaced by a proper step entry as
        # soon as this turn lands, so nothing is rendered twice. Only the tail
        # is kept: a long thinking block would otherwise push the trace off screen.
        if st.session_state.live:
            st.markdown(
                f'<span class="tag tag-think">LIVE</span> '
                f'{html_lib.escape(st.session_state.live[-1200:])}▍',
                unsafe_allow_html=True,
            )


# ── final answer & task status ───────────────────────────────────────────────
status = st.session_state.task_status
answer = st.session_state.final_answer
ran    = bool(steps or st.session_state.error or answer)  # something happened

if ran and not st.session_state.running:
    st.divider()
    st.subheader("Summary")

    # ── status badge ──────────────────────────────────────────────────────
    if st.session_state.error:
        st.markdown("""
        <div style="background:#fff5f5;border-left:6px solid #e74c3c;border-radius:4px;
                    padding:0.8rem 1.2rem;margin-bottom:0.8rem;">
          <span style="font-size:1.2rem;font-weight:700;color:#e74c3c;">✗ ERROR</span>
        </div>""", unsafe_allow_html=True)
        st.error(st.session_state.error)

    elif status == "pass":
        st.markdown("""
        <div style="background:#f0faf4;border-left:6px solid #27ae60;border-radius:4px;
                    padding:0.8rem 1.2rem;margin-bottom:0.8rem;">
          <span style="font-size:1.2rem;font-weight:700;color:#27ae60;">✓ TASK COMPLETED</span>
        </div>""", unsafe_allow_html=True)

    elif status == "fail":
        st.markdown("""
        <div style="background:#fff5f5;border-left:6px solid #e74c3c;border-radius:4px;
                    padding:0.8rem 1.2rem;margin-bottom:0.8rem;">
          <span style="font-size:1.2rem;font-weight:700;color:#e74c3c;">✗ TASK INCOMPLETE</span>
          <div style="color:#666;font-size:0.87rem;margin-top:0.2rem;">Some subtasks failed — partial results below.</div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#fef9c3;border-left:6px solid #ca8a04;border-radius:4px;
                    padding:0.8rem 1.2rem;margin-bottom:0.8rem;">
          <span style="font-size:1.2rem;font-weight:700;color:#ca8a04;">⚠ STOPPED</span>
          <div style="color:#666;font-size:0.87rem;margin-top:0.2rem;">Task did not reach a final answer.</div>
        </div>""", unsafe_allow_html=True)

    # ── final answer (always shown if present) ────────────────────────────
    if answer:
        st.markdown("**Final Answer**")
        st.info(answer)
    elif not st.session_state.error:
        st.warning("No final answer was produced.")

    # ── total token usage for the whole run ────────────────────────────────
    final_usage = {}
    for _, item in reversed(steps):
        u = getattr(item, "cumulative_usage", None)
        if u:
            final_usage = u
            break
    if final_usage:
        st.markdown("**Total tokens used**")
        u1, u2, u3 = st.columns(3)
        u1.metric("Input tokens", final_usage.get("input_tokens", 0))
        u2.metric("Output tokens", final_usage.get("output_tokens", 0))
        u3.metric("Total tokens", final_usage.get("total_tokens", 0))

    # ── stats ─────────────────────────────────────────────────────────────
    if st.session_state.mode == "orch":
        orch_events = [e for k, e in steps if k == "orch"]
        done_ev = next((e for e in orch_events if e.event_type == "done"), None)
        if done_ev:
            total   = len(done_ev.state.plan)
            n_done  = sum(1 for s in done_ev.state.plan if s.status == StepStatus.DONE)
            n_fail  = sum(1 for s in done_ev.state.plan if s.status == StepStatus.FAILED)
            n_tools = len(done_ev.state.all_tool_calls)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Subtasks",   total)
            c2.metric("✓ Done",     n_done)
            c3.metric("✗ Failed",   n_fail,  delta=f"-{n_fail}" if n_fail else None, delta_color="inverse")
            c4.metric("Tool calls", n_tools)

            # Per-step result table
            with st.expander("Step results", expanded=(n_fail > 0)):
                for s in done_ev.state.plan:
                    icon = "✓" if s.status == StepStatus.DONE else "✗"
                    col = "green" if s.status == StepStatus.DONE else "red"
                    st.markdown(
                        f'<span style="color:{col};font-weight:600">{icon}</span> '
                        f'**Step {s.index}:** {s.description}',
                        unsafe_allow_html=True,
                    )
                    if s.status == StepStatus.DONE and s.result:
                        st.caption(f"Result: {_trunc(s.result, 160)}")
                    elif s.status == StepStatus.FAILED and s.error:
                        st.caption(f"Failed: {_trunc(s.error, 160)}")
        else:
            # Partial run — show what we have
            step_evs = [e for e in orch_events if e.event_type in ("step_done", "step_failed")]
            if step_evs:
                st.caption(f"{sum(1 for e in step_evs if e.event_type == 'step_done')} steps completed before stop.")
    elif st.session_state.mode == "guided":
        guided = [e for k, e in steps if k == "guided"]
        n       = sum(1 for e in guided if e.event_type == "act_execute")
        vetoes  = sum(1 for e in guided if e.event_type == "critique" and e.veto)
        misses  = sum(1 for e in guided if e.prediction_mismatch)
        if n or vetoes or misses:
            bits = [f"{n} tool call{'s' if n != 1 else ''} made."]
            if vetoes:
                bits.append(f"{vetoes} action{'s' if vetoes != 1 else ''} vetoed by the critic.")
            if misses:
                bits.append(
                    f"{misses} prediction{'s' if misses != 1 else ''} did not match the page."
                )
            st.caption(" ".join(bits))
    else:
        tool_steps = [s for _, s in steps if not s.final]
        n = len(tool_steps)
        if n:
            st.caption(f"{n} tool call{'s' if n != 1 else ''} made.")

elif not ran and not st.session_state.running:
    st.markdown(
        '<div style="text-align:center;padding:3rem;color:#888;">'
        '<div style="font-size:3rem">🎭</div>'
        '<p>Enter a task above and click <b>Run</b>.</p></div>',
        unsafe_allow_html=True,
    )


# ── live refresh ──────────────────────────────────────────────────────────────
# Must be the LAST thing in the script: st.rerun() aborts the run immediately,
# so anything after it never renders. Reaching here means the trace above has
# been drawn, and we can safely poll the queue again for the next tool call.
if st.session_state.running:
    time.sleep(0.2)
    st.rerun()
