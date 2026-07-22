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
.eval-done  { color:#27ae60; font-weight:600; }
.eval-retry { color:#e67e22; font-weight:600; }
.eval-failed{ color:#e74c3c; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
for k, v in {
    "running": False, "steps": [], "final_answer": "",
    "q": None, "error": "", "mode": "react", "task_status": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
        ["ReAct (recommended)", "Orchestrator"],
        index=0,
        help="ReAct: single agent, full history. Orchestrator: multi-agent decomposition.",
    )
    use_orchestrate = mode == "Orchestrator"

    st.divider()
    st.subheader("Model")
    _opts = [
        "amazon.nova-pro-v1:0",
        "amazon.nova-2-lite-v1:0",
        "anthropic.claude-sonnet-4-6:20250514-v1:0",
        "anthropic.claude-haiku-4-5:20251001-v1:0",
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
    headless   = st.checkbox("Headless browser", value=_env_bool("HEADLESS", True))

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
        st.session_state.steps = []
        st.session_state.final_answer = ""
        st.session_state.error = ""
        st.session_state.task_status = None
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


# ── background runners ────────────────────────────────────────────────────────

def _run_react(task: str, cfg: dict, q: queue.Queue) -> None:
    """Run PlaywrightAgent in a background thread, push ReActStep to queue."""
    def on_step(step: ReActStep):
        q.put(("react", step))

    async def _go():
        agent = PlaywrightAgent(
            model_id=cfg["model_id"], region=cfg["region"],
            profile=cfg["profile"] or None,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            max_iterations=cfg["max_iters"], on_step=on_step,
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

    async def _go():
        agent = OrchestratorAgent(
            model_id=cfg["model_id"], region=cfg["region"],
            profile=cfg["profile"] or None,
            max_tokens=cfg["max_tokens"], temperature=cfg["temperature"],
            max_retries_per_step=cfg["max_iters"], on_event=on_event,
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
    st.session_state.steps   = []
    st.session_state.final_answer = ""
    st.session_state.error   = ""
    st.session_state.mode    = "orch" if use_orchestrate else "react"

    q: queue.Queue = queue.Queue()
    st.session_state.q = q
    cfg = dict(model_id=model_id, region=region, profile=profile,
               max_tokens=max_tokens, temperature=temperature, max_iters=max_iters)
    runner = _run_orchestrator if use_orchestrate else _run_react
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

    st.session_state.steps.extend(new_items)
    if done:
        st.session_state.running = False
        st.session_state.q = None
    else:
        time.sleep(0.2)
        st.rerun()

    if not done:
        st.rerun()


# ── render ────────────────────────────────────────────────────────────────────
if st.session_state.running:
    st.info("⏳ Agent is running…", icon="🔄")

steps = st.session_state.steps

if steps:
    st.divider()

    if st.session_state.mode == "react":
        # ── ReAct view ────────────────────────────────────────────────────
        st.caption(f"Mode: ReAct — {len([s for _, s in steps if not s.final])} tool calls")
        for _, step in steps:
            if step.final:
                continue
            with st.expander(
                f"Step {step.iteration} — `{step.tool_name}`",
                expanded=(step.iteration == len(steps) - 1),
            ):
                if step.thought:
                    st.markdown(
                        f'<span class="tag tag-think">THINK</span> {html_lib.escape(_trunc(step.thought))}',
                        unsafe_allow_html=True,
                    )
                args_str = html_lib.escape(_trunc(json.dumps(step.tool_input, ensure_ascii=False), 160))
                st.markdown(
                    f'<span class="tag tag-act">ACT</span> <code>{step.tool_name}({args_str})</code>',
                    unsafe_allow_html=True,
                )
                if step.observation:
                    st.markdown('<span class="tag tag-obs">OBS</span>', unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="obs-box">{_fmt(step.observation)}</div>',
                        unsafe_allow_html=True,
                    )
                if step.prompt_sent:
                    with st.expander("📨 Prompt sent to model", expanded=False):
                        st.code(step.prompt_sent, language="text")

    else:
        # ── Orchestrator view ─────────────────────────────────────────────
        orch_events = [e for k, e in steps if k == "orch"]

        plan_event = next((e for e in orch_events if e.event_type == "plan"), None)
        if plan_event:
            with st.expander("📋 Plan", expanded=True):
                for s in plan_event.state.plan:
                    latest = s.status
                    for e in reversed(orch_events):
                        if e.step and e.step.index == s.index:
                            latest = e.step.status
                            break
        icon = {StepStatus.DONE: "✓", StepStatus.FAILED: "✗",
                StepStatus.RUNNING: "▶", StepStatus.PENDING: "○",
                StepStatus.SKIPPED: "⏭"}.get(latest, "○")
                    st.markdown(f"{icon} **Step {s.index}** — {s.description}")

        st.divider()

        step_starts = [e for e in orch_events if e.event_type == "step_start"]
        for se in step_starts:
            ps = se.step
            step_evs = [e for e in orch_events if e.step and e.step.index == ps.index]
            done_ev   = next((e for e in step_evs if e.event_type == "step_done"), None)
            failed_ev = next((e for e in step_evs if e.event_type == "step_failed"), None)
            header_icon = "✓" if done_ev else ("✗" if failed_ev else "▶")

            with st.expander(
                f"{header_icon} Subtask {ps.index} — {ps.description}",
                expanded=(not done_ev and not failed_ev),
            ):
                iters: dict = {}
                for e in step_evs:
                    if e.iteration is None: continue
                    n = e.iteration.number
                    if n not in iters: iters[n] = {}
                    iters[n][e.event_type] = e.iteration

                for n in sorted(iters):
                    it = iters[n]
                    sel = it.get("loop_select")
                    exe = it.get("loop_execute")
                    evl = it.get("loop_eval")
                    st.markdown(f"**Loop {n}**")

                    if sel:
                        args_str = html_lib.escape(json.dumps(sel.tool_args, ensure_ascii=False)[:120])
                        st.markdown(
                            f'<span class="tag tag-select">SELECT</span> <code>{sel.tool_name}({args_str})</code>',
                            unsafe_allow_html=True,
                        )
                        if sel.selection_reasoning:
                            st.caption(_trunc(sel.selection_reasoning, 160))
                    if exe and exe.observation:
                        st.markdown('<span class="tag tag-obs">OBSERVE</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="obs-box">{_fmt(exe.observation)}</div>', unsafe_allow_html=True)
                    if evl:
                        ecls = {"done": "eval-done", "retry": "eval-retry", "failed": "eval-failed"}.get(evl.eval_status.value, "")
                        eicon = {"done": "✓", "retry": "↺", "failed": "✗"}.get(evl.eval_status.value, "?")
                        st.markdown(
                            f'<span class="tag tag-eval">EVAL</span> <span class="{ecls}">{eicon} {evl.eval_status.value.upper()}</span> — {html_lib.escape(_trunc(evl.eval_reason, 120))}',
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")

                if done_ev:
                    st.success(f"✓ {done_ev.step.result}")
                elif failed_ev:
                    st.error(f"✗ {failed_ev.step.error}")


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
