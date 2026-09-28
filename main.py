"""
Entry point — Playwright Browser Agent (AWS Bedrock)

Default: single-agent ReAct loop (PlaywrightAgent) — one growing message list,
         all 60 tools available, full context every turn.
--orchestrate: multi-agent pipeline (PlanningAgent + SpecialistAgent, per-group tool selection + self-evaluation)
--guided: ThinkingAgent (read-only search/structure, decides one action at a time) +
          ActorAgent (translates that action into an exact tool call) — tighter,
          per-action loop than --orchestrate, closer to ReAct's granularity but
          split across two LLM roles instead of one.

Run:
    python main.py
    python main.py --task "Go to youtube.com, search 'github tutorial', return the first video name and view count."
    python main.py --orchestrate --task "..."
    python main.py --guided --task "..."
    python main.py --model "amazon.nova-pro-v1:0" --region "us-west-2" --no-headless
    python main.py --list-tools
"""

import argparse
import asyncio
import json
import logging
import os
from typing import Optional

import playwright_tools as pt
from llm_agent import PlaywrightAgent, ReActStep, get_all_tool_schemas
from orchestrator import (
    OrchestratorAgent,
    OrchestratorEvent,
    StepStatus,
    LoopStatus,
    LoopIteration,
    PlanStep,
)
from guided_agent import DeliberationConfig, GuidedAgent, GuidedEvent

logger = logging.getLogger(__name__)

_W = 64
_DIVIDER  = "─" * _W
_HEAVY    = "━" * _W
_DOUBLE   = "═" * _W


def _truncate(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


def _fmt_usage(usage: dict) -> str:
    """Compact one-line token count for a single LLM call."""
    u = usage or {}
    return (f"in={u.get('input_tokens', 0)} out={u.get('output_tokens', 0)} "
            f"total={u.get('total_tokens', 0)}")


# ---------------------------------------------------------------------------
# Orchestrator event renderer — live console output
# ---------------------------------------------------------------------------

_STEP_ICON = {
    StepStatus.PENDING: "○",
    StepStatus.RUNNING: "▶",
    StepStatus.DONE:    "✓",
    StepStatus.FAILED:  "✗",
}

_EVAL_ICON = {
    LoopStatus.DONE:   "✓",
    LoopStatus.RETRY:  "↺",
    LoopStatus.FAILED: "✗",
}


def _obs_str(raw: str) -> str:
    """Format observation JSON for display."""
    try:
        obj = json.loads(raw)
        lines = json.dumps(obj, indent=2, ensure_ascii=False).splitlines()
        return "\n".join("           " + ln for ln in lines)
    except (json.JSONDecodeError, ValueError):
        return "           " + _truncate(raw, 200)


def on_event(event: OrchestratorEvent) -> None:
    """Render every orchestrator event to stdout."""

    # ── Plan printed once after decomposition ───────────────────────────
    if event.event_type == "plan":
        print(f"\n{_DOUBLE}")
        print(f"  PLAN  —  {len(event.state.plan)} subtasks")
        print(_DOUBLE)
        for s in event.state.plan:
            print(f"  {s.index:>2}.  {s.description}")
        print(f"  [tokens: {_fmt_usage(event.usage)}  |  running total: {_fmt_usage(event.cumulative_usage)}]")
        print()

    # ── Starting a new subtask ───────────────────────────────────────────
    elif event.event_type == "step_start":
        s = event.step
        print(f"\n{_HEAVY}")
        print(f"  SUBTASK {s.index}/{len(event.state.plan)}  {s.description}")
        print(_HEAVY)

    # ── Agent 2 chose a tool ─────────────────────────────────────────────
    elif event.event_type == "loop_select":
        it: LoopIteration = event.iteration
        args_str = _truncate(json.dumps(it.tool_args, ensure_ascii=False), 160)
        print(f"\n  {_DIVIDER}")
        print(f"  Loop {it.number}")
        print(f"  {_DIVIDER}")
        print(f"  SELECT   {it.tool_name}({args_str})")
        if it.selection_reasoning:
            print(f"  REASON   {_truncate(it.selection_reasoning, 120)}")
        print(f"  [tokens: {_fmt_usage(event.usage)}  |  running total: {_fmt_usage(event.cumulative_usage)}]")

    # ── Agent 3 executed the tool ────────────────────────────────────────
    elif event.event_type == "loop_execute":
        it: LoopIteration = event.iteration
        print(f"  OBSERVE\n{_obs_str(it.observation)}")

    # ── Agent 4 evaluated the result ────────────────────────────────────
    elif event.event_type == "loop_eval":
        it: LoopIteration = event.iteration
        icon = _EVAL_ICON.get(it.eval_status, "?")
        status_label = it.eval_status.value.upper()
        print(f"  EVAL     {icon} {status_label}  —  {_truncate(it.eval_reason, 120)}")
        if it.eval_status == LoopStatus.DONE and it.eval_answer:
            print(f"  ANSWER   {_truncate(it.eval_answer, 200)}")
        print(f"  [tokens: {_fmt_usage(event.usage)}  |  running total: {_fmt_usage(event.cumulative_usage)}]")

    # ── Subtask completed ────────────────────────────────────────────────
    elif event.event_type == "step_done":
        s: PlanStep = event.step
        loops = len(s.iterations)
        print(f"\n  {_STEP_ICON[StepStatus.DONE]}  Subtask {s.index} done  ({loops} loop{'s' if loops != 1 else ''})")

    # ── Subtask failed ───────────────────────────────────────────────────
    elif event.event_type == "step_failed":
        s: PlanStep = event.step
        print(f"\n  {_STEP_ICON[StepStatus.FAILED]}  Subtask {s.index} FAILED  —  {s.error}")

    # ── Everything done ──────────────────────────────────────────────────
    elif event.event_type == "done":
        total_loops = sum(len(s.iterations) for s in event.state.plan)
        total_tools = len(event.state.all_tool_calls)

        print(f"\n{_DOUBLE}")
        print("  SUMMARY")
        print(_DOUBLE)
        print(f"\n  {'#':<4} {'Status':<10} Subtask")
        print(f"  {'─'*4} {'─'*10} {'─'*44}")
        for s in event.state.plan:
            icon = _STEP_ICON.get(s.status, "?")
            label = f"{icon} {s.status.value}"
            print(f"  {s.index:<4} {label:<10} {_truncate(s.description, 44)}")

        print(f"\n  Total loops: {total_loops}  |  Total tool calls: {total_tools}")
        print(f"  Total tokens: {_fmt_usage(event.cumulative_usage)}")

        print(f"\n{_DOUBLE}")
        print("  FINAL ANSWER")
        print(_DOUBLE)
        print(f"\n{event.final_answer}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Playwright browser automation agent."
    )
    parser.add_argument(
        "--orchestrate", action="store_true", default=False,
        help="Use the multi-agent orchestrator instead of single-agent ReAct loop.",
    )
    parser.add_argument(
        "--guided", action="store_true", default=False,
        help="Use the guided ThinkingAgent+ActorAgent loop (per-action, not per-step planning).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task in plain English. Omit for interactive REPL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
        help="Bedrock model ID.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        help="AWS region.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=os.environ.get("AWS_PROFILE", None),
        help="AWS CLI profile name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per LLM response (default: 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature 0.0–1.0 (default: 0.0).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max Select→Execute→Evaluate loops per subtask (default: 8).",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        default=False,
        help="Enable model reasoning/'thinking' in every mode. The correct Bedrock "
             "key and shape is picked from the model ID (Claude: 'thinking', Nova: "
             "'reasoning_config'), and Claude's temperature=1 requirement is applied "
             "automatically.",
    )
    parser.add_argument(
        "--reasoning-budget",
        type=int,
        default=2048,
        help="Thinking budget in tokens when --reasoning is set (default: 2048). "
             "Clamped into the model's legal range; must be under --max-tokens.",
    )
    parser.add_argument(
        "--no-deliberation",
        action="store_true",
        default=False,
        help="Guided mode: turn off the free deliberation layers (forced candidate "
             "comparison and predict-then-verify), which are on by default.",
    )
    parser.add_argument(
        "--vote",
        type=int,
        default=1,
        metavar="K",
        help="Guided mode: sample each decision K times and take the majority "
             "target. Multiplies decision cost by K (default: 1, off).",
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        default=False,
        help="Guided mode: have a second agent argue against each decision before "
             "it runs. Adds one LLM call per action.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        default=False,
        help="Show the browser window.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        default=False,
        help="Wait for each LLM turn to finish before printing it, instead of "
             "echoing thinking and tool calls token by token as they arrive.",
    )
    parser.add_argument(
        "--screenshot-dir",
        default=None,
        help="Save a numbered screenshot of the page after every tool call into "
             "this directory (created if missing). The fake mouse pointer is "
             "drawn on the element each action targeted.",
    )
    parser.add_argument(
        "--no-cursor",
        action="store_true",
        default=False,
        help="Don't draw the fake mouse pointer into the page.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        default=False,
        help="Print all registered Playwright tools and exit.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def print_tool_list() -> None:
    schemas = get_all_tool_schemas()
    print(f"\n{'Tool':<35} Required params")
    print("-" * 70)
    for schema in schemas:
        spec = schema["toolSpec"]
        required = spec["inputSchema"]["json"].get("required", [])
        print(f"{spec['name']:<35} {', '.join(required) or '(none)'}")
    print(f"\n{len(schemas)} tools registered.\n")


# ---------------------------------------------------------------------------
# ReAct step renderer (PlaywrightAgent)
# ---------------------------------------------------------------------------

def print_react_step(step: ReActStep) -> None:
    print(f"\n{_DIVIDER}")
    print(f"Step {step.iteration}" + (" — DONE" if step.final else ""))
    print(_DIVIDER)
    if step.thought:
        print(f"  THINK  {_truncate(step.thought, 200)}")
    if step.reasoning:
        print(f"  REASON {_truncate(step.reasoning, 300)}")
    print(f"  [tokens: {_fmt_usage(step.usage)}  |  running total: {_fmt_usage(step.cumulative_usage)}]")
    if step.final:
        print(f"\n  ANSWER {step.answer}")
        return
    if step.tool_name:
        args_str = _truncate(json.dumps(step.tool_input, ensure_ascii=False), 160)
        print(f"  ACT    {step.tool_name}({args_str})")
    if step.observation:
        try:
            obj = json.loads(step.observation)
            display = {k: v for k, v in obj.items() if k not in ("html", "page_html")}
            obs_str = json.dumps(display, indent=2, ensure_ascii=False)
        except Exception:
            obs_str = step.observation[:400]
        lines = "\n".join("         " + ln for ln in obs_str.splitlines())
        print(f"  OBS\n{lines}")


def make_delta_printer(args: argparse.Namespace):
    """
    Build the on_delta callback that echoes the model's output to stdout as it
    is generated, or None when --no-stream was passed.

    Called from the boto3 worker thread draining the event stream, so it does
    nothing but write — the structured per-step summary still comes from the
    on_step/on_event renderers once the turn is complete. Chunks are written
    without a newline so text reads as one flowing paragraph; the leading
    newline on the label makes each new block start on its own line.
    """
    if args.no_stream:
        return None

    state = {"kind": None}

    def on_delta(kind: str, text: str) -> None:
        if kind == "notice":
            print(f"\n  NOTE   {text}", flush=True)
            state["kind"] = None
            return
        if kind == "tool":
            print(f"\n  ACT    {text}(", end="", flush=True)
            state["kind"] = kind
            return
        if kind == "tool_input":
            print(text, end="", flush=True)
            return
        if kind != state["kind"]:
            label = "THINK " if kind == "reasoning" else "TEXT  "
            print(f"\n  {label} ", end="", flush=True)
            state["kind"] = kind
        print(text, end="", flush=True)

    return on_delta


def build_react_agent(args: argparse.Namespace) -> PlaywrightAgent:
    return PlaywrightAgent(
        model_id=args.model,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_iterations=args.max_retries,
        on_step=print_react_step,
        reasoning_budget_tokens=_reasoning_budget(args),
        on_delta=make_delta_printer(args),
    )


def _reasoning_budget(args: argparse.Namespace) -> Optional[int]:
    """The thinking budget to request, or None when --reasoning is off."""
    return args.reasoning_budget if args.reasoning else None


def build_orchestrator(args: argparse.Namespace) -> OrchestratorAgent:
    return OrchestratorAgent(
        model_id=args.model,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries_per_step=args.max_retries,
        on_event=on_event,
        reasoning_budget_tokens=_reasoning_budget(args),
        on_delta=make_delta_printer(args),
    )


def print_guided_event(event: GuidedEvent) -> None:
    if event.event_type == "perceive":
        args_str = _truncate(json.dumps(event.perceive_args, ensure_ascii=False), 120)
        print(f"  LOOK   {event.perceive_tool}({args_str})")
        try:
            obj = json.loads(event.perceive_observation)
            display = {k: v for k, v in obj.items() if k not in ("html", "page_html", "structure")}
            print(f"         {_truncate(json.dumps(display, ensure_ascii=False), 300)}")
        except Exception:
            print(f"         {_truncate(event.perceive_observation, 300)}")
    elif event.event_type == "goal":
        print(f"\n{_HEAVY}")
        print(f"  ITER {event.iteration}  GOAL: {event.current_goal}")
        print(_HEAVY)
        if event.model_reasoning:
            print(f"  THINK  {_truncate(event.model_reasoning, 400)}")
        for o in event.options_considered:
            if isinstance(o, dict):
                print(f"  OPT    {o.get('ref', '?')} score={o.get('score', '?')} "
                      f"— {_truncate(str(o.get('label', '')), 60)}")
        print(f"  [tokens: {_fmt_usage(event.usage)}  |  running total: {_fmt_usage(event.cumulative_usage)}]")
    elif event.event_type == "act_select":
        args_str = _truncate(json.dumps(event.tool_args, ensure_ascii=False), 160)
        print(f"  ACT    {event.tool_name}({args_str})")
        if event.reasoning:
            print(f"  REASON {_truncate(event.reasoning, 120)}")
        print(f"  [tokens: {_fmt_usage(event.usage)}  |  running total: {_fmt_usage(event.cumulative_usage)}]")
    elif event.event_type == "critique":
        verdict = "VETO" if event.veto else "PASS"
        print(f"  CRITIC {verdict}  {_truncate(event.critique, 200)}")
    elif event.event_type == "act_execute":
        print(f"  OBSERVE\n{_obs_str(event.observation)}")
        if event.prediction_mismatch:
            print(f"  ⚠ {_truncate(event.prediction_mismatch.strip(), 300)}")
    elif event.event_type == "done":
        print(f"\n{_DOUBLE}\n  DONE\n{_DOUBLE}")
        print(f"\n  ANSWER {event.final_answer}")
        print(f"\n  Total tokens: {_fmt_usage(event.cumulative_usage)}")
    elif event.event_type == "failed":
        print(f"\n  ✗ {event.final_answer}")
        print(f"  Total tokens: {_fmt_usage(event.cumulative_usage)}")


def build_guided_agent(args: argparse.Namespace) -> GuidedAgent:
    return GuidedAgent(
        model_id=args.model,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_iterations=args.max_retries * 5,
        on_event=print_guided_event,
        reasoning_budget_tokens=_reasoning_budget(args),
        on_delta=make_delta_printer(args),
        deliberation=DeliberationConfig(
            forced_schema=not args.no_deliberation,
            predict_verify=not args.no_deliberation,
            self_consistency_k=args.vote,
            critic=args.critic,
        ),
    )


# ---------------------------------------------------------------------------
# Single-task mode
# ---------------------------------------------------------------------------

async def run_task(args: argparse.Namespace) -> None:
    mode = "guided" if args.guided else ("orchestrator" if args.orchestrate else "ReAct")
    print(f"\nModel   : {args.model}")
    print(f"Region  : {args.region}")
    print(f"Mode    : {mode}")
    print(f"Task    : {args.task}")
    print(_DOUBLE)

    if args.guided:
        agent = build_guided_agent(args)
        await agent.run(args.task)
    elif args.orchestrate:
        agent = build_orchestrator(args)
        await agent.run(args.task)
    else:
        agent = build_react_agent(args)
        try:
            await agent.run(args.task)
        finally:
            print(f"\n{_DOUBLE}")
            print(f"  Total tokens: {_fmt_usage(agent._total_usage)}")
            print(_DOUBLE)
            if pt._state["browser"]:
                await pt.stop_browser()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_repl(args: argparse.Namespace) -> None:
    mode = "guided" if args.guided else ("orchestrator" if args.orchestrate else "ReAct")
    print(f"\nPlaywright Agent — Interactive Mode ({mode})")
    print(f"Model : {args.model}  |  Region : {args.region}")
    print("Commands:  exit | tools")
    print(_DOUBLE)

    agent = (
        build_guided_agent(args) if args.guided
        else build_orchestrator(args) if args.orchestrate
        else build_react_agent(args)
    )

    while True:
        try:
            task = input("\nTask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if task.lower() == "tools":
            print_tool_list()
            continue

        if args.guided or args.orchestrate:
            await agent.run(task)
        else:
            try:
                await agent.run(task, reset=False)
            finally:
                pass  # keep browser alive between REPL tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_tools:
        print_tool_list()
        return

    # Screenshot/cursor settings live on the tool layer's global state, so all
    # three modes pick them up without threading them through every agent.
    pt._state["show_cursor"] = not args.no_cursor
    if args.screenshot_dir:
        os.makedirs(args.screenshot_dir, exist_ok=True)
        pt._state["screenshot_dir"] = args.screenshot_dir
        print(f"Frames  : {args.screenshot_dir}")

    if args.task:
        await run_task(args)
    else:
        await run_repl(args)


if __name__ == "__main__":
    asyncio.run(main())
