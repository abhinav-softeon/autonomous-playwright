"""
Entry point — Playwright Browser Agent (AWS Bedrock)

Default: single-agent ReAct loop (PlaywrightAgent) — one growing message list,
         all 60 tools available, full context every turn.
--orchestrate: multi-agent pipeline (DecompositionAgent + ToolSpecialistAgent + EvaluationAgent)

Run:
    python main.py
    python main.py --task "Go to youtube.com, search 'github tutorial', return the first video name and view count."
    python main.py --orchestrate --task "..."
    python main.py --model "amazon.nova-pro-v1:0" --region "us-west-2" --no-headless
    python main.py --list-tools
"""

import argparse
import asyncio
import json
import logging
import os

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

logger = logging.getLogger(__name__)

_W = 64
_DIVIDER  = "─" * _W
_HEAVY    = "━" * _W
_DOUBLE   = "═" * _W


def _truncate(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


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
        "--no-headless",
        action="store_true",
        default=False,
        help="Show the browser window.",
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


def build_react_agent(args: argparse.Namespace) -> PlaywrightAgent:
    return PlaywrightAgent(
        model_id=args.model,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_iterations=args.max_retries,
        on_step=print_react_step,
    )


def build_orchestrator(args: argparse.Namespace) -> OrchestratorAgent:
    return OrchestratorAgent(
        model_id=args.model,
        region=args.region,
        profile=args.profile,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries_per_step=args.max_retries,
        on_event=on_event,
    )


# ---------------------------------------------------------------------------
# Single-task mode
# ---------------------------------------------------------------------------

async def run_task(args: argparse.Namespace) -> None:
    print(f"\nModel   : {args.model}")
    print(f"Region  : {args.region}")
    print(f"Mode    : {'orchestrator' if args.orchestrate else 'ReAct'}")
    print(f"Task    : {args.task}")
    print(_DOUBLE)

    if args.orchestrate:
        agent = build_orchestrator(args)
        await agent.run(args.task)
    else:
        agent = build_react_agent(args)
        try:
            await agent.run(args.task)
        finally:
            if pt._state["browser"]:
                await pt.stop_browser()


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_repl(args: argparse.Namespace) -> None:
    mode = "orchestrator" if args.orchestrate else "ReAct"
    print(f"\nPlaywright Agent — Interactive Mode ({mode})")
    print(f"Model : {args.model}  |  Region : {args.region}")
    print("Commands:  exit | tools")
    print(_DOUBLE)

    agent = build_orchestrator(args) if args.orchestrate else build_react_agent(args)

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

        if args.orchestrate:
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

    if args.task:
        await run_task(args)
    else:
        await run_repl(args)


if __name__ == "__main__":
    asyncio.run(main())
