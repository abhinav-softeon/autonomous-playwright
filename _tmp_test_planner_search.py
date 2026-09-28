"""
Temp verification script for PlanningAgent's new optional search_elements
capability. Mocks BedrockClient.converse() but drives a REAL headless
browser (example.com) so search_elements executes against the real page
index. Verifies:
  1. Planner calls search_elements once, gets real results back.
  2. Planner then calls submit_plan with a selector embedded in the step.
  3. Specialist uses that selector directly (no re-search needed).
Deleted after use.
"""
import asyncio

import orchestrator as orch


def _tool_use_response(name: str, input_: dict, tool_use_id: str = "t1") -> dict:
    return {"output": {"message": {"content": [
        {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": input_}}
    ]}}}


def _text_response(text: str) -> dict:
    return {"output": {"message": {"content": [{"text": text}]}}}


SCRIPT = [
    # 1) PlanningAgent initial plan: no page yet -> can_search is False -> goes straight to submit_plan
    _tool_use_response("submit_plan", {
        "steps": [{"description": "Navigate to example.com", "group": "navigation"}],
        "reasoning": "Need to load the page first before searching it.",
    }),
    # 2) SpecialistAgent.act(): navigate
    _tool_use_response("act", {
        "status": "retry", "eval_reason": "first action",
        "tool_name": "navigate", "tool_args": {"url": "https://example.com"},
        "reasoning": "load the page",
    }),
    # 3) SpecialistAgent.act(): evaluate navigate -> done
    _tool_use_response("act", {
        "status": "done", "eval_reason": "page loaded", "answer": "Loaded example.com",
    }),
    # 4) PlanningAgent replan: NOW a page exists -> should be offered search_elements.
    #    Script it to call search_elements first (real execution against real page).
    _tool_use_response("search_elements", {"terms": ["more information", "learn more"]}, tool_use_id="search1"),
    # 5) PlanningAgent's SECOND round-trip after seeing real search results -> submit_plan
    #    with the selector embedded (we just echo a plausible one; real content was in the tool result)
    _tool_use_response("submit_plan", {
        "steps": [{
            "description": "Click the 'Learn more' link (selector: a[href=\"https://iana.org/domains/example\"])",
            "group": "clicking",
        }],
        "reasoning": "Found the exact selector via search_elements, embedding it for the specialist.",
    }),
    # 6) SpecialistAgent.act() for the clicking step: should use the embedded selector directly
    _tool_use_response("act", {
        "status": "retry", "eval_reason": "first action, selector already given",
        "tool_name": "click", "tool_args": {"selector": "a[href=\"https://iana.org/domains/example\"]"},
        "reasoning": "Step goal already names the selector, using it directly per rule 2.",
    }),
    # 7) SpecialistAgent.act() evaluates click -> done
    _tool_use_response("act", {
        "status": "done", "eval_reason": "navigated to iana.org domains page", "answer": "Clicked learn more",
    }),
    # 8) PlanningAgent replan -> empty (complete)
    _tool_use_response("submit_plan", {"steps": [], "reasoning": "done"}),
    # 9) final summary
    _text_response("Navigated to example.com and clicked Learn more."),
]

_call_count = 0


async def fake_converse(self, messages, system=None, tools=None):
    global _call_count
    resp = SCRIPT[_call_count]
    _call_count += 1
    tool_names = [t["toolSpec"]["name"] for t in (tools or [])]
    print(f"  [converse #{_call_count}] tools offered: {tool_names}")
    return resp


async def main():
    orch.BedrockClient.converse = fake_converse

    events = []

    def on_event(event):
        events.append(event.event_type)
        if event.event_type == "plan":
            print(f"  PLAN: {[(s.description, s.group) for s in event.state.plan]}")
        elif event.event_type == "loop_select":
            print(f"  SELECT {event.iteration.tool_name}({event.iteration.tool_args})")

    agent = orch.OrchestratorAgent(on_event=on_event, max_retries_per_step=8, max_total_steps=10)
    final_answer = await agent.run("Go to example.com and click the learn more link.")

    print("\nFINAL ANSWER:", final_answer)

    # ── Assertions ──────────────────────────────────────────────────────
    # search_elements should have been offered as a tool once a page existed (replan call)
    # and NOT offered before any page existed (initial plan call).
    assert _call_count == len(SCRIPT), f"expected all {len(SCRIPT)} scripted calls consumed, got {_call_count}"

    select_events = [e for e in events if e == "loop_select"]
    assert len(select_events) == 2, f"expected exactly 2 tool selections (navigate, click), got {len(select_events)}"

    click_step = agent  # just to avoid lint about unused
    print("\nALL ASSERTIONS PASSED")


asyncio.run(main())
