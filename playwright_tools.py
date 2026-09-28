"""
Playwright LLM Tool Wrappers
-----------------------------
Each function is a self-contained LLM tool wrapper around Playwright.
All functions accept primitive parameters (str, int, bool, list) so they can
be called directly from LLM tool-call payloads.

Session management:
  - Call `start_browser()` before using any page-level tools.
  - Call `stop_browser()` when done.
  - A global `_state` dict holds the active browser, context, and page.
"""

import asyncio
import os
import re
from difflib import SequenceMatcher
from typing import Any, Optional
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

# ---------------------------------------------------------------------------
# Global session state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {
    "playwright": None,
    "browser": None,
    "context": None,
    "page": None,
    "page_index": None,     # cached _build_page_index() result; None = stale/rebuild needed
    "last_screenshot": None,  # UI-only side channel — never sent to the LLM, see _capture_step_screenshot()
    "show_cursor": True,    # draw the fake mouse pointer overlay (see CURSOR OVERLAY section)
    "cursor_pos": None,     # (x, y) viewport coords of the overlay pointer, or None
    "cursor_settle_ms": 150,  # render time allowed for the pointer glide / click ring
    "screenshot_dir": None,   # if set, every step screenshot is also written here as a PNG-ish JPEG
    "screenshot_count": 0,
}


def _page() -> Page:
    if _state["page"] is None:
        raise RuntimeError("No active browser session. Call start_browser() first.")
    return _state["page"]


def _invalidate_index() -> None:
    """Mark the cached page index as stale so the next search/expand call rebuilds it."""
    _state["page_index"] = None


# ---------------------------------------------------------------------------
# SESSION MANAGEMENT
# ---------------------------------------------------------------------------

async def start_browser(
    browser_type: str = "chromium",
    headless: bool = True,
    slow_mo: int = 0,
    viewport_width: int = 1280,
    viewport_height: int = 720,
    channel: str = "chrome",
) -> dict:
    """
    Launch a browser and open a new page.

    Args:
        browser_type: "chromium", "firefox", or "webkit"
        headless: Run without a visible window
        slow_mo: Milliseconds to slow down each action (useful for debugging)
        viewport_width: Browser viewport width in pixels
        viewport_height: Browser viewport height in pixels
        channel: Browser channel to use. "chrome" uses system-installed Google Chrome,
                 "msedge" uses system Edge. Empty string uses Playwright's own binaries.

    Returns:
        {"status": "ok", "browser": browser_type}
    """
    pw: Playwright = await async_playwright().start()
    launcher = getattr(pw, browser_type)
    launch_kwargs: dict[str, Any] = {"headless": headless, "slow_mo": slow_mo}
    if channel:
        launch_kwargs["channel"] = channel
    browser: Browser = await launcher.launch(**launch_kwargs)
    context: BrowserContext = await browser.new_context(
        viewport={"width": viewport_width, "height": viewport_height},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Hide automation fingerprints so sites don't show CAPTCHA/sorry pages
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
        Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
        window.chrome = {runtime: {}, loadTimes: function(){}, csi: function(){}, app: {}};
    """)
    # Fake mouse pointer, re-installed on every document so it survives
    # navigation (see the CURSOR OVERLAY section).
    if _state.get("show_cursor"):
        await context.add_init_script(_CURSOR_INIT_JS)
    page: Page = await context.new_page()

    _state["playwright"] = pw
    _state["browser"] = browser
    _state["context"] = context
    _state["page"] = page

    return {"status": "ok", "browser": browser_type, "channel": channel}


async def stop_browser() -> dict:
    """
    Close the active browser session and clean up all resources.

    Returns:
        {"status": "closed"}
    """
    if _state["browser"]:
        await _state["browser"].close()
    if _state["playwright"]:
        await _state["playwright"].stop()
    _state.update({"playwright": None, "browser": None, "context": None, "page": None,
                   "cursor_pos": None})
    return {"status": "closed"}


async def _capture_step_screenshot(label: str = "") -> Optional[bytes]:
    """
    Best-effort viewport screenshot for the UI's live step-by-step trace.
    This is a UI-only side channel (stashed in _state["last_screenshot"] by
    execute_tool_call in llm_agent.py) — it is NEVER put into a tool's JSON
    result, so it never reaches the LLM's context. Returns None if there is
    no active page or the capture fails for any reason (never raises).

    If _state["screenshot_dir"] is set, the frame is also written there as a
    zero-padded, label-suffixed JPEG so CLI runs (which can't render images in
    a terminal) still end up with a browsable reel of what the agent did.

    Args:
        label: Short tag for the filename, normally the tool name.
    """
    page = _state.get("page")
    if page is None:
        return None
    try:
        shot = await page.screenshot(type="jpeg", quality=50, timeout=5000)
    except Exception:
        return None

    directory = _state.get("screenshot_dir")
    if directory:
        try:
            _state["screenshot_count"] += 1
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_") or "step"
            path = os.path.join(directory, f"{_state['screenshot_count']:03d}_{safe}.jpg")
            os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(shot)
        except Exception:
            pass  # a UI nicety must never break the run
    return shot


# ---------------------------------------------------------------------------
# CURSOR OVERLAY
# ---------------------------------------------------------------------------
# Playwright screenshots never contain the OS mouse pointer — the browser
# renders the page, not the cursor. To make "here is what the agent just did"
# visible, we draw a fake pointer INTO the page: an arrow + click ring in a
# closed-off overlay layer, moved to the element each action targets.
#
# Two things keep it from polluting what the agent sees:
#   - the layer lives in a shadow root on a single [data-pw-overlay] host, so
#     the site's own DOM is untouched and querySelectorAll can't see inside it
#   - that host is excluded by the HTML cleaner and _build_page_index(), and
#     carries aria-hidden="true" so it stays out of get_page_structure()'s
#     ARIA snapshot
#
# The installer is a function EXPRESSION so it can be reused two ways: appended
# with "()" as a context init script (runs on every document, every navigation),
# and called again inside _place_cursor's evaluate to self-heal if a site's JS
# has blown the overlay away. It is idempotent — installing twice is a no-op,
# and build() re-attaches the layer if it was detached.

_CURSOR_INSTALLER_JS = r"""
(function installPwCursor() {
    if (window.top !== window.self) return;   // top document only, not every iframe
    if (window.__pwCursor) { window.__pwCursor.build(); return; }

    const ARROW_SVG =
        '<svg width="26" height="26" viewBox="0 0 26 26" xmlns="http://www.w3.org/2000/svg">' +
        '<path d="M5 2 L5 21 L10.2 15.8 L13.6 24 L17.4 22.3 L14 14.4 L21.5 14.4 Z" ' +
        'fill="#111" stroke="#fff" stroke-width="1.7" stroke-linejoin="round"/></svg>';

    const S = { x: null, y: null, layer: null, arrow: null, ring: null, timer: null };
    window.__pwCursor = S;

    S.build = function build() {
        if (!document.body) return false;
        if (S.layer && S.layer.isConnected) return true;

        const host = document.createElement('div');
        host.setAttribute('data-pw-overlay', '1');
        host.setAttribute('aria-hidden', 'true');
        host.style.cssText = 'all:initial;position:fixed;left:0;top:0;width:0;height:0;' +
                             'z-index:2147483647;pointer-events:none;';
        // CLOSED, not open: Playwright's selector engine and aria_snapshot both
        // pierce open shadow roots, which put the arrow's <svg> into
        // get_page_structure()'s outline as a stray "img" node. Closed keeps the
        // pointer out of every view the agent has. We hold direct references to
        // the nodes below, so never needing to query them back is not a problem.
        const root = host.attachShadow({ mode: 'closed' });

        const layer = document.createElement('div');
        layer.style.cssText = 'position:fixed;left:0;top:0;width:100vw;height:100vh;' +
                              'pointer-events:none;overflow:hidden;';

        const ring = document.createElement('div');
        ring.style.cssText = 'position:absolute;left:0;top:0;width:34px;height:34px;' +
                             'margin:-17px 0 0 -17px;border-radius:50%;box-sizing:border-box;' +
                             'border:3px solid rgba(255,45,85,.95);background:rgba(255,45,85,.18);' +
                             'opacity:0;transform:translate(-200px,-200px) scale(.35);';

        const arrow = document.createElement('div');
        arrow.style.cssText = 'position:absolute;left:0;top:0;width:26px;height:26px;' +
                              'transform:translate(-200px,-200px);' +
                              'transition:transform 220ms cubic-bezier(.33,.02,.29,1);' +
                              'filter:drop-shadow(0 1px 3px rgba(0,0,0,.5));';
        arrow.innerHTML = ARROW_SVG;

        layer.appendChild(ring);
        layer.appendChild(arrow);
        root.appendChild(layer);
        document.body.appendChild(host);

        S.layer = layer; S.arrow = arrow; S.ring = ring;
        if (S.x !== null) S.place(S.x, S.y);   // restore position after a navigation
        return true;
    };

    // The arrow's hotspot is its tip at (5, 2) in the SVG's own box.
    // instant=true skips the glide, for restoring the pointer onto a document
    // that just replaced the old one — there is no movement to show, and an
    // animation would still be in flight when the screenshot is taken.
    S.place = function place(x, y, instant) {
        if (x === null || x === undefined) return;
        S.x = x; S.y = y;
        if (!S.build()) return;
        const transition = S.arrow.style.transition;
        if (instant) S.arrow.style.transition = 'none';
        S.arrow.style.transform = 'translate(' + (x - 5) + 'px, ' + (y - 2) + 'px)';
        if (instant) {
            void S.arrow.offsetWidth;               // paint before restoring the glide
            S.arrow.style.transition = transition;
        }
    };

    // Ring expands from the click point and lingers, so it is still on screen
    // when the post-action screenshot is taken a moment later.
    S.pulse = function pulse(x, y) {
        S.place(x, y);
        if (!S.layer) return;
        const r = S.ring;
        r.style.transition = 'none';
        r.style.opacity = '0';
        r.style.transform = 'translate(' + x + 'px, ' + y + 'px) scale(.35)';
        void r.offsetWidth;                     // flush the reset before animating
        r.style.transition = 'opacity 140ms ease-out, transform 260ms cubic-bezier(.2,.9,.3,1)';
        r.style.opacity = '1';
        r.style.transform = 'translate(' + x + 'px, ' + y + 'px) scale(1)';
        clearTimeout(S.timer);
        S.timer = setTimeout(function () { r.style.opacity = '0'; }, 3000);
    };

    // Real Playwright mouse input (click/hover move the actual mouse) is
    // mirrored for free, which also covers page.mouse.* calls we never see.
    addEventListener('mousemove', function (e) { S.place(e.clientX, e.clientY); }, true);
    addEventListener('mousedown', function (e) { S.pulse(e.clientX, e.clientY); }, true);

    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { S.build(); }, { once: true });
    else
        S.build();
})
"""

_CURSOR_INIT_JS = _CURSOR_INSTALLER_JS + "();"

# Self-healing placement call: reinstall if needed, then move/pulse.
_CURSOR_PLACE_JS = (
    "([x, y, click, instant]) => { "
    + _CURSOR_INSTALLER_JS
    + "(); const c = window.__pwCursor; if (!c) return; "
    "if (click) c.pulse(x, y); else c.place(x, y, instant); }"
)


async def _place_cursor(
    x: float, y: float, click: bool = False, instant: bool = False
) -> None:
    """Move the fake pointer to viewport coords (x, y); optionally flash the click ring."""
    page = _state.get("page")
    if page is None or not _state.get("show_cursor"):
        return
    try:
        await page.evaluate(_CURSOR_PLACE_JS, [x, y, click, instant])
        _state["cursor_pos"] = (x, y)
    except Exception:
        pass  # cross-origin nav mid-flight, page closed, CSP — never fail an action for this


async def _restore_cursor() -> None:
    """
    Redraw the pointer where it already was, with no animation.

    Every navigation hands us a fresh document whose overlay starts empty, so
    without this the pointer would blink out of the trace after any navigate/
    reload and after every read-only tool that follows one. Costs one evaluate
    and no settle delay, since nothing is animating.
    """
    position = _state.get("cursor_pos")
    if position is None:
        return
    await _place_cursor(position[0], position[1], instant=True)


async def _cursor_to_selector(selector: str) -> Optional[tuple[float, float]]:
    """
    Glide the fake pointer to the centre of `selector` and return that point.

    Called BEFORE the real action so the pointer is already on the target when
    it fires (and so a human watching a headed run sees the approach). Returns
    None if the element can't be measured — a missing pointer is never a reason
    to stop; the real action's own error handling still applies.
    """
    page = _state.get("page")
    if page is None or not _state.get("show_cursor"):
        return None
    try:
        locator = page.locator(selector).first
        try:
            await locator.scroll_into_view_if_needed(timeout=2000)
        except Exception:
            pass  # the action itself will scroll; we just wanted correct coords
        box = await locator.bounding_box(timeout=2000)
        if not box or not box.get("width") or not box.get("height"):
            return None
        point = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        await _place_cursor(point[0], point[1])
        await _settle_cursor()
        return point
    except Exception:
        return None


async def _mark_cursor_action(point: Optional[tuple[float, float]], click: bool = False) -> None:
    """
    Re-assert the pointer at `point` after the action has run, then let it render.

    Needed because the action may have navigated or re-rendered, which drops the
    overlay — the init script rebuilds it on the new document, but only this call
    knows where the action actually happened. Also where the click ring is drawn:
    a ring only makes sense once the click has landed.
    """
    if point is None:
        return
    await _place_cursor(point[0], point[1], click=click)
    await _settle_cursor()


async def _settle_cursor() -> None:
    """Give the CSS glide/ring time to paint before the next screenshot."""
    page = _state.get("page")
    delay = _state.get("cursor_settle_ms") or 0
    if page is None or delay <= 0:
        return
    try:
        await page.wait_for_timeout(delay)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# THE PAGE VIEW
# ---------------------------------------------------------------------------
# There used to be three overlapping ways to describe the same DOM, and any
# action returned two of them unasked. Measured on news.ycombinator.com:
#
#     get_page_structure()   2,323 tok   roles + names + VISIBLE TEXT + [ref=eN]
#     _get_page_html()        3,479 tok   raw HTML, no refs
#     _get_page_snapshot()    5,386 tok   interactive elements only, no text
#
# get_page_structure wins on every axis that matters: it is the smallest, it is
# the only one carrying readable text (prices, view counts, dates), and its
# [ref=eN] refs are usable DIRECTLY as a selector ("aria-ref=e12") so the model
# never has to construct CSS. It is now the single representation every tool
# returns, and the only one exposed to the LLM — _get_page_html and
# _get_page_snapshot are kept as private helpers for internal callers only
# (underscore = excluded from the LLM tool registry, see llm_agent._TOOL_REGISTRY).

#: Characters of page view attached to an action's result. The full view is
#: always available via get_page_structure(chunk_index=N) if the model wants more.
_PAGE_VIEW_MAX_CHARS = 6000


async def _page_view(max_chars: int = _PAGE_VIEW_MAX_CHARS) -> dict:
    """
    The one canonical description of the current page, for attaching to results.

    Returns {"url", "title", "structure"} — or {"url", "title"} alone if the
    snapshot fails, since a missing view must never fail the action that
    produced it.
    """
    page = _page()
    view: dict[str, Any] = {"url": page.url}
    try:
        view["title"] = await page.title()
    except Exception:
        pass
    try:
        snapshot = await get_page_structure(max_chars=max_chars)
        if "structure" in snapshot:
            view["structure"] = snapshot["structure"]
            if snapshot.get("total_chunks", 1) > 1:
                view["structure_note"] = (
                    f"Showing chunk 0 of {snapshot['total_chunks']}. Call "
                    f"get_page_structure(chunk_index=1) for the rest."
                )
    except Exception:
        pass
    return view


# ---------------------------------------------------------------------------
# NAVIGATION
# ---------------------------------------------------------------------------

async def navigate(url: str, wait_until: str = "domcontentloaded", timeout: int = 60000) -> dict:
    """
    Navigate the browser to a URL.
    Waits for the page to settle, then returns cleaned HTML so the agent
    can immediately see the real page structure and selectors.

    Args:
        url: Full URL to navigate to (e.g. "https://example.com")
        wait_until: When to consider navigation done.
                    Options: "load", "domcontentloaded", "networkidle", "commit"
        timeout: Max wait time in milliseconds (default 60000)

    Returns:
        {"status": "ok", "url": ..., "title": ..., "html": <cleaned page HTML>}
    """
    page = _page()
    await page.goto(url, wait_until=wait_until, timeout=timeout)

    # Wait a moment for JS-rendered content (e.g. YouTube video grid) to appear
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except Exception:
        pass  # networkidle timeout is fine — content may still be useful

    _invalidate_index()
    return {"status": "ok", **await _page_view()}


async def _get_page_snapshot(chunk_index: int = 0, chunk_size: int = 80) -> dict:
    """
    Return a structured snapshot of every visible interactive element on the page.
    Each element includes its selector, position (x/y/w/h), DOM order, and whether
    it is currently in the visible viewport — so the agent knows WHERE things are,
    not just what exists.

    Results are paginated, not truncated — if there are more than chunk_size
    elements, call again with a higher chunk_index to see the rest. Nothing is
    ever permanently lost; total_chunks tells you how many pages exist.

    Args:
        chunk_index: Which page of elements to return (0-based)
        chunk_size: Elements per page (default 80)

    Returns:
        {
          "url": ..., "title": ..., "viewport": {"width": ..., "height": ...},
          "chunk_index": ..., "total_chunks": ..., "total_elements": ...,
          "elements": [
            {
              "order": 1,               # DOM order, 1 = first on page
              "tag": "textarea",
              "aria_label": "Search",
              "selector": "textarea[aria-label='Search']",
              "x": 445, "y": 312,       # top-left corner in pixels
              "width": 526, "height": 44,
              "in_viewport": true,      # fully or partially visible right now
              "position_hint": "center" # rough screen zone
            },
            ...
          ]
        }
    """
    page = _page()
    vp = page.viewport_size or {"width": 1280, "height": 720}
    elements = await page.evaluate("""
    ([vpW, vpH]) => {
        const results = [];
        let order = 0;
        // Generic safety net: ANY site's custom elements/JS could throw while
        // we read properties off them. Per-element try/catch skips just that
        // element; the outer try/catch falls through to whatever was already
        // collected if the walk itself is interrupted.
        try {
            const tags = ['input', 'button', 'a', 'select', 'textarea'];
            // Use document order (TreeWalker keeps DOM order across all tag types)
            const all = Array.from(document.querySelectorAll(tags.join(',')));
            all.forEach(el => {
                try {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' ||
                        style.opacity === '0') return;
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 && rect.height === 0) return;

                    order++;
                    const tag = el.tagName.toLowerCase();
                    const info = { order, tag };

                    if (el.type)        info.type        = el.type;
                    if (el.name)        info.name        = el.name;
                    if (el.id)          info.id          = el.id;
                    if (el.placeholder) info.placeholder = el.placeholder;

                    // Track which attribute provided the label so the selector uses the right one
                    const ariaLabel = el.getAttribute('aria-label');
                    const titleAttr = el.getAttribute('title');
                    if (ariaLabel)       { info.aria_label = ariaLabel; info._label_attr = 'aria-label'; }
                    else if (titleAttr)  { info.aria_label = titleAttr; info._label_attr = 'title'; }

                    const text = (el.innerText || el.textContent || '').trim().slice(0, 100);
                    if (text) info.text = text;

                    // href is useful for links — lets the agent navigate directly
                    if (tag === 'a' && el.getAttribute('href')) info.href = el.getAttribute('href');

                    // Position data
                    info.x      = Math.round(rect.left);
                    info.y      = Math.round(rect.top);
                    info.width  = Math.round(rect.width);
                    info.height = Math.round(rect.height);

                    // In-viewport check
                    info.in_viewport = (
                        rect.top < vpH && rect.bottom > 0 &&
                        rect.left < vpW && rect.right > 0
                    );

                    // Rough zone hint so the LLM can reason about layout
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top  + rect.height / 2;
                    const vZone = cy < vpH * 0.33 ? 'top' : cy < vpH * 0.66 ? 'middle' : 'bottom';
                    const hZone = cx < vpW * 0.33 ? 'left' : cx < vpW * 0.66 ? 'center' : 'right';
                    info.position_hint = `${vZone}-${hZone}`;

                    // Build a unique selector using the CORRECT attribute.
                    // Use JSON.stringify for safe quoting (handles apostrophes, backslashes).
                    const attrVal = v => JSON.stringify(v).slice(1, -1);  // strip outer quotes
                    let selector = tag;
                    if (info._label_attr && info.aria_label)
                                             selector = `${tag}[${info._label_attr}="${attrVal(info.aria_label)}"]`;
                    else if (el.id)          selector = `${tag}#${CSS.escape(el.id)}`;
                    else if (el.name)        selector = `${tag}[name="${attrVal(el.name)}"]`;
                    else if (el.placeholder) selector = `${tag}[placeholder="${attrVal(el.placeholder)}"]`;
                    else if (el.type && el.type !== 'submit' && el.type !== 'button')
                                             selector = `${tag}[type='${el.type}']`;
                    else if (tag === 'a' && info.href)
                                             selector = `a[href="${attrVal(info.href)}"]`;

                    // Disambiguate if selector still matches multiple elements
                    try {
                        if (document.querySelectorAll(selector).length > 1) {
                            const idx = Array.from(document.querySelectorAll(selector)).indexOf(el);
                            if (idx > 0) selector = `${selector} >> nth=${idx}`;
                        }
                    } catch (e) { /* selector had characters querySelectorAll can't parse — leave as-is */ }
                    info.selector = selector;

                    results.push(info);
                } catch (e) {
                    // This one element's properties threw for some site-specific
                    // reason — skip it and keep processing the rest.
                }
            });
        } catch (e) {
            // Unexpected failure walking the page — fall through and return
            // whatever elements were already collected.
        }
        return results.slice(0, 5000);
    }
    """, [vp["width"], vp["height"]])
    total = len(elements)
    total_chunks = max(1, -(-total // chunk_size))
    start = chunk_index * chunk_size
    return {
        "url":            page.url,
        "title":          await page.title(),
        "viewport":       vp,
        "chunk_index":    chunk_index,
        "total_chunks":   total_chunks,
        "total_elements": total,
        "elements":       elements[start:start + chunk_size],
    }

async def go_back() -> dict:
    """Navigate to the previous page in browser history."""
    page = _page()
    await page.go_back()
    return {"status": "ok", "url": page.url}


async def _get_page_html(
    max_chars: int = 12000, container_selector: str = "body", chunk_index: int = 0
) -> dict:
    """
    Return semantically cleaned HTML of the current page (or a subtree).
    Strips navigation chrome, comments, and noise attributes so the budget
    is spent on the content region the agent actually needs.

    The cleaned HTML is paginated, not truncated — if it's longer than
    max_chars, call again with a higher chunk_index to page through the
    rest. Nothing is ever permanently lost; total_chunks tells you how
    many chunks exist.

    Args:
        max_chars: Characters per chunk (default 12000)
        container_selector: Restrict to a subtree, e.g. 'ytd-search' for search
                            results, 'ytd-video-renderer' for the first result.
                            Default 'body' returns the whole page.
        chunk_index: Which chunk to return (0-based)

    Returns:
        {"url": ..., "title": ..., "html": <chunk>, "chunk_index": ...,
         "total_chunks": ...}
    """
    page = _page()
    html = await page.evaluate("""
    ([root, maxLen]) => {
        // container_selector must be a real CSS selector (not aria-ref=eN, which
        // only Playwright's own locator engine understands) — fall back to body
        // instead of throwing a JS syntax error if it isn't.
        let src;
        try { src = document.querySelector(root) || document.body; }
        catch (e) { src = document.body; }

        // Everything below is wrapped in try/catch: ANY site's own JS/custom
        // elements could throw somewhere in this pipeline for reasons specific
        // to that site (seen on YouTube's Web Components; could happen on any
        // other site built the same way) — if it does, fall back to raw,
        // unfiltered HTML instead of failing the whole extraction. This is a
        // generic safety net, not a per-site patch.
        try {
            const clone = src.cloneNode(true);

            // 1. Remove page-chrome and noise tags
            [
                'script','style','svg','noscript','iframe','canvas',
                'picture','video','audio','source','track','template',
                // YouTube-specific shell elements that eat all the budget
                'ytd-masthead','tp-yt-app-drawer','ytd-mini-guide-renderer',
                'ytd-miniplayer','ytd-popup-container','ytd-ad-slot-renderer',
                'ytd-promoted-video-renderer','ytd-in-feed-ad-layout-renderer',
                'ytd-third-party-manager','ytd-permission-role-bottom-bar-renderer',
            ].forEach(tag => {
                clone.querySelectorAll(tag).forEach(el => el.remove());
            });

            // 2. Remove [hidden] elements — do this BEFORE attribute filtering
            //    so hidden="" is still queryable at this point
            clone.querySelectorAll('[hidden]').forEach(el => el.remove());

            // 2b. Drop our own fake-cursor host — it is a UI overlay we injected,
            //     not part of the page, and must never reach the agent
            clone.querySelectorAll('[data-pw-overlay]').forEach(el => el.remove());

            // 3. Strip all HTML comments
            const stripComments = (node) => {
                for (let i = node.childNodes.length - 1; i >= 0; i--) {
                    const child = node.childNodes[i];
                    if (child.nodeType === 8) node.removeChild(child);
                    else if (child.nodeType === 1) stripComments(child);
                }
            };
            stripComments(clone);

            // 4. Keep only semantic attributes
            //    'hidden' excluded — hidden elements already removed above
            const KEEP_ATTRS = new Set([
                'id','name','aria-label','aria-pressed','aria-disabled',
                'aria-expanded','aria-haspopup','aria-current','role',
                'href','src','type','placeholder','title','value',
                'action','for','checked','selected','disabled',
                'multiple','readonly','required'
            ]);
            clone.querySelectorAll('*').forEach(el => {
                Array.from(el.attributes).forEach(attr => {
                    if (!KEEP_ATTRS.has(attr.name)) {
                        // Cloning a custom element (common on modern sites, e.g. YouTube's
                        // ytd-*/yt-* Web Components) still produces a live, upgraded
                        // instance — removeAttribute can trigger the site's OWN
                        // attributeChangedCallback, which may throw reading internal
                        // state the orphaned clone never had. Skip that one attribute
                        // rather than losing the whole HTML extraction.
                        try {
                            el.removeAttribute(attr.name);
                        } catch (e) { /* leave this attribute in place */ }
                    }
                });
            });

            // 5. Collapse whitespace
            const raw = clone.innerHTML
                .replace(/\\n\\s*\\n/g, '\\n')
                .replace(/[ \\t]+/g, ' ')
                .trim();
            return raw;
        } catch (e) {
            try {
                return src.innerHTML;
            } catch (e2) {
                return '';
            }
        }
    }
    """, [container_selector, max_chars])
    total_chunks = max(1, -(-len(html) // max_chars))
    start = chunk_index * max_chars
    return {
        "url":          page.url,
        "title":        await page.title(),
        "html":         html[start:start + max_chars],
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
    }

async def go_forward() -> dict:
    """Navigate to the next page in browser history."""
    await _page().go_forward()
    return {"status": "ok", "url": _page().url}


async def reload(wait_until: str = "load") -> dict:
    """
    Reload the current page.

    Args:
        wait_until: "load", "domcontentloaded", "networkidle", or "commit"
    """
    await _page().reload(wait_until=wait_until)
    return {"status": "ok", "url": _page().url}


async def get_current_url() -> dict:
    """Return the current page URL and title."""
    page = _page()
    return {"url": page.url, "title": await page.title()}


# ---------------------------------------------------------------------------
# CLICKING & INTERACTION
# ---------------------------------------------------------------------------

async def click(
    selector: str,
    button: str = "left",
    click_count: int = 1,
    timeout: int = 10000,
) -> dict:
    """
    Click an element on the page.
    If the click causes navigation to a new URL, automatically returns a fresh
    page snapshot so the next tool call can use real selectors for the new page.

    Args:
        selector: CSS selector, XPath, or text selector (e.g. "button#submit", "text=Login")
        button: Mouse button — "left", "right", or "middle"
        click_count: Number of clicks (2 for double-click)
        timeout: Max wait time in milliseconds

    Returns:
        {"status": "ok", "selector": selector, "navigated": bool,
         "url": ..., "title": ..., "structure": <fresh page outline>}
    """
    page = _page()
    url_before = page.url
    await page.click(selector, button=button, click_count=click_count, timeout=timeout)

    # Wait briefly for DOM changes (navigation, dropdown open, SPA route change)
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass

    # One page view, not two. This used to return _get_page_html() AND
    # _get_page_snapshot() on every single click — 9,452 tok measured, of which
    # the element list alone was 5,972 and survived history trimming forever.
    _invalidate_index()
    view = await _page_view()
    return {
        "status":    "ok",
        "selector":  selector,
        "navigated": view.get("url") != url_before,
        **view,
    }


async def hover(selector: str, timeout: int = 30000) -> dict:
    """
    Hover the mouse over an element.

    Args:
        selector: Element selector
        timeout: Max wait time in milliseconds
    """
    await _page().hover(selector, timeout=timeout)
    return {"status": "ok", "selector": selector}


async def focus(selector: str, timeout: int = 30000) -> dict:
    """
    Focus a DOM element (e.g. an input field).

    Args:
        selector: Element selector
        timeout: Max wait time in milliseconds
    """
    await _page().focus(selector, timeout=timeout)
    return {"status": "ok", "selector": selector}


async def drag_and_drop(source_selector: str, target_selector: str, timeout: int = 30000) -> dict:
    """
    Drag an element and drop it onto another element.

    Args:
        source_selector: Selector of the element to drag
        target_selector: Selector of the drop target
        timeout: Max wait time in milliseconds
    """
    await _page().drag_and_drop(source_selector, target_selector, timeout=timeout)
    return {"status": "ok", "source": source_selector, "target": target_selector}


# ---------------------------------------------------------------------------
# TYPING & INPUT
# ---------------------------------------------------------------------------

async def fill(selector: str, value: str, timeout: int = 60000) -> dict:
    """
    Set the value of an input field instantly (clears first).

    Args:
        selector: Input element selector
        value: Value to set
        timeout: Max wait time in milliseconds
    """
    await _page().fill(selector, value, timeout=timeout)
    return {"status": "ok", "selector": selector, "value": value}


async def type_text(selector: str, text: str, delay: int = 0, timeout: int = 60000) -> dict:
    """
    Type text into an input field character by character (simulates real typing).

    Args:
        selector: Input element selector
        text: Text to type
        delay: Delay between keypresses in milliseconds
        timeout: Max wait time in milliseconds
    """
    await _page().type(selector, text, delay=delay, timeout=timeout)
    return {"status": "ok", "selector": selector, "text": text}


async def press_key(selector: str, key: str, timeout: int = 60000) -> dict:
    """
    Press a keyboard key while focused on an element.

    Args:
        selector: Element selector to focus before pressing
        key: Key to press (e.g. "Enter", "Tab", "Escape", "ArrowDown")
        timeout: Max wait time in milliseconds
    """
    await _page().press(selector, key, timeout=timeout)
    return {"status": "ok", "selector": selector, "key": key}


async def select_option(
    selector: str,
    value: Optional[str] = None,
    label: Optional[str] = None,
    index: Optional[int] = None,
    timeout: int = 30000,
) -> dict:
    """
    Select an option in a <select> dropdown.

    Args:
        selector: The <select> element selector
        value: Option value attribute to select
        label: Option visible text to select
        index: Option index (0-based) to select
        timeout: Max wait time in milliseconds

    Note: Provide exactly one of value, label, or index.
    """
    kwargs: dict[str, Any] = {"timeout": timeout}
    if value is not None:
        kwargs["value"] = value
    elif label is not None:
        kwargs["label"] = label
    elif index is not None:
        kwargs["index"] = index

    selected = await _page().select_option(selector, **kwargs)
    return {"status": "ok", "selector": selector, "selected": selected}


async def check_checkbox(selector: str, timeout: int = 30000) -> dict:
    """
    Check a checkbox or radio button.

    Args:
        selector: Checkbox/radio element selector
        timeout: Max wait time in milliseconds
    """
    await _page().check(selector, timeout=timeout)
    return {"status": "ok", "selector": selector, "checked": True}


async def uncheck_checkbox(selector: str, timeout: int = 30000) -> dict:
    """
    Uncheck a checkbox.

    Args:
        selector: Checkbox element selector
        timeout: Max wait time in milliseconds
    """
    await _page().uncheck(selector, timeout=timeout)
    return {"status": "ok", "selector": selector, "checked": False}


async def upload_file(selector: str, file_paths: list[str], timeout: int = 30000) -> dict:
    """
    Upload one or more files via a file input element.

    Args:
        selector: File input element selector
        file_paths: List of absolute file paths to upload
        timeout: Max wait time in milliseconds
    """
    await _page().set_input_files(selector, file_paths, timeout=timeout)
    return {"status": "ok", "selector": selector, "files": file_paths}


# ---------------------------------------------------------------------------
# READING PAGE CONTENT
# ---------------------------------------------------------------------------

async def get_page_structure(
    container_selector: str = "body",
    max_chars: int = 8000,
    chunk_index: int = 0,
    depth: Optional[int] = None,
) -> dict:
    """
    THE FIRST TOOL TO CALL on any new page. Returns a compact, nested outline
    of roles, accessible names, and visible text (Playwright's built-in
    AI-optimized ARIA snapshot) — NOT raw HTML and NOT a flat element list.
    This shows how elements are grouped/nested (e.g. "this button is inside
    this form, inside this dialog") AND their actual visible text content
    (view counts, prices, dates, titles — anything readable), all in one
    compact document.

    THIS IS THE ONLY PAGE-WIDE VIEW. There is no raw-HTML tool and no flat
    element-list tool — every action (click, fill, navigate, ...) returns this
    same outline for the page it left you on, so you rarely need to call it
    yourself except on a page you have not acted on yet.

    Every element is tagged with a short reference like [ref=e12]. Pass that
    reference DIRECTLY as any tool's selector argument in the form
    "aria-ref=e12" (e.g. click(selector="aria-ref=e12")) — no need to
    construct or copy a CSS selector. Refs are only valid until the next
    page mutation/navigation; call this again after any action to get fresh
    refs.

    For things this outline doesn't carry: search_elements(terms) to filter a
    huge page by keyword, expand_element(selector) for one element's full
    detail (including its href and pixel box), get_text_blocks()/get_text()
    to read specific content.

    Paginated like the other page-reading tools — never silently truncated;
    if total_chunks > 1, call again with a higher chunk_index for the rest.

    Args:
        container_selector: Restrict to a subtree, e.g. 'form#login'. Default 'body'.
        max_chars: Characters per chunk (default 8000)
        chunk_index: Which chunk to return (0-based)
        depth: Optional max nesting depth, to shrink the output on huge pages

    Returns:
        {"structure": <chunk>, "chunk_index": ..., "total_chunks": ...}
        or {"error": ...} if the container doesn't resolve.
    """
    kwargs: dict[str, Any] = {"mode": "ai"}
    if depth is not None:
        kwargs["depth"] = depth
    try:
        structure = await _page().locator(container_selector).aria_snapshot(**kwargs)
    except Exception as e:
        return {"error": f"Could not build structure for '{container_selector}': {e}"}

    total_chunks = max(1, -(-len(structure) // max_chars))
    start = chunk_index * max_chars
    return {
        "structure":    structure[start:start + max_chars],
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
    }


async def get_text_blocks(
    container_selector: str = "body", chunk_index: int = 0, chunk_size: int = 60
) -> dict:
    """
    Return visible text-bearing leaf elements (spans, divs, headings, yt-formatted-string)
    with their text and a usable selector. Use this when you need to READ data such as
    view counts, dates, prices, labels — content that is not a clickable element and
    therefore are not clickable elements in the page outline.

    Results are paginated, not truncated — if there are more than chunk_size
    blocks, call again with a higher chunk_index to see the rest. Nothing is
    ever permanently lost; total_chunks tells you how many pages exist.

    Args:
        container_selector: Restrict to a subtree, e.g. 'ytd-video-renderer' for the
                            first search result, or 'body' for the whole page.
        chunk_index: Which page of blocks to return (0-based)
        chunk_size: Blocks per page (default 60)

    Returns:
        {"container": ..., "chunk_index": ..., "total_chunks": ..., "total_blocks": ...,
         "blocks": [{"text": ..., "selector": ..., "x": ..., "y": ...}]}
    """
    page = _page()
    blocks = await page.evaluate("""
    ([root]) => {
        // container_selector must be a real CSS selector (not aria-ref=eN, which
        // only Playwright's own locator engine understands) — fall back to body
        // instead of throwing a JS syntax error if it isn't.
        let host;
        try { host = document.querySelector(root) || document.body; }
        catch (e) { host = document.body; }
        const out  = [];
        // Generic safety net: ANY site's custom elements/JS could throw while
        // we read properties off them. Per-element try/catch skips just that
        // element; the outer try/catch falls through to whatever was already
        // collected if the walk itself is interrupted. The 5000 ceiling below
        // is a safety net against pathological pages, not a normal-use limit —
        // pagination (chunk_index/chunk_size) covers everything under it.
        try {
            // Include a, button, li, td, th so nav items and table cells appear
            const tags = 'span,div,p,li,td,th,h1,h2,h3,h4,h5,a,button,label,yt-formatted-string,ytd-video-meta-block';
            for (const el of host.querySelectorAll(tags)) {
                if (out.length >= 5000) break;
                try {
                    if (el.children.length > 0) continue;          // leaf nodes only
                    const t = (el.innerText || '').trim();
                    if (!t || t.length > 120) continue;
                    const r = el.getBoundingClientRect();
                    if (r.width === 0 || r.height === 0) continue;
                    const s = window.getComputedStyle(el);
                    if (s.display === 'none' || s.visibility === 'hidden') continue;

                    let sel = el.tagName.toLowerCase();
                    if (el.id) sel = `#${CSS.escape(el.id)}`;
                    try {
                        const all = document.querySelectorAll(sel);
                        if (all.length > 1) {
                            const idx = Array.from(all).indexOf(el);
                            sel = `${sel} >> nth=${idx}`;
                        }
                    } catch (e) { /* selector had characters querySelectorAll can't parse — leave as-is */ }
                    const info = { text: t, selector: sel, x: Math.round(r.left), y: Math.round(r.top) };
                    // Include href for links so agent can navigate directly
                    if (el.tagName === 'A' && el.getAttribute('href')) info.href = el.getAttribute('href');
                    out.push(info);
                } catch (e) {
                    // This one element's properties threw for some site-specific
                    // reason — skip it and keep processing the rest.
                }
            }
        } catch (e) {
            // Unexpected failure walking the page — fall through and return
            // whatever blocks were already collected.
        }
        return out;
    }
    """, [container_selector])
    total = len(blocks)
    total_chunks = max(1, -(-total // chunk_size))
    start = chunk_index * chunk_size
    return {
        "container":    container_selector,
        "blocks":       blocks[start:start + chunk_size],
        "chunk_index":  chunk_index,
        "total_chunks": total_chunks,
        "total_blocks": total,
    }


async def get_text(selector: str, timeout: int = 30000) -> dict:
    """
    Get the visible inner text of an element.

    If the selector matches more than one element, this returns text from
    ALL of them instead of failing (Playwright's single-element read would
    otherwise throw a strict-mode error) — so it succeeds on the first call
    no matter which selector/tool combination was picked. Use get_all_text
    directly when you already know you want every match.

    Args:
        selector: Element selector
        timeout: Max wait time in milliseconds

    Returns:
        {"text": <inner text>} for a single match, or
        {"text": <first match's text>, "texts": [<all matches>],
         "note": "selector matched N elements — texts[] has all of them"}
        if the selector matched more than one element.
    """
    locator = _page().locator(selector)
    count = await locator.count()
    if count > 1:
        texts = await locator.all_inner_texts()
        return {
            "text": texts[0] if texts else "",
            "texts": texts,
            "note": f"selector matched {count} elements — texts[] has all of them",
        }
    text = await locator.inner_text(timeout=timeout)
    return {"text": text}


async def get_attribute(selector: str, attribute: str, timeout: int = 30000) -> dict:
    """
    Get the value of an HTML attribute on an element.

    Args:
        selector: Element selector
        attribute: Attribute name (e.g. "href", "src", "value", "class")
        timeout: Max wait time in milliseconds

    Returns:
        {"attribute": attribute, "value": <attribute value>}
    """
    value = await _page().get_attribute(selector, attribute, timeout=timeout)
    return {"attribute": attribute, "value": value}


async def get_input_value(selector: str, timeout: int = 30000) -> dict:
    """
    Get the current value of an input, textarea, or select element.

    Args:
        selector: Input element selector
        timeout: Max wait time in milliseconds
    """
    value = await _page().input_value(selector, timeout=timeout)
    return {"value": value}


async def get_page_content() -> dict:
    """
    Get the full HTML source of the current page.

    Returns:
        {"html": <full page HTML>}
    """
    html = await _page().content()
    return {"html": html}


async def get_all_text(selector: str) -> dict:
    """
    Get all text nodes from all elements matching the selector.

    Args:
        selector: Element selector (may match multiple elements)

    Returns:
        {"texts": [list of text strings]}
    """
    texts = await _page().locator(selector).all_inner_texts()
    return {"texts": texts}


# ---------------------------------------------------------------------------
# SEARCH & DISCOVERY (indexed DOM)
# ---------------------------------------------------------------------------
#
# _build_page_index() walks the WHOLE page once and records every element
# worth knowing about (interactive controls, headings, anything with an
# id/aria-label/role/name/placeholder, and leaf text nodes) into a flat list
# kept server-side in _state["page_index"] — never sent to the LLM directly.
# It is rebuilt lazily whenever it's None (see _invalidate_index(), called
# after navigate()/new_tab()/switch_tab() and after any page-mutating tool
# executes — see execute_tool_call() in llm_agent.py).
#
# search_elements() and expand_element() are the only things that read this
# index — they let the agent search/inspect the ENTIRE page deterministically
# without ever dumping the entire page into the prompt.

async def _build_page_index() -> list[dict]:
    """Walk the DOM once and build the flat element index (internal helper)."""
    page = _page()
    vp = page.viewport_size or {"width": 1280, "height": 720}
    nodes = await page.evaluate("""
    ([vpW, vpH]) => {
        const results = [];
        let order = 0;
        const attrVal = v => JSON.stringify(v).slice(1, -1);

        const buildSelector = (el, tag) => {
            let selector = tag;
            const ariaLabel = el.getAttribute('aria-label');
            const titleAttr = el.getAttribute('title');
            if (ariaLabel)                          selector = `${tag}[aria-label="${attrVal(ariaLabel)}"]`;
            else if (titleAttr)                     selector = `${tag}[title="${attrVal(titleAttr)}"]`;
            else if (el.id)                         selector = `${tag}#${CSS.escape(el.id)}`;
            else if (el.name)                       selector = `${tag}[name="${attrVal(el.name)}"]`;
            else if (el.getAttribute('placeholder')) selector = `${tag}[placeholder="${attrVal(el.getAttribute('placeholder'))}"]`;
            else if (tag === 'a' && el.getAttribute('href'))
                                                     selector = `a[href="${attrVal(el.getAttribute('href'))}"]`;
            try {
                if (document.querySelectorAll(selector).length > 1) {
                    const idx = Array.from(document.querySelectorAll(selector)).indexOf(el);
                    if (idx > 0) selector = `${selector} >> nth=${idx}`;
                }
            } catch (e) { /* selector had characters querySelectorAll can't parse — leave as-is */ }
            return selector;
        };

        const INTERACTIVE = new Set(['input', 'button', 'a', 'select', 'textarea']);
        const HEADING     = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']);
        const SKIP_TAGS    = new Set(['script', 'style', 'svg', 'noscript', 'template']);
        const all = Array.from(document.querySelectorAll('*'));

        for (const el of all) {
            try {
                if (results.length >= 5000) break;
                const tag = el.tagName.toLowerCase();
                if (SKIP_TAGS.has(tag)) continue;
                if (el.hasAttribute('data-pw-overlay')) continue;  // our fake-cursor host

                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') continue;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 && rect.height === 0) continue;

                const id_        = el.id || '';
                const name_       = el.name || '';
                const ariaLabel   = el.getAttribute('aria-label') || el.getAttribute('title') || '';
                const role        = el.getAttribute('role') || '';
                const placeholder = el.getAttribute('placeholder') || '';
                const className   = (el.className && typeof el.className === 'string') ? el.className : '';
                const isInteractive = INTERACTIVE.has(tag);
                const isHeading      = HEADING.has(tag);
                const hasIdentity    = !!(id_ || ariaLabel || role || name_ || placeholder);

                let kind = null;
                if (isInteractive) kind = 'interactive';
                else if (isHeading) kind = 'heading';
                else if (hasIdentity) kind = 'landmark';

                let text = '';
                if (el.children.length === 0) {
                    text = (el.innerText || el.textContent || '').trim().slice(0, 150);
                    if (text && !kind) kind = 'text';
                }

                if (!kind) continue;  // skip generic, unidentifiable wrapper elements

                order++;
                const info = {
                    order, tag, kind,
                    id: id_, name: name_, aria_label: ariaLabel, role, placeholder,
                    class_name: className.toString().slice(0, 100),
                    text,
                    selector: buildSelector(el, tag),
                };
                if (tag === 'a' && el.getAttribute('href')) info.href = el.getAttribute('href');
                info.x = Math.round(rect.left);
                info.y = Math.round(rect.top);
                results.push(info);
            } catch (e) {
                // This one element's properties threw for some site-specific
                // reason (generic risk on any custom-element-heavy site) —
                // skip it and keep walking the rest of the page.
            }
        }
        return results;
    }
    """, [vp["width"], vp["height"]])
    return nodes


async def _ensure_page_index() -> list[dict]:
    """Return the cached page index, rebuilding it if it's stale (internal helper)."""
    if _state.get("page_index") is None:
        _state["page_index"] = await _build_page_index()
    return _state["page_index"]


# Common intent -> synonym expansions, so a single guessed word like "login"
# also matches pages that label things "sign in" / "log in" / etc.
_SEARCH_SYNONYMS: dict[str, list[str]] = {
    "login":  ["login", "log in", "sign in", "signin"],
    "signup": ["sign up", "signup", "register", "create account"],
    "search": ["search", "find", "query"],
    "submit": ["submit", "send", "confirm", "continue", "ok"],
    "cancel": ["cancel", "close", "dismiss"],
    "close":  ["close", "dismiss", "x"],
    "menu":   ["menu", "navigation", "nav"],
    "next":   ["next", "forward", "continue"],
    "back":   ["back", "previous", "return"],
}

# Fields checked for every element, weighted — id/name/aria-label/placeholder
# matches are a much stronger signal than a generic text/class match, but
# plain visible text is still weighted enough to compete for data-extraction
# searches (view counts, prices, dates) where there's no labeled control at all.
_SEARCH_FIELD_WEIGHTS: dict[str, float] = {
    "id": 3, "name": 3, "aria_label": 3, "placeholder": 3,
    "role": 2,
    "text": 2, "class_name": 1, "href": 1,
}


def _stem(word: str) -> str:
    """Strip a simple trailing English plural so "view"/"views" compare equal."""
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _term_value_score(term: str, value: str) -> float:
    """
    Score how well one guessed term matches one field value, as a multiplier
    in [0, 1]. Handles both directions of substring containment (a short
    guess like "login" inside a longer label, AND a longer descriptive guess
    like "more information" containing a shorter label like "learn more"),
    shared-word overlap (plural-insensitive, e.g. "view" matches "views"),
    and per-word fuzzy matching for typos/near-misses.
    """
    if not term or not value:
        return 0.0
    if term in value or value in term:
        return 1.0

    term_words = set(term.split())
    value_words = set(value.split())
    if term_words & value_words:
        return 0.7
    if {_stem(w) for w in term_words} & {_stem(w) for w in value_words}:
        return 0.65

    best_ratio = SequenceMatcher(None, term, value[:60]).ratio()
    for vw in value_words:
        for tw in term_words:
            best_ratio = max(best_ratio, SequenceMatcher(None, tw, vw).ratio())
    return 0.5 if best_ratio > 0.6 else 0.0


async def search_elements(terms: list[str], top_k: int = 10) -> dict:
    """
    Deterministically search the ENTIRE page for elements matching guessed
    terms (e.g. ["login", "sign in", "email"]) — no LLM involved. Matches
    each term against every element's id/name/aria-label/placeholder/role/
    text/class via substring matching (either direction) and shared-word
    overlap, with a per-word fuzzy fallback for near-misses (typos), so the
    search covers the whole page even though only the best matches are
    returned. Common intents (login, signup, search, submit, cancel, close,
    menu, next, back) are auto-expanded with synonyms.

    Args:
        terms: Guessed words/phrases describing what you're looking for
        top_k: Max number of matches to return (default 10)

    Returns:
        {"terms": [...], "matches": [{"selector":..., "tag":..., "kind":...,
         "label":..., "text_snippet":..., "score":...}, ...]}
    """
    nodes = await _ensure_page_index()

    expanded: set[str] = set()
    for t in terms:
        t_norm = t.strip().lower()
        if not t_norm:
            continue
        expanded.add(t_norm)
        expanded.update(_SEARCH_SYNONYMS.get(t_norm, []))

    scored: list[tuple[float, dict]] = []
    for node in nodes:
        score = 0.0
        for field, weight in _SEARCH_FIELD_WEIGHTS.items():
            value = str(node.get(field, "") or "").lower()
            if not value:
                continue
            for term in expanded:
                score += weight * _term_value_score(term, value)
        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    matches = []
    for score, node in scored[:top_k]:
        label = node.get("aria_label") or node.get("placeholder") or node.get("name") or node.get("id") or ""
        matches.append({
            "selector":     node.get("selector"),
            "tag":          node.get("tag"),
            "kind":         node.get("kind"),
            "label":        label,
            "text_snippet": (node.get("text") or "")[:100],
            "score":        round(score, 1),
        })

    return {"terms": sorted(expanded), "matches": matches}


async def expand_element(selector: str) -> dict:
    """
    Return full detail for ONE element you already have a selector for
    (typically copied from search_elements results) — its complete text,
    key attributes, and the text of its immediate children. Use this to
    confirm a candidate before acting on it, or to read a full paragraph/
    value that was truncated elsewhere.

    Args:
        selector: Element selector, typically copied from search_elements results

    Returns:
        {"selector":..., "tag":..., "attributes": {...}, "text": ...,
         "children_text": [...]}
        or {"error": ...} if the selector does not resolve.
    """
    page = _page()
    locator = page.locator(selector)
    try:
        count = await locator.count()
    except Exception as e:
        return {"error": f"Invalid selector '{selector}': {e}"}
    if count == 0:
        return {"error": f"Selector '{selector}' matches no elements."}

    first = locator.first
    try:
        detail = await first.evaluate("""
        (el) => {
            const attrs = {};
            for (const a of el.attributes) attrs[a.name] = a.value;
            const children_text = Array.from(el.children)
                .map(c => (c.innerText || c.textContent || '').trim())
                .filter(Boolean)
                .slice(0, 20);
            return {
                tag: el.tagName.toLowerCase(),
                attributes: attrs,
                text: (el.innerText || el.textContent || '').trim().slice(0, 3000),
                children_text,
            };
        }
        """)
    except Exception as e:
        return {"error": f"Could not read element '{selector}': {e}"}

    return {"selector": selector, **detail}


# ---------------------------------------------------------------------------
# WAITING
# ---------------------------------------------------------------------------

async def wait_for_selector(
    selector: str,
    state: str = "visible",
    timeout: int = 60000,
) -> dict:
    """
    Wait until an element reaches a given state.

    Args:
        selector: Element selector
        state: "visible", "hidden", "attached", or "detached"
        timeout: Max wait time in milliseconds
    """
    await _page().wait_for_selector(selector, state=state, timeout=timeout)
    return {"status": "ok", "selector": selector, "state": state}


async def wait_for_url(url_pattern: str, timeout: int = 30000) -> dict:
    """
    Wait until the page URL matches a pattern.

    Args:
        url_pattern: Exact URL string or glob pattern (e.g. "**/dashboard*")
        timeout: Max wait time in milliseconds
    """
    await _page().wait_for_url(url_pattern, timeout=timeout)
    return {"status": "ok", "url": _page().url}


async def wait_for_load_state(state: str = "load", timeout: int = 30000) -> dict:
    """
    Wait for the page to reach a specific load state.

    Args:
        state: "load", "domcontentloaded", or "networkidle"
        timeout: Max wait time in milliseconds
    """
    await _page().wait_for_load_state(state, timeout=timeout)
    return {"status": "ok", "state": state}


async def wait_for_timeout(milliseconds: int) -> dict:
    """
    Pause execution for a fixed amount of time.

    Args:
        milliseconds: How long to wait in milliseconds

    Note: Prefer wait_for_selector or wait_for_load_state over this where possible.
    """
    await _page().wait_for_timeout(milliseconds)
    return {"status": "ok", "waited_ms": milliseconds}


# ---------------------------------------------------------------------------
# NETWORK
# ---------------------------------------------------------------------------

async def fetch_url(
    url: str,
    method: str = "GET",
    headers: Optional[dict] = None,
    body: Optional[str] = None,
) -> dict:
    """
    Make a direct HTTP request without navigating the browser.

    Args:
        url: Request URL
        method: HTTP method — "GET", "POST", "PUT", "DELETE", "PATCH"
        headers: Optional dict of request headers
        body: Optional request body string (for POST/PUT/PATCH)

    Returns:
        {"status": <http status code>, "body": <response text>}
    """
    context = _state["context"]
    if context is None:
        raise RuntimeError("No active browser session. Call start_browser() first.")

    kwargs: dict[str, Any] = {"method": method}
    if headers:
        kwargs["headers"] = headers
    if body:
        kwargs["data"] = body

    response = await context.request.fetch(url, **kwargs)
    body_text = await response.text()
    return {"status": response.status, "body": body_text}


async def intercept_route(url_pattern: str, mock_body: str, status: int = 200) -> dict:
    """
    Intercept network requests matching a URL pattern and return a mock response.
    Call this BEFORE navigating to the page.

    Args:
        url_pattern: Glob pattern to match (e.g. "**/api/users*")
        mock_body: Response body to return (JSON string or plain text)
        status: HTTP status code to return

    Returns:
        {"status": "ok", "pattern": url_pattern}
    """
    async def handler(route):
        await route.fulfill(status=status, body=mock_body)

    await _page().route(url_pattern, handler)
    return {"status": "ok", "pattern": url_pattern}


async def abort_route(url_pattern: str) -> dict:
    """
    Abort all network requests matching a URL pattern (simulates offline/blocked).

    Args:
        url_pattern: Glob pattern to match (e.g. "**/analytics*")
    """
    await _page().route(url_pattern, lambda route: route.abort())
    return {"status": "ok", "pattern": url_pattern}


# ---------------------------------------------------------------------------
# SCREENSHOTS & MEDIA
# ---------------------------------------------------------------------------

async def screenshot(
    path: str,
    full_page: bool = False,
    selector: Optional[str] = None,
    timeout: int = 30000,
) -> dict:
    """
    Take a screenshot of the page or a specific element.

    Args:
        path: File path to save the screenshot (e.g. "screenshots/home.png")
        full_page: Capture the full scrollable page (ignored if selector is set)
        selector: If set, screenshot only this element
        timeout: Max wait time in milliseconds

    Returns:
        {"status": "ok", "path": path}
    """
    if selector:
        element = _page().locator(selector)
        await element.screenshot(path=path, timeout=timeout)
    else:
        await _page().screenshot(path=path, full_page=full_page)
    return {"status": "ok", "path": path}


async def save_pdf(path: str) -> dict:
    """
    Export the current page as a PDF file (Chromium only).

    Args:
        path: File path to save the PDF (e.g. "output/page.pdf")

    Returns:
        {"status": "ok", "path": path}
    """
    await _page().pdf(path=path)
    return {"status": "ok", "path": path}


# ---------------------------------------------------------------------------
# JAVASCRIPT EXECUTION
# ---------------------------------------------------------------------------

async def evaluate_js(script: str) -> dict:
    """
    Execute JavaScript in the browser context and return the result.

    Args:
        script: JavaScript expression or function body to evaluate.
                For a function, use: "() => document.title"
                For an expression: "document.querySelectorAll('a').length"

    Returns:
        {"result": <return value of the script>}
    """
    result = await _page().evaluate(script)
    # If the script navigated (e.g. window.location.href = ...), settle and report actual URL
    try:
        await _page().wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass
    return {"result": result, "url": _page().url, "title": await _page().title()}


async def evaluate_js_on_element(selector: str, script: str, timeout: int = 30000) -> dict:
    """
    Execute JavaScript on a specific DOM element.

    Args:
        selector: Element selector
        script: JS function receiving the element as first argument.
                Example: "(el) => el.getAttribute('data-id')"
        timeout: Max wait time in milliseconds

    Returns:
        {"result": <return value>}
    """
    element = await _page().wait_for_selector(selector, timeout=timeout)
    result = await element.evaluate(script)
    return {"result": result}


# ---------------------------------------------------------------------------
# FRAMES & IFRAMES
# ---------------------------------------------------------------------------

async def switch_to_frame(frame_selector: str) -> dict:
    """
    Get a handle to an iframe so subsequent actions target it.
    Note: After calling this, use frame-specific tool calls or evaluate_js.

    Args:
        frame_selector: CSS selector of the <iframe> element

    Returns:
        {"status": "ok", "frame": frame_selector, "url": <frame src url>}
    """
    frame = _page().frame_locator(frame_selector)
    return {"status": "ok", "frame": frame_selector}


async def get_frame_text(frame_selector: str, inner_selector: str) -> dict:
    """
    Get text from an element inside an iframe.

    Args:
        frame_selector: CSS selector of the <iframe> element
        inner_selector: Selector of the element inside the iframe

    Returns:
        {"text": <element text>}
    """
    frame = _page().frame_locator(frame_selector)
    text = await frame.locator(inner_selector).inner_text()
    return {"text": text}


async def click_in_frame(frame_selector: str, inner_selector: str, timeout: int = 30000) -> dict:
    """
    Click an element inside an iframe.

    Args:
        frame_selector: CSS selector of the <iframe> element
        inner_selector: Selector of the element inside the iframe
        timeout: Max wait time in milliseconds
    """
    frame = _page().frame_locator(frame_selector)
    await frame.locator(inner_selector).click(timeout=timeout)
    return {"status": "ok", "frame": frame_selector, "clicked": inner_selector}


# ---------------------------------------------------------------------------
# TABS & MULTI-PAGE
# ---------------------------------------------------------------------------

async def new_tab(url: Optional[str] = None) -> dict:
    """
    Open a new browser tab and switch to it.

    Args:
        url: Optional URL to navigate the new tab to

    Returns:
        {"status": "ok", "tab_index": <index of new tab>}
    """
    context = _state["context"]
    page = await context.new_page()
    if url:
        await page.goto(url)
    _state["page"] = page
    _invalidate_index()
    pages = context.pages
    return {"status": "ok", "tab_index": pages.index(page), "url": page.url}


async def switch_tab(index: int) -> dict:
    """
    Switch to an already-open tab by its index.

    Args:
        index: 0-based tab index

    Returns:
        {"status": "ok", "url": <url of the tab>}
    """
    context = _state["context"]
    pages = context.pages
    if index >= len(pages):
        return {"error": f"No tab at index {index}. Total tabs: {len(pages)}"}
    _state["page"] = pages[index]
    _invalidate_index()
    await pages[index].bring_to_front()
    return {"status": "ok", "url": pages[index].url}


async def close_tab(index: Optional[int] = None) -> dict:
    """
    Close a tab. If no index is given, closes the current tab.

    Args:
        index: 0-based tab index (optional — closes current tab if omitted)
    """
    context = _state["context"]
    pages = context.pages
    target = pages[index] if index is not None else _state["page"]
    await target.close()
    _state["page"] = context.pages[-1] if context.pages else None
    return {"status": "ok", "remaining_tabs": len(context.pages)}


async def list_tabs() -> dict:
    """
    List all currently open tabs with their URLs and titles.

    Returns:
        {"tabs": [{"index": int, "url": str, "title": str}, ...]}
    """
    context = _state["context"]
    tabs = []
    for i, p in enumerate(context.pages):
        tabs.append({"index": i, "url": p.url, "title": await p.title()})
    return {"tabs": tabs}


# ---------------------------------------------------------------------------
# COOKIES & STORAGE
# ---------------------------------------------------------------------------

async def get_cookies() -> dict:
    """
    Get all cookies for the current browser context.

    Returns:
        {"cookies": [list of cookie dicts]}
    """
    cookies = await _state["context"].cookies()
    return {"cookies": cookies}


async def set_cookies(cookies: list[dict]) -> dict:
    """
    Add cookies to the current browser context.

    Args:
        cookies: List of cookie dicts. Each dict must have at minimum:
                 {"name": str, "value": str, "url": str}
                 Optional keys: "domain", "path", "expires", "httpOnly", "secure"

    Returns:
        {"status": "ok", "count": <number of cookies added>}
    """
    await _state["context"].add_cookies(cookies)
    return {"status": "ok", "count": len(cookies)}


async def clear_cookies() -> dict:
    """Clear all cookies from the current browser context."""
    await _state["context"].clear_cookies()
    return {"status": "ok"}


async def get_local_storage(key: str) -> dict:
    """
    Read a value from the page's localStorage.

    Args:
        key: localStorage key name

    Returns:
        {"key": key, "value": <stored value or null>}
    """
    value = await _page().evaluate(f"() => localStorage.getItem('{key}')")
    return {"key": key, "value": value}


async def set_local_storage(key: str, value: str) -> dict:
    """
    Set a value in the page's localStorage.

    Args:
        key: localStorage key name
        value: Value to store (string)
    """
    await _page().evaluate("([k, v]) => localStorage.setItem(k, v)", [key, value])
    return {"status": "ok", "key": key, "value": value}


async def save_storage_state(path: str) -> dict:
    """
    Save cookies and localStorage to a JSON file (for reusing login sessions).

    Args:
        path: File path to save state to (e.g. "auth/session.json")

    Returns:
        {"status": "ok", "path": path}
    """
    await _state["context"].storage_state(path=path)
    return {"status": "ok", "path": path}


# ---------------------------------------------------------------------------
# ASSERTIONS (returns pass/fail instead of raising)
# ---------------------------------------------------------------------------

async def assert_visible(selector: str, timeout: int = 5000) -> dict:
    """
    Check whether an element is visible on the page.

    Args:
        selector: Element selector
        timeout: Max wait time in milliseconds

    Returns:
        {"selector": selector, "visible": True/False}
    """
    try:
        await _page().wait_for_selector(selector, state="visible", timeout=timeout)
        return {"selector": selector, "visible": True}
    except Exception:
        return {"selector": selector, "visible": False}


async def assert_text(selector: str, expected_text: str, timeout: int = 5000) -> dict:
    """
    Check whether an element contains expected text.

    Args:
        selector: Element selector
        expected_text: Text expected to be present in the element
        timeout: Max wait time in milliseconds

    Returns:
        {"selector": selector, "match": True/False, "actual": <actual text>}
    """
    try:
        await _page().wait_for_selector(selector, state="visible", timeout=timeout)
        actual = await _page().inner_text(selector)
        match = expected_text in actual
        return {"selector": selector, "match": match, "actual": actual}
    except Exception as e:
        return {"selector": selector, "match": False, "error": str(e)}


async def assert_url_contains(pattern: str) -> dict:
    """
    Check whether the current URL contains a given string.

    Args:
        pattern: Substring to look for in the current URL

    Returns:
        {"match": True/False, "url": <current url>}
    """
    url = _page().url
    return {"match": pattern in url, "url": url}


async def assert_title(expected_title: str) -> dict:
    """
    Check whether the page title matches (or contains) expected text.

    Args:
        expected_title: Expected text in the page title

    Returns:
        {"match": True/False, "actual_title": <current title>}
    """
    title = await _page().title()
    return {"match": expected_title in title, "actual_title": title}


# ---------------------------------------------------------------------------
# TRACING & DEBUGGING
# ---------------------------------------------------------------------------

async def start_tracing(screenshots: bool = True, snapshots: bool = True) -> dict:
    """
    Start recording a Playwright trace (for post-run debugging).

    Args:
        screenshots: Include screenshots in the trace
        snapshots: Include DOM snapshots in the trace
    """
    await _state["context"].tracing.start(screenshots=screenshots, snapshots=snapshots)
    return {"status": "tracing_started"}


async def stop_tracing(path: str = "trace.zip") -> dict:
    """
    Stop recording and save the trace to a file.
    Open with: playwright show-trace <path>

    Args:
        path: File path to save the trace zip (e.g. "traces/run1.zip")

    Returns:
        {"status": "ok", "path": path}
    """
    await _state["context"].tracing.stop(path=path)
    return {"status": "ok", "path": path}


# ---------------------------------------------------------------------------
# UTILITY: run any async tool from sync context
# ---------------------------------------------------------------------------

def run(coro) -> Any:
    """
    Helper to run an async tool function from synchronous code.
    Example:
        run(start_browser())
        run(navigate("https://example.com"))
        run(stop_browser())
    """
    return asyncio.get_event_loop().run_until_complete(coro)
