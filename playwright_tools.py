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
}


def _page() -> Page:
    if _state["page"] is None:
        raise RuntimeError("No active browser session. Call start_browser() first.")
    return _state["page"]


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
    _state.update({"playwright": None, "browser": None, "context": None, "page": None})
    return {"status": "closed"}


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

    page_html = await get_page_html()
    return {"status": "ok", "url": page.url, "title": await page.title(), "html": page_html["html"]}


async def get_page_snapshot() -> dict:
    """
    Return a structured snapshot of every visible interactive element on the page.
    Each element includes its selector, position (x/y/w/h), DOM order, and whether
    it is currently in the visible viewport — so the agent knows WHERE things are,
    not just what exists.

    Returns:
        {
          "url": ..., "title": ..., "viewport": {"width": ..., "height": ...},
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
        const tags = ['input', 'button', 'a', 'select', 'textarea'];
        // Use document order (TreeWalker keeps DOM order across all tag types)
        const all = Array.from(document.querySelectorAll(tags.join(',')));
        all.forEach(el => {
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
            else if (el.id)          selector = `${tag}#${el.id}`;
            else if (el.name)        selector = `${tag}[name="${attrVal(el.name)}"]`;
            else if (el.placeholder) selector = `${tag}[placeholder="${attrVal(el.placeholder)}"]`;
            else if (el.type && el.type !== 'submit' && el.type !== 'button')
                                     selector = `${tag}[type='${el.type}']`;
            else if (tag === 'a' && info.href)
                                     selector = `a[href="${attrVal(info.href)}"]`;

            // Disambiguate if selector still matches multiple elements
            if (document.querySelectorAll(selector).length > 1) {
                const idx = Array.from(document.querySelectorAll(selector)).indexOf(el);
                if (idx > 0) selector = `${selector} >> nth=${idx}`;
            }
            info.selector = selector;

            results.push(info);
        });
        return results.slice(0, 80);
    }
    """, [vp["width"], vp["height"]])
    return {
        "url":      page.url,
        "title":    await page.title(),
        "viewport": vp,
        "elements": elements,
    }

async def go_back() -> dict:
    """Navigate to the previous page in browser history."""
    page = _page()
    await page.go_back()
    return {"status": "ok", "url": page.url}


async def get_page_html(max_chars: int = 12000, container_selector: str = "body") -> dict:
    """
    Return semantically cleaned HTML of the current page (or a subtree).
    Strips navigation chrome, comments, and noise attributes so the budget
    is spent on the content region the agent actually needs.

    Args:
        max_chars: Truncate to this many characters (default 12000)
        container_selector: Restrict to a subtree, e.g. 'ytd-search' for search
                            results, 'ytd-video-renderer' for the first result.
                            Default 'body' returns the whole page.

    Returns:
        {"url": ..., "title": ..., "html": <clean HTML>}
    """
    page = _page()
    html = await page.evaluate("""
    ([root, maxLen]) => {
        const src = document.querySelector(root) || document.body;
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
                if (!KEEP_ATTRS.has(attr.name)) el.removeAttribute(attr.name);
            });
        });

        // 5. Collapse whitespace
        const raw = clone.innerHTML
            .replace(/\\n\\s*\\n/g, '\\n')
            .replace(/[ \\t]+/g, ' ')
            .trim();
        return raw;
    }
    """, [container_selector, max_chars])
    return {
        "url":   page.url,
        "title": await page.title(),
        "html":  html[:max_chars],
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
        {"status": "ok", "selector": selector}
        — or if navigation occurred —
        {"status": "ok", "selector": selector, "navigated_to": <new url>,
         "title": <new title>, "elements": [...]}
    """
    page = _page()
    url_before = page.url
    await page.click(selector, button=button, click_count=click_count, timeout=timeout)

    # Wait briefly for DOM changes (navigation, dropdown open, SPA route change)
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass

    url_after = page.url
    page_html = await get_page_html()
    snap = await get_page_snapshot()
    return {
        "status":    "ok",
        "selector":  selector,
        "url":       url_after,
        "title":     await page.title(),
        "navigated": url_after != url_before,
        "html":      page_html["html"],
        "elements":  snap["elements"],
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

async def get_text_blocks(container_selector: str = "body", limit: int = 60) -> dict:
    """
    Return visible text-bearing leaf elements (spans, divs, headings, yt-formatted-string)
    with their text and a usable selector. Use this when you need to READ data such as
    view counts, dates, prices, labels — content that is not a clickable element and
    therefore won't appear in get_page_snapshot().

    Args:
        container_selector: Restrict to a subtree, e.g. 'ytd-video-renderer' for the
                            first search result, or 'body' for the whole page.
        limit: Max blocks to return (default 60)

    Returns:
        {"container": ..., "blocks": [{"text": ..., "selector": ..., "x": ..., "y": ...}]}
    """
    page = _page()
    blocks = await page.evaluate("""
    ([root, lim]) => {
        const host = document.querySelector(root) || document.body;
        const out  = [];
        // Include a, button, li, td, th so nav items and table cells appear
        const tags = 'span,div,p,li,td,th,h1,h2,h3,h4,h5,a,button,label,yt-formatted-string,ytd-video-meta-block';
        host.querySelectorAll(tags).forEach(el => {
            if (el.children.length > 0) return;          // leaf nodes only
            const t = (el.innerText || '').trim();
            if (!t || t.length > 120) return;
            const r = el.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return;
            const s = window.getComputedStyle(el);
            if (s.display === 'none' || s.visibility === 'hidden') return;

            let sel = el.tagName.toLowerCase();
            if (el.id) sel = `#${el.id}`;
            const all = document.querySelectorAll(sel);
            if (all.length > 1) {
                const idx = Array.from(all).indexOf(el);
                sel = `${sel} >> nth=${idx}`;
            }
            const info = { text: t, selector: sel, x: Math.round(r.left), y: Math.round(r.top) };
            // Include href for links so agent can navigate directly
            if (el.tagName === 'A' && el.getAttribute('href')) info.href = el.getAttribute('href');
            out.push(info);
        });
        return out.slice(0, lim);
    }
    """, [container_selector, limit])
    return {"container": container_selector, "blocks": blocks}


async def get_text(selector: str, timeout: int = 30000) -> dict:
    """
    Get the visible inner text of an element.

    Args:
        selector: Element selector
        timeout: Max wait time in milliseconds

    Returns:
        {"text": <inner text>}
    """
    text = await _page().inner_text(selector, timeout=timeout)
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
