"""Handshake discovery via authenticated browser session.

This module syncs jobs from a Handshake search results URL that already
contains user-specific filters/ranking. It launches a persistent browser
context so the user's existing logged-in session can be reused.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from patchright.sync_api import TimeoutError as PlaywrightTimeoutError
from patchright.sync_api import sync_playwright

from applypilot import config
from applypilot.database import get_connection, init_db

# Dedicated user_data_dir for patchright — separate from the user's
# real Chrome profile so the SSO browser can run alongside their daily
# Chrome without singleton-lock collisions. Persistent context with
# channel="chrome" is what patchright requires for full Cloudflare
# stealth — vanilla launch() + storage_state can still be fingerprinted.
PATCHRIGHT_USER_DATA_DIR = config.APP_DIR / "patchright-profile"


def _ensure_patchright_user_data_dir() -> Path:
    PATCHRIGHT_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return PATCHRIGHT_USER_DATA_DIR

log = logging.getLogger(__name__)


def _resolve_session_state_path(session_state_path: str | None = None) -> Path:
    """Resolve where the reusable Handshake auth session state is stored."""
    if session_state_path:
        return Path(session_state_path).expanduser()
    env_path = os.environ.get("APPLYPILOT_HANDSHAKE_SESSION_STATE", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    return config.APP_DIR / "handshake_state.json"


def _normalize_job_url(url: str) -> str:
    """Normalize Handshake job URLs for deduplication."""
    base = url.split("#", 1)[0]
    # Keep query params if present; they are sometimes needed for tenancy routing.
    if base.endswith("/"):
        base = base[:-1]
    return base


def _extract_job_links(page) -> list[str]:
    """Extract Handshake job links from the current search/results page.

    Matches three URL shapes Handshake has used:
      /stu/jobs/<id>      — legacy student app
      /jobs/<id>          — modern unified app, direct job URL
      /job-search/<id>    — modern unified app, in-context (a results
                            page with one card focused; the <id> in the
                            path identifies the focused job)
    All matches are normalised to /jobs/<id> so the detail extractor
    lands on a clean job page rather than a results layout, and so
    dedup works across patterns.
    """
    urls = page.evaluate(
        """
        () => {
          const anchors = Array.from(document.querySelectorAll('a[href]'));
          const links = [];
          for (const a of anchors) {
            const href = a.getAttribute('href') || '';
            const m = href.match(/\\/(?:stu\\/jobs|jobs|job-search)\\/(\\d+)/);
            if (!m) continue;
            const jobId = m[1];
            const abs = new URL('/jobs/' + jobId, window.location.origin).toString();
            links.push(abs);
          }
          return Array.from(new Set(links));
        }
        """
    )
    return [str(u) for u in urls]


def _extract_job_detail(page) -> dict:
    """Extract title, location, and description text from a Handshake job page."""
    data = page.evaluate(
        """
        () => {
          const pick = (selectors) => {
            for (const s of selectors) {
              const el = document.querySelector(s);
              if (el && el.innerText && el.innerText.trim().length > 0) {
                return el.innerText.trim();
              }
            }
            return '';
          };

          const title = pick(['h1', '[data-testid*="job-title"]', 'main h1']);

          let description = '';
          const descCandidates = [
            '[data-testid*="job-description"]',
            'section[aria-label="Job description"]',
            '#job-description',
            'main'
          ];
          for (const s of descCandidates) {
            const el = document.querySelector(s);
            if (!el || !el.innerText) continue;
            const txt = el.innerText.trim();
            if (txt.length > description.length) description = txt;
          }

          // Heuristic location extraction from text blocks.
          const pageText = document.body ? document.body.innerText : '';
          let location = '';
          const patterns = [
                        /based in\\s+([^\\n]+)/i,
                        /Location\\s*\\n\\s*([^\\n]+)/i,
                        /Onsite,\\s*based in\\s+([^\\n]+)/i,
                        /Hybrid,\\s*based in\\s+([^\\n]+)/i,
          ];
          for (const p of patterns) {
            const m = pageText.match(p);
            if (m && m[1]) {
              location = m[1].trim();
              break;
            }
          }

          return { title, description, location };
        }
        """
    )
    return {
        "title": (data.get("title") or "").strip(),
        "description": (data.get("description") or "").strip(),
        "location": (data.get("location") or "").strip(),
    }


def _looks_logged_out(url: str) -> bool:
    """Heuristic to detect auth redirects when Handshake session is missing."""
    u = (url or "").lower()
    return any(token in u for token in ("/login", "saml", "oauth", "signin", "auth"))


# Tokens that indicate the browser landed on a Handshake content page (not
# the SSO/login flow). Used as a fallback signal when the modern unified
# app renders job cards without `<a href>` so `_extract_job_links` returns
# zero anchors even though the user is fully logged in.
_CONTENT_URL_TOKENS = ("/job-search", "/jobs/", "/postings", "/stu/")


def _looks_on_content_page(url: str) -> bool:
    u = (url or "").lower()
    if _looks_logged_out(u):
        return False
    return any(token in u for token in _CONTENT_URL_TOKENS)


# Minimum body text length to trust a "we landed on the real content page"
# signal. Real Handshake search pages have many KB of text; Cloudflare's
# "Just a moment..." has ~270 chars; the login page has ~90 chars. 1500
# is well above the gotcha-page sizes and well below any authenticated
# Handshake view.
_MIN_AUTHED_BODY_LEN = 1500


def _looks_like_cf_challenge(page) -> bool:
    """Cloudflare's interactive challenge keeps the original URL but
    sets title to 'Just a moment...' or similar."""
    try:
        title = (page.title() or "").lower()
    except Exception:
        return False
    return title.startswith("just a moment") or "checking your browser" in title


def _page_body_len(page) -> int:
    try:
        return len(page.inner_text("body", timeout=2000) or "")
    except Exception:
        return 0


def capture_handshake_session(
    search_url: str,
    profile_dir: str = "Default",
    max_wait_seconds: int = 300,
    session_state_path: str | None = None,
) -> dict:
    """Open a browser for interactive login and persist Handshake auth state.

    This is intended for SSO flows (e.g., university IdP). The saved state can be
    reused by `run_handshake_sync` in headless mode.
    """
    state_path = _resolve_session_state_path(session_state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    user_data_dir = _ensure_patchright_user_data_dir()

    with sync_playwright() as p:
        context = None
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=str(user_data_dir),
                channel="chrome",
                headless=False,
                no_viewport=True,
            )

            page = context.new_page()
            page.goto(search_url, wait_until="domcontentloaded", timeout=90000)

            deadline = time.time() + max(30, max_wait_seconds)
            authenticated = False
            discovered = 0
            final_url = page.url

            while time.time() < deadline:
                final_url = page.url
                # Login redirects and SSO interstitial URLs disqualify
                # immediately so we don't wait for them to "ripen".
                if _looks_logged_out(final_url):
                    time.sleep(1.5)
                    continue
                # Cloudflare's "Just a moment..." keeps the original URL
                # while challenging — refuse the success signal until the
                # title clears.
                if _looks_like_cf_challenge(page):
                    time.sleep(1.5)
                    continue
                # `wait_until="domcontentloaded"` can fire BEFORE client-side
                # redirects finish, so the URL alone is unreliable right
                # after navigation. Require substantial body content as
                # confirmation that an actual page rendered.
                if _page_body_len(page) < _MIN_AUTHED_BODY_LEN:
                    time.sleep(1.5)
                    continue
                links = _extract_job_links(page)
                if links:
                    authenticated = True
                    discovered = len(links)
                    break
                # Modern Handshake SPA renders job cards as <button>/<div>
                # with onClick instead of <a href>, so the anchor sweep
                # finds zero links even when logged in. With the body-len
                # and CF-challenge guards above, URL-on-content is now
                # safe to trust as a fallback authenticated signal.
                if _looks_on_content_page(final_url):
                    authenticated = True
                    discovered = 0
                    break
                time.sleep(1.5)

            context.storage_state(path=str(state_path))

            return {
                "authenticated": authenticated,
                "discovered": discovered,
                "final_url": final_url,
                "session_state": str(state_path),
                "user_data_dir": str(user_data_dir),
            }
        finally:
            # Save storage_state in finally so Ctrl+C or unexpected exit after
            # the user has completed SSO doesn't lose the cookies. Idempotent
            # — the success path also saves before returning. The persistent
            # context's user_data_dir is the primary cookie store now;
            # storage_state.json is a portable backup.
            if context is not None:
                try:
                    context.storage_state(path=str(state_path))
                except Exception:
                    log.debug("Final storage_state save failed", exc_info=True)
                context.close()


def run_handshake_sync(
    search_url: str,
    max_jobs: int = 50,
    scroll_rounds: int = 8,
    headless: bool = False,
    profile_dir: str = "Default",
    session_state_path: str | None = None,
) -> dict:
    """Sync jobs from a filtered Handshake search URL.

    Args:
        search_url: Full Handshake search URL from the logged-in session.
        max_jobs: Max number of job URLs to ingest.
        scroll_rounds: Number of scroll cycles to load additional cards.
        headless: Whether to run browser headless.
        profile_dir: Chrome profile directory name (e.g., Default, Profile 1).

    Returns:
        Stats dict with counts.
    """
    init_db()
    conn = get_connection()

    state_path = _resolve_session_state_path(session_state_path)
    has_saved_session = state_path.exists()
    user_data_dir = _ensure_patchright_user_data_dir()
    now = datetime.now(timezone.utc).isoformat()

    discovered: list[str] = []
    new_rows = 0
    updated_rows = 0
    duplicates = 0
    errors = 0

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            channel="chrome",
            headless=headless,
            no_viewport=True,
        )

        try:
            page = context.new_page()
            page.goto(search_url, wait_until="domcontentloaded", timeout=90000)
            time.sleep(2)

            if _looks_logged_out(page.url) and not has_saved_session:
                raise RuntimeError(
                    "Handshake requires login in this browser context. "
                    "Run `applypilot handshake-login` once to complete SSO and save session state."
                )

            # Collect links while scrolling the search results.
            for _ in range(max(1, scroll_rounds)):
                links = _extract_job_links(page)
                for link in links:
                    if link not in discovered:
                        discovered.append(link)
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                time.sleep(1.2)
                if len(discovered) >= max_jobs:
                    break

            targets = discovered[:max_jobs]
            log.info("Handshake sync: discovered %d links (max=%d)", len(targets), max_jobs)

            for idx, url in enumerate(targets, start=1):
                norm_url = _normalize_job_url(url)
                try:
                    detail_page = context.new_page()
                    detail_page.goto(norm_url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(1.0)
                    detail = _extract_job_detail(detail_page)
                    detail_page.close()
                except PlaywrightTimeoutError:
                    errors += 1
                    continue
                except Exception:
                    errors += 1
                    continue

                title = detail.get("title") or "(pending enrichment)"
                description = detail.get("description") or None
                location = detail.get("location") or None
                full_description = description if description and len(description) >= 200 else None
                detail_scraped_at = now if full_description else None

                try:
                    conn.execute(
                        """
                        INSERT INTO jobs (
                            url, title, salary, description, location, site, strategy, discovered_at,
                            full_description, application_url, detail_scraped_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            norm_url,
                            title,
                            None,
                            description,
                            location,
                            "Handshake",
                            "handshake_sync",
                            now,
                            full_description,
                            norm_url,
                            detail_scraped_at,
                        ),
                    )
                    new_rows += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
                    # Update existing row with fresher details if missing.
                    cur = conn.execute(
                        """
                        UPDATE jobs
                        SET title = COALESCE(NULLIF(title, ''), ?),
                            description = COALESCE(description, ?),
                            location = COALESCE(location, ?),
                            full_description = COALESCE(full_description, ?),
                            application_url = COALESCE(application_url, ?),
                            detail_scraped_at = COALESCE(detail_scraped_at, ?),
                            site = COALESCE(site, 'Handshake'),
                            strategy = COALESCE(strategy, 'handshake_sync')
                        WHERE url = ?
                        """,
                        (
                            title,
                            description,
                            location,
                            full_description,
                            norm_url,
                            detail_scraped_at,
                            norm_url,
                        ),
                    )
                    if cur.rowcount > 0:
                        updated_rows += 1

                if idx % 10 == 0:
                    conn.commit()

            conn.commit()
        finally:
            if context is not None:
                context.close()

    return {
        "discovered": len(discovered),
        "processed": min(len(discovered), max_jobs),
        "new": new_rows,
        "updated": updated_rows,
        "duplicates": duplicates,
        "errors": errors,
        "used_saved_session": has_saved_session,
        "session_state": str(state_path),
        "user_data_dir": str(user_data_dir),
    }
