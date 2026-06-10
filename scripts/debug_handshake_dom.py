"""Diagnostic: dump DOM signals from a Handshake results page.

Uses the saved storage_state from `applypilot handshake-login` to load
the page in an authenticated state, scrolls a few times to trigger lazy
rendering, then prints what the extractor would see. Also dumps a
screenshot so we can confirm what's actually rendering (Cloudflare
challenge? real Handshake page? something else?).

Usage:
    .venv/bin/python scripts/debug_handshake_dom.py "<search-url>" [--headless]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from patchright.sync_api import sync_playwright

from applypilot import config
from applypilot.discovery.handshake import (
    _ensure_patchright_user_data_dir,
    _resolve_session_state_path,
)


JS = r"""
() => {
  const out = {
    final_url: window.location.href,
    title: document.title,
    body_text_len: (document.body && document.body.innerText || '').length,
    total_anchors: document.querySelectorAll('a[href]').length,
    job_pattern_anchor_count: 0,
    job_pattern_anchor_sample: [],
    href_sample: [],
    data_job_id_count: document.querySelectorAll('[data-job-id]').length,
    data_job_attr_samples: [],
    article_count: document.querySelectorAll('article').length,
    role_listitem_count: document.querySelectorAll('[role="listitem"]').length,
    role_button_count: document.querySelectorAll('[role="button"]').length,
    role_link_count: document.querySelectorAll('[role="link"]').length,
    unique_data_attrs: [],
    sample_card_outerHTML: '',
    job_ids_from_text: [],
  };

  // Anchor analysis
  const all_a = document.querySelectorAll('a[href]');
  const job_anchors = [];
  for (const a of all_a) {
    if (/\/(stu\/jobs|jobs|job-search)\/\d+/.test(a.href)) {
      job_anchors.push(a.href);
    }
  }
  out.job_pattern_anchor_count = job_anchors.length;
  out.job_pattern_anchor_sample = job_anchors.slice(0, 5);
  for (let i = 0; i < Math.min(all_a.length, 20); i++) {
    out.href_sample.push(all_a[i].getAttribute('href'));
  }

  // data-* attribute exploration
  const seen = new Set();
  const data_job_attrs = [];
  document.querySelectorAll('*').forEach(el => {
    for (const attr of el.attributes) {
      if (!attr.name.startsWith('data-')) continue;
      if (!seen.has(attr.name)) {
        seen.add(attr.name);
        out.unique_data_attrs.push({ name: attr.name, sample_value: (attr.value || '').substring(0, 60) });
      }
      if (attr.name.includes('job') || /^\d{6,9}$/.test(attr.value)) {
        if (data_job_attrs.length < 10) {
          data_job_attrs.push({ tag: el.tagName.toLowerCase(), name: attr.name, value: (attr.value || '').substring(0, 60) });
        }
      }
    }
  });
  out.data_job_attr_samples = data_job_attrs;

  // Card sample
  const card = document.querySelector('article')
    || document.querySelector('[role="listitem"]')
    || document.querySelector('[data-hook="job-result-card"]')
    || document.querySelector('[data-test*="job"]');
  if (card) {
    out.sample_card_outerHTML = card.outerHTML.substring(0, 3000);
  }

  // Job IDs that appear free-floating in text (e.g., "Job #11072419")
  const body_text = (document.body && document.body.innerText || '');
  const id_matches = body_text.match(/\b\d{7,9}\b/g) || [];
  out.job_ids_from_text = Array.from(new Set(id_matches)).slice(0, 15);

  return out;
}
"""


def main() -> None:
    args = [a for a in sys.argv[1:] if a]
    headless = "--headless" in args
    args = [a for a in args if not a.startswith("--")]
    if not args:
        print("Usage: python debug_handshake_dom.py <search-url> [--headless]")
        sys.exit(1)
    url = args[0]

    state_path = _resolve_session_state_path(None)
    if not state_path.exists():
        print(f"No session state at {state_path}. Run handshake-login first.")
        sys.exit(1)

    screenshot_path = Path("/tmp/handshake_debug.png")
    user_data_dir = _ensure_patchright_user_data_dir()
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            channel="chrome",
            headless=headless,
            no_viewport=True,
        )
        try:
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            try:
                page.wait_for_load_state("networkidle", timeout=20_000)
            except Exception:
                pass
            # Scroll a few times to trigger lazy-render
            for _ in range(4):
                page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                time.sleep(1.2)
            data = page.evaluate(JS)
            try:
                page.screenshot(path=str(screenshot_path), full_page=False)
                data["_screenshot"] = str(screenshot_path)
            except Exception as e:
                data["_screenshot_error"] = str(e)
        finally:
            context.close()

    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
