"""Handshake autonomous apply: Playwright-driven, session-state-based.

Uses the storage_state captured by `discovery/handshake.py` to skip SSO on
every run. Targets Handshake's React SPA application modal directly — no
Claude Code, no MCP overhead. All locators use ARIA roles + accessible
names so they survive Handshake's frequent CSS-class churn.

Pipeline per job:
  1. Navigate to the job URL (saved storage_state authenticates).
  2. Detect "expired" / "apply externally" early-exit signals.
  3. Click an Apply button, wait for the modal `div[role="dialog"]`.
  4. Upload the tailored resume PDF via the file-input intercept.
  5. Scrape <label> elements inside the modal; for each:
        - try the profile-derived known-answer table,
        - fall back to the optional LLM callback for free-text questions.
  6. In dry-run, stop here. Otherwise click Submit and verify a success
     toast appeared (or the modal closed) before marking applied.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from patchright.sync_api import (
    Locator,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

from applypilot import config
from applypilot.database import get_connection, init_db
from applypilot.discovery.handshake import (
    _ensure_patchright_user_data_dir,
    _looks_logged_out,
    _resolve_session_state_path,
)

log = logging.getLogger(__name__)


# ── Locator vocabulary ────────────────────────────────────────────────────
# Accessible-name fragments only. Order matters — first match wins.

APPLY_BUTTON_NAMES: tuple[str, ...] = (
    "Apply",
    "Apply now",
    "Quick Apply",
    "Submit application",
)

SUBMIT_BUTTON_NAMES: tuple[str, ...] = (
    "Submit application",
    "Submit",
    "Send application",
    "Apply",
)

MODAL_DIALOG = 'div[role="dialog"]'

SUCCESS_TOAST_PATTERNS: tuple[str, ...] = (
    "Application Submitted",
    "Application submitted",
    "Successfully applied",
    "Your application was sent",
    "Application sent",
    "Thanks for applying",
)

EXPIRED_PATTERNS: tuple[str, ...] = (
    "No longer accepting applications",
    "This job is no longer",
    "Posting expired",
    "Job posting closed",
)

EXTERNAL_PATTERNS: tuple[str, ...] = (
    "Apply on company website",
    "Apply externally",
    "Apply on employer site",
)

DEFAULT_MODAL_TIMEOUT_MS = 15_000
DEFAULT_NAV_TIMEOUT_MS = 60_000

# Human-pacing defaults — Handshake is tied to a single university account
# so we move at human speed instead of as-fast-as-Playwright-allows.
DEFAULT_MIN_SLEEP_SECONDS = 30.0
DEFAULT_MAX_SLEEP_SECONDS = 90.0

# Abort the whole run after this many consecutive login_issue results — the
# saved storage_state has clearly expired and continuing just burns retries.
MAX_CONSECUTIVE_LOGIN_FAILURES = 2

# Role-PDF bootstrap: substrings (lowercase) used to map a job to one of
# the four pre-compiled per-role resumes. Checked in dict-iteration order,
# so put the most specific role first.
ROLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Quant": (
        "quant", "trader", "trading", "derivat", "market mak",
        "high frequency", " hft", "algorithmic trading",
        "low latency", "options ", "fixed income",
    ),
    "SWE": (
        "software engineer", "swe ", "backend", "front end", "frontend",
        "full stack", "full-stack", "devops", "platform engineer",
        "site reliab", "infrastructure", "mobile engineer",
        "ml engineer", "machine learning engineer", "ios engineer",
        "android engineer", "systems engineer",
    ),
    "Analyst": (
        "data analyst", "data scientist", "data engineer",
        "research", "business analyst", "financial analyst",
        "consulting analyst", "investment analyst",
        "intelligence analyst", "strategy analyst",
    ),
}

# Labels that should NEVER hit the LLM fallback — checkbox-style consent
# fields where a generated paragraph would be wrong (and possibly fill a
# wrong widget anyway).
LLM_FALLBACK_DENYLIST: tuple[str, ...] = (
    "agree", "consent", "terms", "privacy", "policy",
    "verify", "captcha", "robot",
    "i confirm", "acknowledg",
)

# Hard ceiling on LLM-generated answers — Handshake textareas are
# typically capped at 2000 chars; keep some headroom.
LLM_ANSWER_MAX_CHARS = 1500


class ApplyResult:
    """Result tag enum (string constants for DB / log readability)."""

    APPLIED = "applied"
    EXPIRED = "expired"
    EXTERNAL = "external"
    CAPTCHA = "captcha"
    LOGIN_ISSUE = "login_issue"
    NO_APPLY_BUTTON = "no_apply_button"
    UPLOAD_FAILED = "upload_failed"
    SUBMIT_MISSING = "submit_button_missing"
    SUBMIT_TIMEOUT = "submit_timeout"
    DRY_RUN = "dry_run"


# Permanent failures should never be retried.
PERMANENT_REASONS: frozenset[str] = frozenset({
    ApplyResult.EXPIRED,
    ApplyResult.EXTERNAL,
    ApplyResult.LOGIN_ISSUE,
    ApplyResult.CAPTCHA,
})


# ── Profile → known-answers table ─────────────────────────────────────────

KnownAnswerTable = list[tuple[tuple[str, ...], str]]


def build_known_answers(profile: dict) -> KnownAnswerTable:
    """Build a label-substring → answer table from the user's profile.

    Match semantics: case-insensitive substring containment. The first
    needle that appears in a label wins. So "GitHub URL", "Your GitHub",
    and "GitHub profile" all hit the same answer.
    """
    personal = profile.get("personal", {})
    work_auth = profile.get("work_authorization", {})
    comp = profile.get("compensation", {})
    exp = profile.get("experience", {})
    avail = profile.get("availability", {})
    eeo = profile.get("eeo_voluntary", {})

    full_name = personal.get("full_name", "")
    name_parts = full_name.split() if full_name else []
    first_name = personal.get("preferred_name") or (name_parts[0] if name_parts else "")
    last_name = name_parts[-1] if len(name_parts) > 1 else ""

    salary_amount = comp.get("salary_expectation", "")
    salary_currency = comp.get("salary_currency", "USD")
    salary_str = f"{salary_amount} {salary_currency}".strip() if salary_amount else ""

    table: KnownAnswerTable = [
        # Identity — order matters: more specific labels before generic "name"
        (("legal name", "full name"), full_name),
        (("first name", "preferred name", "given name"), first_name),
        (("last name", "surname", "family name"), last_name),
        (("email",), personal.get("email", "")),
        (("phone", "mobile", "cell"), personal.get("phone", "")),
        # Links
        (("github",), personal.get("github_url", "")),
        (("linkedin",), personal.get("linkedin_url", "")),
        (
            ("portfolio", "personal site", "website"),
            personal.get("portfolio_url") or personal.get("website_url", ""),
        ),
        # Address
        (("street", "address line"), personal.get("address", "")),
        (("city",), personal.get("city", "")),
        (("state", "province", "region"), personal.get("province_state", "")),
        (("country",), personal.get("country", "")),
        (("zip", "postal code"), personal.get("postal_code", "")),
        # Work auth
        (
            ("authorized to work", "legally authorized", "work authorization"),
            work_auth.get("legally_authorized_to_work", "Yes"),
        ),
        (
            ("sponsorship", "visa sponsor", "require sponsor"),
            work_auth.get("require_sponsorship", "No"),
        ),
        # Compensation
        (
            (
                "salary expectation",
                "expected salary",
                "desired salary",
                "compensation expectation",
            ),
            salary_str,
        ),
        # Experience
        (("years of experience", "total experience"), exp.get("years_of_experience_total", "")),
        (("education", "highest degree"), exp.get("education_level", "")),
        # Availability
        (
            ("start date", "available to start", "earliest start"),
            avail.get("earliest_start_date", "Immediately"),
        ),
        # EEO
        (("gender",), eeo.get("gender", "Decline to self-identify")),
        (("race", "ethnicity"), eeo.get("race_ethnicity", "Decline to self-identify")),
        (("veteran",), eeo.get("veteran_status", "I am not a protected veteran")),
        (("disability",), eeo.get("disability_status", "I do not wish to answer")),
    ]
    return [(needles, ans) for needles, ans in table if ans]


def lookup_known_answer(label: str, table: KnownAnswerTable) -> str | None:
    """Substring-match a field label against the known-answer table."""
    lo = label.lower().strip()
    for needles, ans in table:
        for needle in needles:
            if needle in lo:
                return ans
    return None


# ── Role-PDF bootstrap ────────────────────────────────────────────────────


def _default_role_resumes_dir() -> Path:
    """Locate the user's per-role PDF directory.

    Walks up from this source file to find `Resumes/Resume/Roles/` sitting
    next to the ApplyPilot checkout. Override via env var
    `APPLYPILOT_ROLE_RESUMES_DIR` for non-standard layouts.
    """
    env = os.environ.get("APPLYPILOT_ROLE_RESUMES_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    # apply/handshake.py → applypilot/ → src/ → ApplyPilot/ → Handshake/
    return Path(__file__).resolve().parents[4] / "Resumes" / "Resume" / "Roles"


def pick_role_for_job(job: dict) -> str:
    """Choose Quant / SWE / Analyst / General by keyword-matching the job.

    Title is checked first across all roles; description is only consulted
    when the title is generic. Without this, a "Backend Engineer" job with
    "low latency" in the description would be miscategorized as Quant
    because of description noise.
    """
    title = (job.get("title") or "").lower()
    description = (
        job.get("full_description") or job.get("description") or ""
    ).lower()

    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in title:
                return role
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in description:
                return role
    return "General"


def bootstrap_resume_path(job: dict, role_resumes_dir: Path) -> Path | None:
    """Return the static per-role Digital PDF path for this job, or None.

    Prefers `*_Digital.pdf`; falls back to any `*.pdf` excluding the
    embedded `github-mark.pdf` asset. None when the role-resumes dir
    doesn't exist or contains no usable PDF.
    """
    if not role_resumes_dir.exists():
        return None
    role = pick_role_for_job(job)
    role_dir = role_resumes_dir / role
    if not role_dir.exists():
        return None
    digital = sorted(role_dir.glob("*_Digital.pdf"))
    if digital:
        return digital[0]
    for pdf in sorted(role_dir.glob("*.pdf")):
        if "github" in pdf.name.lower():
            continue
        return pdf
    return None


# ── LLM Q&A callback ──────────────────────────────────────────────────────


def _should_llm_answer(label: str) -> bool:
    """Skip labels that look like consent checkboxes rather than free-text."""
    lo = label.lower()
    return not any(needle in lo for needle in LLM_FALLBACK_DENYLIST)


def _flatten_skills(skills_boundary: dict) -> str:
    flat: list[str] = []
    for v in skills_boundary.values():
        if isinstance(v, list):
            flat.extend(v)
    return ", ".join(flat)


def build_llm_qa_callback(
    profile: dict, *, max_chars: int = LLM_ANSWER_MAX_CHARS
) -> QaCallback:
    """Build a QaCallback bound to `applypilot.llm` for free-text questions.

    System prompt enforces:
      - PLAIN TEXT only — Handshake textareas don't render markdown or
        auto-link URLs (saved in memory as a platform constraint).
      - Hard char ceiling well under Handshake's typical 2000 cap.
      - No fabrication: stay grounded in the user's resume_facts.

    Raises whatever `get_client()` raises (typically RuntimeError when no
    LLM provider is configured). Callers should treat that as a signal to
    proceed without LLM Q&A rather than aborting the whole run.
    """
    from applypilot.llm import get_client

    client = get_client()

    personal = profile.get("personal", {})
    exp = profile.get("experience", {})
    resume_facts = profile.get("resume_facts", {})
    skills = profile.get("skills_boundary", {}) or {}

    persona_lines = [
        f"Name: {personal.get('full_name', '')}",
        f"Education: {exp.get('education_level', '')}",
        f"Target role: {exp.get('target_role', '')}",
        f"Companies on resume: "
        + (", ".join(resume_facts.get("preserved_companies", [])) or "none"),
        f"Projects on resume: "
        + (", ".join(resume_facts.get("preserved_projects", [])) or "none"),
        f"School on resume: {resume_facts.get('preserved_school', '')}",
        f"Skills: {_flatten_skills(skills) or 'see resume'}",
    ]
    persona_block = "\n".join(persona_lines)

    system_prompt = (
        "You answer free-text questions on a Handshake job application form "
        "on behalf of the applicant below.\n\n"
        "HARD RULES — violating any of these makes the answer unusable:\n"
        "- Output PLAIN TEXT only. No markdown syntax (no **bold**, no "
        "*italics*, no `code`, no bullet symbols, no hyphens used as "
        "bullets, no headers). Handshake textareas display the literal "
        "characters and do not auto-link URLs.\n"
        f"- Stay under {max_chars} characters. Aim for 2-4 sentences on "
        "short prompts, one short paragraph for longer ones.\n"
        "- Ground every concrete claim in the applicant profile below. "
        "Never invent companies, schools, certifications, projects, or "
        "experiences not listed.\n"
        "- First-person voice. Start with the actual answer — no 'Sure!', "
        "no 'I would say', no echoing the question.\n"
        "- If the question requires information not in the profile, write "
        "a brief honest answer that stays generic rather than fabricating "
        "specifics.\n\n"
        "APPLICANT PROFILE\n"
        f"{persona_block}"
    )

    def _callback(question: str, ctx: dict) -> str | None:
        if not _should_llm_answer(question):
            return None
        job = ctx.get("job", {}) if isinstance(ctx, dict) else {}
        description = (
            job.get("full_description") or job.get("description") or ""
        )
        job_block = (
            f"Job title: {job.get('title', '')}\n"
            f"Company / site: {job.get('site', '')}\n"
            f"Location: {job.get('location', '')}\n"
            f"Description excerpt: {description[:1500]}"
        )
        user_prompt = (
            "JOB CONTEXT\n"
            f"{job_block}\n\n"
            "QUESTION\n"
            f"{question.strip()}"
        )
        try:
            answer = client.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
        except Exception:
            log.exception("LLM Q&A failed for label=%r", question[:80])
            return None

        cleaned = (answer or "").strip()
        if not cleaned:
            return None
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(",.;:") + "."
        return cleaned

    return _callback


# ── Modal navigation + field-fill primitives ──────────────────────────────


def _open_apply_modal(page: Page) -> Locator | None:
    """Click an Apply-style button and return the visible modal locator.

    Returns None when no Apply button matched (likely an external apply
    redirect Handshake didn't flag explicitly).
    """
    for name in APPLY_BUTTON_NAMES:
        btn = page.get_by_role("button", name=name)
        try:
            if btn.count() == 0:
                continue
            btn.first.click(timeout=5_000)
        except PlaywrightTimeoutError:
            continue
        except Exception:
            continue
        try:
            page.wait_for_selector(
                MODAL_DIALOG, state="visible", timeout=DEFAULT_MODAL_TIMEOUT_MS
            )
            return page.locator(MODAL_DIALOG).first
        except PlaywrightTimeoutError:
            continue
    return None


def _upload_file_to_modal(
    modal: Locator, label_candidates: tuple[str, ...], path: Path
) -> bool:
    """Attach a file to the first matching upload input inside the modal.

    Tries each label candidate via accessible-name first, then falls back
    to any `input[type=file]` Playwright can see inside the modal.
    """
    for label in label_candidates:
        try:
            field = modal.get_by_label(label)
            if field.count() == 0:
                continue
            field.first.set_input_files(str(path))
            return True
        except Exception:
            continue

    try:
        any_file = modal.locator('input[type="file"]')
        if any_file.count() > 0:
            any_file.first.set_input_files(str(path))
            return True
    except Exception:
        pass

    return False


def _upload_resume(modal: Locator, resume_path: Path) -> bool:
    return _upload_file_to_modal(
        modal,
        ("Resume", "Upload resume", "Attach resume", "Resume file"),
        resume_path,
    )


def _upload_cover_letter(modal: Locator, cover_path: Path | None) -> bool:
    if cover_path is None or not cover_path.exists():
        return False
    return _upload_file_to_modal(
        modal,
        ("Cover letter", "Upload cover letter", "Attach cover letter"),
        cover_path,
    )


def _scrape_form_labels(modal: Locator) -> list[str]:
    """Return visible <label> text content inside the modal."""
    out: list[str] = []
    try:
        labels = modal.locator("label")
        count = labels.count()
    except Exception:
        return out
    for i in range(count):
        try:
            txt = labels.nth(i).inner_text(timeout=2_000).strip()
        except Exception:
            continue
        if txt and txt not in out:
            out.append(txt)
    return out


def _answer_field(modal: Locator, label: str, answer: str) -> bool:
    """Fill a single labeled field. Tries text → combobox → radio in order."""
    if not answer:
        return False

    try:
        field = modal.get_by_label(label)
        if field.count() > 0:
            field.first.fill(answer, timeout=4_000)
            return True
    except Exception:
        pass

    try:
        combo = modal.get_by_role("combobox", name=label)
        if combo.count() > 0:
            combo.first.click(timeout=4_000)
            opt = modal.get_by_role("option", name=answer)
            if opt.count() > 0:
                opt.first.click(timeout=4_000)
                return True
    except Exception:
        pass

    try:
        radio = modal.get_by_role("radio", name=answer)
        if radio.count() > 0:
            radio.first.check(timeout=4_000)
            return True
    except Exception:
        pass

    return False


# ── Post-submit result classification ─────────────────────────────────────


def _page_text_lower(page: Page) -> str:
    try:
        return page.inner_text("body", timeout=5_000).lower()
    except Exception:
        return ""


def _classify_post_submit(page: Page, modal: Locator) -> str:
    """Decide whether the submit click actually went through."""
    body = _page_text_lower(page)
    if any(p.lower() in body for p in SUCCESS_TOAST_PATTERNS):
        return ApplyResult.APPLIED

    try:
        if not modal.is_visible(timeout=1_000):
            return ApplyResult.APPLIED
    except Exception:
        return ApplyResult.APPLIED

    return ApplyResult.SUBMIT_TIMEOUT


# ── Per-job applicator ────────────────────────────────────────────────────

QaCallback = Callable[[str, dict], str | None]


def apply_to_handshake_job(
    page: Page,
    job: dict,
    profile: dict,
    *,
    dry_run: bool = True,
    qa_callback: QaCallback | None = None,
) -> dict:
    """Apply to a single Handshake job inside an already-authenticated page.

    Args:
        page: Playwright page from a context built with storage_state.
        job: DB row dict (must include `url` and `tailored_resume_path`).
        profile: User profile from `config.load_profile()`.
        dry_run: If True, fill the form but do not click Submit.
        qa_callback: Optional fn(question, context) → answer. Called for
            labels the known-answer table doesn't cover.

    Returns:
        Dict with at minimum: status, reason, duration_ms, answered,
        unanswered. Status is one of ApplyResult.* constants; reason is
        a finer-grained tag for the DB.
    """
    start = time.time()

    def _elapsed_ms() -> int:
        return int((time.time() - start) * 1000)

    url = job.get("application_url") or job.get("url")
    resume_path_raw = job.get("tailored_resume_path") or ""
    resume_path = Path(resume_path_raw)
    cover_path_raw = job.get("cover_letter_path")
    cover_path = Path(cover_path_raw) if cover_path_raw else None

    if not url:
        return {"status": "failed", "reason": "missing_url", "duration_ms": _elapsed_ms()}
    if not resume_path.exists():
        return {
            "status": "failed",
            "reason": "tailored_resume_missing",
            "duration_ms": _elapsed_ms(),
        }

    page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_NAV_TIMEOUT_MS)
    if _looks_logged_out(page.url):
        return {
            "status": "failed",
            "reason": ApplyResult.LOGIN_ISSUE,
            "permanent": True,
            "duration_ms": _elapsed_ms(),
        }

    body = _page_text_lower(page)
    if any(p.lower() in body for p in EXPIRED_PATTERNS):
        return {
            "status": "failed",
            "reason": ApplyResult.EXPIRED,
            "permanent": True,
            "duration_ms": _elapsed_ms(),
        }
    if any(p.lower() in body for p in EXTERNAL_PATTERNS):
        return {
            "status": "failed",
            "reason": ApplyResult.EXTERNAL,
            "permanent": True,
            "duration_ms": _elapsed_ms(),
        }

    modal = _open_apply_modal(page)
    if modal is None:
        return {
            "status": "failed",
            "reason": ApplyResult.NO_APPLY_BUTTON,
            "duration_ms": _elapsed_ms(),
        }

    if not _upload_resume(modal, resume_path):
        return {
            "status": "failed",
            "reason": ApplyResult.UPLOAD_FAILED,
            "duration_ms": _elapsed_ms(),
        }
    _upload_cover_letter(modal, cover_path)

    table = build_known_answers(profile)
    answered: list[str] = []
    unanswered: list[str] = []
    for label in _scrape_form_labels(modal):
        ans = lookup_known_answer(label, table)
        if ans is None and qa_callback is not None:
            try:
                ans = qa_callback(label, {"job": job, "profile": profile})
            except Exception:
                log.exception("qa_callback raised for label=%r", label)
                ans = None
        if ans and _answer_field(modal, label, ans):
            answered.append(label)
        else:
            unanswered.append(label)

    if dry_run:
        return {
            "status": ApplyResult.DRY_RUN,
            "reason": "dry_run_complete",
            "duration_ms": _elapsed_ms(),
            "answered": answered,
            "unanswered": unanswered,
        }

    submitted = False
    for name in SUBMIT_BUTTON_NAMES:
        try:
            btn = modal.get_by_role("button", name=name)
            if btn.count() == 0:
                continue
            btn.first.click(timeout=5_000)
            submitted = True
            break
        except Exception:
            continue

    if not submitted:
        return {
            "status": "failed",
            "reason": ApplyResult.SUBMIT_MISSING,
            "duration_ms": _elapsed_ms(),
            "answered": answered,
            "unanswered": unanswered,
        }

    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except PlaywrightTimeoutError:
        pass

    tag = _classify_post_submit(page, modal)
    status = ApplyResult.APPLIED if tag == ApplyResult.APPLIED else "failed"
    return {
        "status": status,
        "reason": tag,
        "duration_ms": _elapsed_ms(),
        "answered": answered,
        "unanswered": unanswered,
    }


# ── Batch orchestrator ────────────────────────────────────────────────────


def _count_applied_today() -> int:
    """Return the number of jobs that were marked applied today (local time).

    Used as the basis for the daily-cap guard. Dry-runs are not counted
    because they don't write apply_status='applied'.
    """
    conn = get_connection()
    row = conn.execute(
        """
        SELECT COUNT(*)
        FROM jobs
        WHERE apply_status = 'applied'
          AND date(applied_at, 'localtime') = date('now', 'localtime')
        """
    ).fetchone()
    return int(row[0]) if row else 0


def _acquire_handshake_jobs(
    min_score: int,
    max_jobs: int,
    *,
    season_keywords: list[str] | None = None,
    include_null_resume_path: bool = False,
) -> list[dict]:
    """Pull Handshake jobs ready for apply, ordered by score desc.

    Args:
        min_score: Minimum fit_score threshold.
        max_jobs: Hard cap on returned rows.
        season_keywords: If set, restrict to jobs whose title or full
            description contains any of these substrings (case-insensitive
            via SQLite LIKE). Use to target e.g. "Fall 2026".
        include_null_resume_path: If True, allow jobs with no
            tailored_resume_path. Required when relying on the role-PDF
            bootstrap to attach a path in-memory at apply time.
    """
    init_db()
    conn = get_connection()

    where: list[str] = [
        "site = 'Handshake'",
        "fit_score >= ?",
        "(apply_status IS NULL OR apply_status = 'failed')",
    ]
    params: list = [min_score]

    if not include_null_resume_path:
        where.append("tailored_resume_path IS NOT NULL")

    if season_keywords:
        clauses = []
        for kw in season_keywords:
            clauses.append("(title LIKE ? OR full_description LIKE ?)")
            params.append(f"%{kw}%")
            params.append(f"%{kw}%")
        where.append("(" + " OR ".join(clauses) + ")")

    params.append(max_jobs)
    sql = f"""
        SELECT url, title, site, application_url, tailored_resume_path,
               fit_score, location, full_description, cover_letter_path
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY fit_score DESC, url
        LIMIT ?
    """
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _mark_job(url: str, result: dict) -> None:
    """Persist apply result to the jobs row."""
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    status = result.get("status", "failed")
    reason = result.get("reason")
    duration_ms = result.get("duration_ms")
    permanent = bool(result.get("permanent")) or (reason in PERMANENT_REASONS)

    if status == ApplyResult.APPLIED:
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'applied', applied_at = ?,
                           apply_error = NULL, agent_id = NULL,
                           apply_duration_ms = ?
            WHERE url = ?
            """,
            (now, duration_ms, url),
        )
    elif status == ApplyResult.DRY_RUN:
        # Leave apply_status alone — dry runs keep the job re-applyable.
        return
    else:
        attempts_expr = "99" if permanent else "COALESCE(apply_attempts, 0) + 1"
        conn.execute(
            f"""
            UPDATE jobs SET apply_status = ?, apply_error = ?,
                           apply_attempts = {attempts_expr}, agent_id = NULL,
                           apply_duration_ms = ?
            WHERE url = ?
            """,
            ("failed", reason or "unknown", duration_ms, url),
        )
    conn.commit()


def run_handshake_apply(
    max_jobs: int = 10,
    min_score: int = 7,
    *,
    dry_run: bool = True,
    headless: bool = False,
    session_state_path: str | None = None,
    qa_callback: QaCallback | None = None,
    season_keywords: list[str] | None = None,
    bootstrap_resumes: bool = True,
    role_resumes_dir: Path | None = None,
    use_llm_qa: bool = True,
    min_sleep_seconds: float = DEFAULT_MIN_SLEEP_SECONDS,
    max_sleep_seconds: float = DEFAULT_MAX_SLEEP_SECONDS,
    max_per_day: int = 30,
) -> dict:
    """Apply to Handshake jobs in sequence using the saved storage_state.

    Reuses the JSON state captured by `applypilot handshake-login` so this
    can run headless without re-doing university SSO. One Playwright
    browser, one context, one page — jobs are processed sequentially.

    Autonomy-relevant knobs:
      bootstrap_resumes: when True (default), jobs lacking a
        `tailored_resume_path` get an in-memory path picked from the
        user's per-role PDF directory based on keyword match against the
        job title/description. DB row is not mutated.
      use_llm_qa: when True (default), builds an LLM-backed callback for
        free-text screener questions. Silently falls back to known-answers
        only if no LLM provider is configured.
      season_keywords: e.g. ["Fall 2026", "Summer 2026"] — restricts the
        job query to postings mentioning any of these in title or body.
      min/max_sleep_seconds: random sleep between submits to look human
        and avoid tripping bot detection on a university-bound account.

    Aborts the run after MAX_CONSECUTIVE_LOGIN_FAILURES consecutive
    login_issue results — the storage_state has expired and continuing
    just chews through retries.
    """
    state_path = _resolve_session_state_path(session_state_path)
    if not state_path.exists():
        raise RuntimeError(
            "No Handshake session state found at "
            f"{state_path}. Run `applypilot handshake-login` first to "
            "complete SSO and save reusable session state."
        )

    profile = config.load_profile()

    # Daily-cap guard. Counts real applies (not dry-runs) for today in
    # local time. Dry-runs always run regardless of cap so the user can
    # still preview later jobs after the cap is reached.
    today_applied = _count_applied_today()
    remaining_today = max(0, max_per_day - today_applied)
    if not dry_run and remaining_today == 0:
        return {
            "processed": 0,
            "applied": 0,
            "failed": 0,
            "dry_runs": 0,
            "bootstrapped": 0,
            "skipped_no_resume": 0,
            "dry_run": dry_run,
            "aborted": True,
            "abort_reason": "daily_cap_reached",
            "today_applied": today_applied,
            "max_per_day": max_per_day,
            "llm_qa": "off",
            "role_resumes_dir": "",
            "session_state": str(state_path),
            "results": [],
        }
    effective_max_jobs = max_jobs if dry_run else min(max_jobs, remaining_today)

    # LLM Q&A callback — silently degrades to known-answers-only if the
    # LLM provider isn't configured (RuntimeError from get_client()) so a
    # missing API key doesn't take down the whole apply run.
    llm_qa_status = "off"
    if qa_callback is None and use_llm_qa:
        try:
            qa_callback = build_llm_qa_callback(profile)
            llm_qa_status = "on"
        except Exception as e:  # noqa: BLE001 -- intentional degradation
            log.warning("LLM Q&A disabled — %s", e)
            llm_qa_status = f"disabled: {e}"

    role_dir = role_resumes_dir or _default_role_resumes_dir()

    raw_jobs = _acquire_handshake_jobs(
        min_score=min_score,
        max_jobs=effective_max_jobs,
        season_keywords=season_keywords,
        include_null_resume_path=bootstrap_resumes,
    )

    # Bootstrap missing resume paths in-memory (DB row stays NULL).
    bootstrapped_count = 0
    skipped_no_resume = 0
    jobs: list[dict] = []
    for raw_job in raw_jobs:
        if raw_job.get("tailored_resume_path"):
            jobs.append(raw_job)
            continue
        if not bootstrap_resumes:
            skipped_no_resume += 1
            continue
        pdf_path = bootstrap_resume_path(raw_job, role_dir)
        if pdf_path is None:
            skipped_no_resume += 1
            continue
        bootstrapped_count += 1
        jobs.append(
            {
                **raw_job,
                "tailored_resume_path": str(pdf_path),
                "_bootstrapped_role": pick_role_for_job(raw_job),
            }
        )

    if not jobs:
        return {
            "processed": 0,
            "applied": 0,
            "failed": 0,
            "dry_runs": 0,
            "bootstrapped": 0,
            "skipped_no_resume": skipped_no_resume,
            "dry_run": dry_run,
            "aborted": False,
            "abort_reason": None,
            "today_applied": today_applied,
            "max_per_day": max_per_day,
            "llm_qa": llm_qa_status,
            "role_resumes_dir": str(role_dir),
            "session_state": str(state_path),
            "results": [],
        }

    applied = 0
    failed = 0
    dry_runs = 0
    per_job: list[dict] = []
    consecutive_login_failures = 0
    aborted = False
    abort_reason: str | None = None

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
            for i, job in enumerate(jobs):
                try:
                    result = apply_to_handshake_job(
                        page,
                        job,
                        profile,
                        dry_run=dry_run,
                        qa_callback=qa_callback,
                    )
                except Exception as e:  # noqa: BLE001 -- isolate per-job failures
                    log.exception("Handshake apply crashed: %s", job.get("url"))
                    result = {
                        "status": "failed",
                        "reason": f"exception:{type(e).__name__}",
                        "duration_ms": 0,
                    }
                _mark_job(job["url"], result)
                per_job.append({"url": job["url"], **result})

                status = result.get("status")
                reason = result.get("reason")
                if status == ApplyResult.APPLIED:
                    applied += 1
                    consecutive_login_failures = 0
                elif status == ApplyResult.DRY_RUN:
                    dry_runs += 1
                    consecutive_login_failures = 0
                else:
                    failed += 1
                    if reason == ApplyResult.LOGIN_ISSUE:
                        consecutive_login_failures += 1
                    else:
                        consecutive_login_failures = 0

                if consecutive_login_failures >= MAX_CONSECUTIVE_LOGIN_FAILURES:
                    log.warning(
                        "Aborting Handshake apply: %d consecutive login_issue "
                        "results. Storage state has expired — re-run "
                        "`applypilot handshake-login`.",
                        consecutive_login_failures,
                    )
                    aborted = True
                    abort_reason = "consecutive_login_failures"
                    break

                # Human-pacing sleep between jobs (skip after the last).
                is_last = i >= len(jobs) - 1
                if not is_last and max_sleep_seconds > 0:
                    lo = max(0.0, min_sleep_seconds)
                    hi = max(lo, max_sleep_seconds)
                    pause = random.uniform(lo, hi)
                    log.info(
                        "Sleeping %.1fs before next job (%d/%d done)",
                        pause,
                        i + 1,
                        len(jobs),
                    )
                    time.sleep(pause)
        finally:
            context.close()

    return {
        "processed": len(per_job),
        "applied": applied,
        "failed": failed,
        "dry_runs": dry_runs,
        "bootstrapped": bootstrapped_count,
        "skipped_no_resume": skipped_no_resume,
        "dry_run": dry_run,
        "aborted": aborted,
        "abort_reason": abort_reason,
        "today_applied": today_applied + applied,
        "max_per_day": max_per_day,
        "llm_qa": llm_qa_status,
        "role_resumes_dir": str(role_dir),
        "session_state": str(state_path),
        "results": per_job,
    }
