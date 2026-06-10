"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    validation: str = typer.Option(
        "normal",
        "--validation",
        help=(
            "Validation strictness for tailor/cover stages. "
            "strict: banned words = errors, judge must pass. "
            "normal: banned words = warnings only (default, recommended for Gemini free tier). "
            "lenient: banned words ignored, LLM judge skipped (fastest, fewest API calls)."
        ),
    ),
) -> None:
    """Run pipeline stages: discover, enrich, score, tailor, cover, pdf."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["all"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    # Validate the --validation flag value
    valid_modes = ("strict", "normal", "lenient")
    if validation not in valid_modes:
        console.print(
            f"[red]Invalid --validation value:[/red] '{validation}'. "
            f"Choose from: {', '.join(valid_modes)}"
        )
        raise typer.Exit(code=1)

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
        validation_mode=validation,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
    )


@app.command()
def ingest(
    url: Optional[list[str]] = typer.Option(
        None,
        "--url",
        help="Job URL to ingest (repeatable). Use Handshake job links for Handshake-first flow.",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        help="Path to a text file containing one URL per line.",
    ),
    site: str = typer.Option(
        "Handshake",
        "--site",
        help="Source label stored in DB (default: Handshake).",
    ),
) -> None:
    """Ingest job URLs directly into the database (Handshake-first workflow)."""
    _bootstrap()

    from applypilot.database import get_connection

    urls: list[str] = []
    if url:
        urls.extend(url)

    if file:
        p = Path(file).expanduser()
        if not p.exists():
            console.print(f"[red]File not found:[/red] {p}")
            raise typer.Exit(code=1)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)

    # Normalize and deduplicate preserving order.
    cleaned: list[str] = []
    seen: set[str] = set()
    for u in urls:
        v = u.strip()
        if not v:
            continue
        if not (v.startswith("http://") or v.startswith("https://")):
            continue
        key = v.split("#", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(v)

    if not cleaned:
        console.print("[yellow]No valid URLs provided. Use --url or --file.[/yellow]")
        raise typer.Exit(code=1)

    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    new = 0
    dupes = 0

    for u in cleaned:
        try:
            conn.execute(
                "INSERT INTO jobs (url, title, site, strategy, discovered_at) VALUES (?, ?, ?, ?, ?)",
                (u, "(pending enrichment)", site, "manual_url", now),
            )
            new += 1
        except Exception:
            dupes += 1

    conn.commit()

    console.print("\n[bold]Ingest complete[/bold]")
    console.print(f"  Site label: {site}")
    console.print(f"  New URLs:   {new}")
    console.print(f"  Duplicates: {dupes}")
    console.print("\nNext: [bold]applypilot run enrich score tailor cover[/bold]")


@app.command("focus-site")
def focus_site(
    site: str = typer.Option("Handshake", "--site", help="Only this site remains eligible for scoring/tailoring."),
) -> None:
    """Deprioritize all other sites by assigning low fit scores.

    This keeps historical rows in the DB but prevents non-target sources from
    being picked up by `run score`/`run tailor` in normal workflows.
    """
    _bootstrap()
    from applypilot.database import get_connection

    conn = get_connection()
    # Mark non-target rows as already scored low-fit so they are skipped.
    cur = conn.execute(
        """
        UPDATE jobs
        SET fit_score = 1,
            score_reasoning = COALESCE(score_reasoning, 'Deprioritized by focus-site filter'),
            scored_at = COALESCE(scored_at, datetime('now'))
        WHERE COALESCE(site, '') != ?
          AND (fit_score IS NULL OR fit_score > 1)
        """,
        (site,),
    )
    conn.commit()

    console.print("\n[bold]Focus filter applied[/bold]")
    console.print(f"  Target site:   {site}")
    console.print(f"  Rows adjusted: {cur.rowcount}")
    console.print("\nNext: [bold]applypilot run enrich score tailor cover[/bold]")


@app.command("handshake-sync")
def handshake_sync(
    search_url: str = typer.Option(
        "",
        "--search-url",
        help="Filtered Handshake search URL from your logged-in account.",
    ),
    max_jobs: int = typer.Option(50, "--max-jobs", help="Maximum number of jobs to sync."),
    scroll_rounds: int = typer.Option(8, "--scroll-rounds", help="Scroll cycles to load more job cards."),
    headless: bool = typer.Option(False, "--headless", help="Run browser headless."),
    profile_dir: str = typer.Option("Default", "--profile-dir", help="Chrome profile directory name."),
    session_state: str = typer.Option(
        "",
        "--session-state",
        help="Path to Playwright storage state JSON (for SSO). Defaults to ~/.applypilot/handshake_state.json.",
    ),
) -> None:
    """Sync jobs directly from your filtered Handshake search page.

    This uses your existing logged-in browser session and imports Handshake job
    URLs/details into the ApplyPilot DB so you do not need manual URL collection.
    """
    _bootstrap()

    import os
    from applypilot.discovery.handshake import run_handshake_sync

    target_url = search_url.strip() or os.environ.get("APPLYPILOT_HANDSHAKE_SEARCH_URL", "").strip()
    if not target_url:
        console.print(
            "[red]Missing search URL.[/red]\n"
            "Provide --search-url or set APPLYPILOT_HANDSHAKE_SEARCH_URL in ~/.applypilot/.env"
        )
        raise typer.Exit(code=1)

    if "joinhandshake.com" not in target_url:
        console.print("[yellow]Warning:[/yellow] URL does not look like Handshake. Continuing anyway.")

    console.print("\n[bold blue]Handshake Sync[/bold blue]")
    console.print(f"  URL:          {target_url[:120]}")
    console.print(f"  Max jobs:     {max_jobs}")
    console.print(f"  Scroll rounds:{scroll_rounds}")
    console.print(f"  Headless:     {headless}")
    console.print(f"  Profile dir:  {profile_dir}")
    if session_state.strip():
        console.print(f"  Session file: {session_state.strip()}")

    try:
        stats = run_handshake_sync(
            search_url=target_url,
            max_jobs=max_jobs,
            scroll_rounds=scroll_rounds,
            headless=headless,
            profile_dir=profile_dir,
            session_state_path=session_state.strip() or None,
        )
    except Exception as e:
        console.print(f"\n[red]Handshake sync failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold]Sync complete[/bold]")
    console.print(f"  Links discovered: {stats.get('discovered', 0)}")
    console.print(f"  Jobs processed:   {stats.get('processed', 0)}")
    console.print(f"  New rows:         {stats.get('new', 0)}")
    console.print(f"  Updated rows:     {stats.get('updated', 0)}")
    console.print(f"  Duplicates:       {stats.get('duplicates', 0)}")
    console.print(f"  Errors:           {stats.get('errors', 0)}")
    console.print(f"  Saved session:    {stats.get('session_state', '')}")
    console.print(f"  Used session:     {stats.get('used_saved_session', False)}")

    if stats.get("discovered", 0) == 0:
        console.print("\n[yellow]No Handshake jobs were discovered.[/yellow]")
        console.print("If you use SSO, run [bold]applypilot handshake-login[/bold] once, then retry sync.")

    console.print("\nNext:")
    console.print("  1) applypilot focus-site --site Handshake")
    console.print("  2) applypilot run score tailor cover")
    console.print("  3) applypilot apply --dry-run")


@app.command("handshake-login")
def handshake_login(
    search_url: str = typer.Option(
        "",
        "--search-url",
        help="Filtered Handshake search URL from your logged-in account.",
    ),
    profile_dir: str = typer.Option("Default", "--profile-dir", help="Chrome profile directory name."),
    max_wait: int = typer.Option(300, "--max-wait", help="Seconds to wait for you to complete SSO login."),
    session_state: str = typer.Option(
        "",
        "--session-state",
        help="Where to save session state JSON. Defaults to ~/.applypilot/handshake_state.json.",
    ),
) -> None:
    """Open a browser, complete SSO login, and save reusable Handshake session state."""
    _bootstrap()

    import os
    from applypilot.discovery.handshake import capture_handshake_session

    target_url = search_url.strip() or os.environ.get("APPLYPILOT_HANDSHAKE_SEARCH_URL", "").strip()
    if not target_url:
        console.print(
            "[red]Missing search URL.[/red]\n"
            "Provide --search-url or set APPLYPILOT_HANDSHAKE_SEARCH_URL in ~/.applypilot/.env"
        )
        raise typer.Exit(code=1)

    console.print("\n[bold blue]Handshake Login (SSO)[/bold blue]")
    console.print(f"  URL:         {target_url[:120]}")
    console.print(f"  Profile dir: {profile_dir}")
    console.print(f"  Wait time:   {max_wait}s")
    if session_state.strip():
        console.print(f"  Session file:{session_state.strip()}")

    console.print("\nComplete Wesleyan SSO in the opened browser window. This command will save your session automatically.")

    try:
        result = capture_handshake_session(
            search_url=target_url,
            profile_dir=profile_dir,
            max_wait_seconds=max_wait,
            session_state_path=session_state.strip() or None,
        )
    except Exception as e:
        console.print(f"\n[red]Handshake login failed:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold]Handshake session saved[/bold]")
    console.print(f"  Session file: {result.get('session_state', '')}")
    console.print(f"  Auth detected: {result.get('authenticated', False)}")
    console.print(f"  Final URL:     {result.get('final_url', '')[:120]}")
    if not result.get("authenticated", False):
        console.print("[yellow]I did not detect job cards before timeout. You can still retry handshake-sync with this session state.[/yellow]")
    console.print("\nNext: [bold]applypilot handshake-sync --headless[/bold]")


@app.command("handshake-apply")
def handshake_apply(
    max_jobs: int = typer.Option(10, "--max-jobs", help="Maximum number of Handshake jobs to apply to."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit_score threshold."),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--submit",
        help="Fill forms but do not click Submit (default). Pass --submit to actually apply.",
    ),
    headless: bool = typer.Option(False, "--headless", help="Run browser headless."),
    session_state: str = typer.Option(
        "",
        "--session-state",
        help="Path to Playwright storage state JSON. Defaults to ~/.applypilot/handshake_state.json.",
    ),
    season: list[str] = typer.Option(
        None,
        "--season",
        help=(
            "Restrict to jobs mentioning this string in title or description. "
            "Repeatable: --season 'Fall 2026' --season 'Summer 2026'."
        ),
    ),
    bootstrap_resumes: bool = typer.Option(
        True,
        "--bootstrap-resumes/--no-bootstrap-resumes",
        help=(
            "Attach a static per-role PDF (Quant/SWE/Analyst/General) when "
            "tailored_resume_path is NULL. DB rows are not mutated."
        ),
    ),
    role_resumes_dir: str = typer.Option(
        "",
        "--role-resumes-dir",
        help=(
            "Override path to the directory holding SWE/Quant/General/Analyst "
            "subdirs. Defaults to ../../Resumes/Resume/Roles relative to the "
            "project (or env APPLYPILOT_ROLE_RESUMES_DIR)."
        ),
    ),
    use_llm_qa: bool = typer.Option(
        True,
        "--llm-qa/--no-llm-qa",
        help=(
            "Answer free-text screener questions via the configured LLM. "
            "Silently degrades to known-answers-only if no LLM is configured."
        ),
    ),
    min_sleep: float = typer.Option(
        30.0,
        "--min-sleep",
        help="Minimum seconds of random sleep between submits (human-pacing).",
    ),
    max_sleep: float = typer.Option(
        90.0,
        "--max-sleep",
        help="Maximum seconds of random sleep between submits (human-pacing).",
    ),
    max_per_day: int = typer.Option(
        30,
        "--max-per-day",
        help=(
            "Hard cap on real submits per day (local time). Dry-runs don't "
            "count. Defends a university-tied account against bulk-apply "
            "flags. 30 is a conservative default."
        ),
    ),
) -> None:
    """Apply to Handshake jobs using the saved SSO session state.

    Drives Handshake's React SPA modal directly with Playwright — no Claude
    Code / MCP overhead. Pulls Handshake-site jobs from the DB matching the
    score threshold (and optional --season filter), bootstraps a static
    per-role resume PDF for jobs missing one, uses an LLM to answer
    free-text screener questions, and paces submits 30-90s apart to look
    human. Aborts automatically after 2 consecutive login_issue results.
    """
    _bootstrap()

    from applypilot.apply.handshake import run_handshake_apply
    from pathlib import Path as _Path

    season_keywords = list(season) if season else None
    role_dir_override = (
        _Path(role_resumes_dir).expanduser() if role_resumes_dir.strip() else None
    )

    console.print("\n[bold blue]Handshake Apply[/bold blue]")
    console.print(f"  Max jobs:        {max_jobs}")
    console.print(f"  Min score:       {min_score}")
    console.print(f"  Mode:            {'DRY-RUN (no submit)' if dry_run else 'SUBMIT'}")
    console.print(f"  Headless:        {headless}")
    console.print(f"  Bootstrap PDFs:  {bootstrap_resumes}")
    console.print(f"  LLM Q&A:         {use_llm_qa}")
    if season_keywords:
        console.print(f"  Season filter:   {', '.join(season_keywords)}")
    console.print(f"  Sleep range:     {min_sleep:.0f}-{max_sleep:.0f}s")
    console.print(f"  Daily cap:       {max_per_day} (real submits, local-day)")
    if session_state.strip():
        console.print(f"  Session:         {session_state.strip()}")
    if role_dir_override is not None:
        console.print(f"  Role PDFs dir:   {role_dir_override}")

    try:
        stats = run_handshake_apply(
            max_jobs=max_jobs,
            min_score=min_score,
            dry_run=dry_run,
            headless=headless,
            session_state_path=session_state.strip() or None,
            season_keywords=season_keywords,
            bootstrap_resumes=bootstrap_resumes,
            role_resumes_dir=role_dir_override,
            use_llm_qa=use_llm_qa,
            min_sleep_seconds=min_sleep,
            max_sleep_seconds=max_sleep,
            max_per_day=max_per_day,
        )
    except RuntimeError as e:
        console.print(f"\n[red]Handshake apply failed:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold]Apply complete[/bold]")
    console.print(f"  Processed:       {stats.get('processed', 0)}")
    console.print(f"  Applied:         {stats.get('applied', 0)}")
    console.print(f"  Dry runs:        {stats.get('dry_runs', 0)}")
    console.print(f"  Failed:          {stats.get('failed', 0)}")
    console.print(f"  Bootstrapped:    {stats.get('bootstrapped', 0)}")
    if stats.get("skipped_no_resume", 0):
        console.print(f"  Skipped (no resume): {stats.get('skipped_no_resume', 0)}")
    console.print(f"  LLM Q&A status:  {stats.get('llm_qa', 'off')}")

    if stats.get("aborted"):
        if stats.get("abort_reason") == "daily_cap_reached":
            console.print(
                f"\n[yellow]Daily cap reached:[/yellow] {stats.get('today_applied', 0)}/{stats.get('max_per_day', 0)} "
                "submits already done today. Re-run after midnight (local time) or raise --max-per-day."
            )
        else:
            console.print(
                f"\n[red]Aborted:[/red] {stats.get('abort_reason')}. "
                "Run [bold]applypilot handshake-login[/bold] to refresh the SSO session state."
            )
    elif stats.get("processed", 0) == 0:
        console.print(
            "\n[yellow]No eligible Handshake jobs found.[/yellow] "
            "Check fit_score threshold and season filter; ensure jobs were sync'd."
        )
    elif dry_run and stats.get("dry_runs", 0) > 0:
        console.print(
            "\n[dim]Dry-run jobs remain re-applyable. Re-run with [bold]--submit[/bold] to actually apply.[/dim]"
        )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def doctor() -> None:
    """Check your setup and diagnose missing requirements."""
    import shutil
    from applypilot.config import (
        load_env, PROFILE_PATH, RESUME_PATH, RESUME_PDF_PATH,
        SEARCH_CONFIG_PATH, ENV_PATH, get_chrome_path,
    )

    load_env()

    ok_mark = "[green]OK[/green]"
    fail_mark = "[red]MISSING[/red]"
    warn_mark = "[yellow]WARN[/yellow]"

    results: list[tuple[str, str, str]] = []  # (check, status, note)

    # --- Tier 1 checks ---
    # Profile
    if PROFILE_PATH.exists():
        results.append(("profile.json", ok_mark, str(PROFILE_PATH)))
    else:
        results.append(("profile.json", fail_mark, "Run 'applypilot init' to create"))

    # Resume
    if RESUME_PATH.exists():
        results.append(("resume.txt", ok_mark, str(RESUME_PATH)))
    elif RESUME_PDF_PATH.exists():
        results.append(("resume.txt", warn_mark, "Only PDF found — plain-text needed for AI stages"))
    else:
        results.append(("resume.txt", fail_mark, "Run 'applypilot init' to add your resume"))

    # Search config
    if SEARCH_CONFIG_PATH.exists():
        results.append(("searches.yaml", ok_mark, str(SEARCH_CONFIG_PATH)))
    else:
        results.append(("searches.yaml", warn_mark, "Will use example config — run 'applypilot init'"))

    # jobspy (discovery dep installed separately)
    try:
        import jobspy  # noqa: F401
        results.append(("python-jobspy", ok_mark, "Job board scraping available"))
    except ImportError:
        results.append(("python-jobspy", warn_mark,
                        "pip install --no-deps python-jobspy && pip install pydantic tls-client requests markdownify regex"))

    # --- Tier 2 checks ---
    import os
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_local = bool(os.environ.get("LLM_URL"))
    use_claude_code_llm = os.environ.get("LLM_PROVIDER", "").strip().lower() == "claude_code"
    claude_bin = shutil.which("claude")
    if has_gemini:
        model = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
        results.append(("LLM API key", ok_mark, f"Gemini ({model})"))
    elif use_claude_code_llm and claude_bin:
        model = os.environ.get("LLM_MODEL", "haiku")
        results.append(("LLM backend", ok_mark, f"Claude Code CLI ({model})"))
    elif use_claude_code_llm and not claude_bin:
        results.append(("LLM backend", fail_mark, "LLM_PROVIDER=claude_code set, but Claude Code CLI not found"))
    elif has_openai:
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        results.append(("LLM API key", ok_mark, f"OpenAI ({model})"))
    elif has_local:
        results.append(("LLM API key", ok_mark, f"Local: {os.environ.get('LLM_URL')}"))
    else:
        results.append(("LLM backend", fail_mark,
                "Set GEMINI_API_KEY or LLM_PROVIDER=claude_code in ~/.applypilot/.env (run 'applypilot init')"))

    # --- Tier 3 checks ---
    # Claude Code CLI
    claude_bin = shutil.which("claude")
    if claude_bin:
        results.append(("Claude Code CLI", ok_mark, claude_bin))
    else:
        results.append(("Claude Code CLI", fail_mark,
                        "Install from https://claude.ai/code (needed for auto-apply)"))

    # Chrome
    try:
        chrome_path = get_chrome_path()
        results.append(("Chrome/Chromium", ok_mark, chrome_path))
    except FileNotFoundError:
        results.append(("Chrome/Chromium", fail_mark,
                        "Install Chrome or set CHROME_PATH env var (needed for auto-apply)"))

    # Node.js / npx (for Playwright MCP)
    npx_bin = shutil.which("npx")
    if npx_bin:
        results.append(("Node.js (npx)", ok_mark, npx_bin))
    else:
        results.append(("Node.js (npx)", fail_mark,
                        "Install Node.js 18+ from nodejs.org (needed for auto-apply)"))

    # CapSolver (optional)
    capsolver = os.environ.get("CAPSOLVER_API_KEY")
    if capsolver:
        results.append(("CapSolver API key", ok_mark, "CAPTCHA solving enabled"))
    else:
        results.append(("CapSolver API key", "[dim]optional[/dim]",
                        "Set CAPSOLVER_API_KEY in .env for CAPTCHA solving"))

    # --- Render results ---
    console.print()
    console.print("[bold]ApplyPilot Doctor[/bold]\n")

    col_w = max(len(r[0]) for r in results) + 2
    for check, status, note in results:
        pad = " " * (col_w - len(check))
        console.print(f"  {check}{pad}{status}  [dim]{note}[/dim]")

    console.print()

    # Tier summary
    from applypilot.config import get_tier, TIER_LABELS
    tier = get_tier()
    console.print(f"[bold]Current tier: Tier {tier} — {TIER_LABELS[tier]}[/bold]")

    if tier == 1:
        console.print("[dim]  → Tier 2 unlocks: scoring, tailoring, cover letters (needs an LLM API key)[/dim]")
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")
    elif tier == 2:
        console.print("[dim]  → Tier 3 unlocks: auto-apply (needs Claude Code CLI + Chrome + Node.js)[/dim]")

    console.print()


if __name__ == "__main__":
    app()
