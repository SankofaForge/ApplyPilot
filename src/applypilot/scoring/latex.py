"""LaTeX rendering for tailored resumes — stage 2 of the tailoring pipeline.

Stage 1 (``scoring/tailor.py``) produces validated, judged JSON. This module
takes that JSON plus the user's own role template and asks the LLM to author a
complete ``.tex`` in the user's exact style — same preamble, macros, header,
and Education block as the template, with the Summary / Skills / Experience /
Projects content replaced by the tailored JSON. No Jinja templating: the model
writes the LaTeX. ``pdflatex`` then compiles it to a PDF that matches the
hand-built design.

Because the LaTeX-authoring step is a second LLM pass, the de-macro'd text is
re-scanned for banned words before compiling — content was already locked in
JSON, this just catches rephrasing that slips a banned word back in.
"""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from applypilot.config import RESUME_ROLES, resume_template_path
from applypilot.llm import get_client
from applypilot.scoring.validator import BANNED_WORDS

log = logging.getLogger(__name__)

# Image asset extensions copied next to the .tex at compile time (github-mark,
# QR codes, etc. live in the same directory as the template).
_ASSET_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".eps"}

_PDFLATEX_PASSES = 2  # second pass resolves hyperref/\ref cross-references


# ── De-macro text extraction (for re-validation) ──────────────────────────

def extract_text_from_latex(tex: str) -> str:
    """Best-effort plain text from a LaTeX body, for banned-word scanning.

    Not a real LaTeX parser — strips the preamble, comments, and control
    sequences while keeping the human-readable argument text. Good enough to
    catch banned words; not used for rendering.
    """
    # Keep only the document body.
    m = re.search(r"\\begin\{document\}(.*?)\\end\{document\}", tex, re.DOTALL)
    body = m.group(1) if m else tex

    # Drop line comments (a real % escaped as \% is preserved).
    body = re.sub(r"(?<!\\)%.*", "", body)

    # \href{url}{shown text} -> shown text
    body = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", body)

    # Text-wrapping macros: keep the argument, drop the command.
    for cmd in ("textbf", "textit", "emph", "small", "Huge", "large", "resumeItem"):
        body = re.sub(rf"\\{cmd}\s*\{{", "{", body)

    # Remaining control sequences (\section*, \vspace{..}, \\, etc.) -> space.
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    body = body.replace("\\\\", " ")

    # Strip leftover braces and collapse whitespace.
    body = body.replace("{", " ").replace("}", " ")
    body = re.sub(r"[ \t]+", " ", body)
    return "\n".join(line.strip() for line in body.splitlines() if line.strip())


def _strip_code_fences(raw: str) -> str:
    """Remove markdown fences if the model wrapped its LaTeX in them."""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        # Prefer the fenced block that looks like a LaTeX document.
        for part in parts[1::2]:
            part = re.sub(r"^(latex|tex)\b", "", part.strip(), flags=re.IGNORECASE).strip()
            if "\\documentclass" in part or "\\begin{document}" in part:
                return part
    return raw


# ── Role classification (opt-in auto-routing) ─────────────────────────────

def classify_role(job: dict, default: str) -> str:
    """Classify a job into one of RESUME_ROLES via a cheap LLM read.

    Returns ``default`` if the response doesn't name a known role.
    """
    desc = (job.get("full_description") or "")[:2000]
    prompt = (
        "Classify this job into exactly one resume track. "
        f"Reply with ONLY one word from: {', '.join(RESUME_ROLES)}.\n\n"
        f"TITLE: {job.get('title', '')}\n"
        f"DESCRIPTION:\n{desc}"
    )
    try:
        client = get_client()
        resp = client.chat(
            [{"role": "user", "content": prompt}], max_tokens=8, temperature=0.0
        ).strip()
    except Exception:
        log.debug("Role classification failed; using default %s", default, exc_info=True)
        return default

    for role in RESUME_ROLES:
        if role.lower() in resp.lower():
            return role
    return default


def resolve_role(job: dict, *, default_role: str, auto_role: bool) -> str:
    """Pick the role variant for a job: classify when auto_role, else default."""
    if auto_role:
        return classify_role(job, default_role)
    return default_role


# ── LaTeX authoring (LLM writes the .tex) ──────────────────────────────────

def _build_render_prompt(template_tex: str) -> str:
    return f"""You are a LaTeX typesetter. You are given a resume template (the user's exact \
style, with their preamble, custom macros, header, and Education section) and a \
JSON object holding tailored resume content.

Produce a COMPLETE LaTeX document that compiles with pdflatex.

RULES:
- Reuse the template's preamble, \\newcommand macros, and the EDUCATION section \
VERBATIM. Do not alter them.
- HEADER: keep the candidate's name and the contact line exactly as in the \
template. Inject the JSON "title" as a target-role line placed directly UNDER \
the name and ABOVE the contact line, styled to match the header (e.g. right \
after the name's `\\\\`, add `\\small\\textbf{{<title>}} \\\\` then keep the \
existing `\\vspace{{...}}`). If the header already has a role/title line under \
the name, REPLACE its text with the JSON title rather than adding a second one. \
Do not duplicate the name or contact line.
- Replace ONLY the Technical Skills, Experience, and Projects content with the \
JSON content, using the SAME macros the template uses (\\resumeSubheading, \
\\resumeItem, \\resumeProjectHeading, \\section*, etc.).
- SUMMARY: if the template has a Summary/Objective section, replace its text \
with the JSON "summary". If the template has NO such section, do NOT add one — \
let the summary inform emphasis only.
- Map JSON fields: title -> header role line; skills -> Technical Skills lines; \
experience[] -> Experience entries; projects[] -> Projects entries.
- Preserve any \\href links and the github icon usage from the template's style.
- Escape LaTeX specials in content text: & % $ # _ become \\& \\% \\$ \\# \\_ . \
Leave existing template commands untouched.
- Keep it to ONE page. Match the template's verb density and bullet counts.
- Output ONLY the LaTeX source. No markdown fences, no commentary, no preamble \
text like "here is".

TEMPLATE:
{template_tex}"""


def render_latex(data: dict, template_tex: str, *, max_retries: int = 1) -> str:
    """Have the LLM author a tailored .tex from validated JSON + the template.

    Re-scans the de-macro'd output for banned words and retries once with a
    corrective note before giving up (and returning the last attempt anyway —
    the JSON content was already validated upstream).
    """
    import json as _json

    client = get_client()
    system = _build_render_prompt(template_tex)
    avoid = ""
    tex = ""

    for attempt in range(max_retries + 1):
        user = (
            f"TAILORED CONTENT (JSON):\n{_json.dumps(data, indent=2)}\n\n"
            f"{avoid}Return the complete LaTeX document:"
        )
        raw = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=4096,
            temperature=0.1,
        )
        tex = _strip_code_fences(raw)

        text = extract_text_from_latex(tex).lower()
        hits = [w for w in BANNED_WORDS if w.lower() in text]
        if not hits or attempt == max_retries:
            if hits:
                log.warning(
                    "LaTeX render still contains banned words after retry: %s",
                    ", ".join(hits[:5]),
                )
            return tex
        avoid = (
            "Your previous draft used these BANNED words — rewrite without them: "
            f"{', '.join(hits)}.\n\n"
        )

    return tex


# ── Compilation ────────────────────────────────────────────────────────────

def compile_pdf(tex_source: str, out_path: Path, asset_dir: Path) -> Path:
    """Compile LaTeX source to a PDF via pdflatex in an isolated temp dir.

    Copies image assets (github-mark.pdf, QR codes) from ``asset_dir`` next to
    the source so \\includegraphics resolves. Runs pdflatex twice for
    hyperref refs. Raises RuntimeError with the log tail on failure.
    """
    out_path = Path(out_path)
    if shutil.which("pdflatex") is None:
        raise RuntimeError("pdflatex not found on PATH — install a TeX distribution.")

    with tempfile.TemporaryDirectory(prefix="applypilot-tex-") as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "resume.tex").write_text(tex_source, encoding="utf-8")

        # Copy sibling image assets the template references.
        for asset in asset_dir.iterdir():
            if asset.is_file() and asset.suffix.lower() in _ASSET_SUFFIXES:
                shutil.copy2(asset, tmp_dir / asset.name)

        log_tail = ""
        for _ in range(_PDFLATEX_PASSES):
            proc = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(tmp_dir),
                    str(tmp_dir / "resume.tex"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            log_tail = (proc.stdout or "")[-1500:]
            if proc.returncode != 0:
                raise RuntimeError(f"pdflatex failed:\n{log_tail}")

        produced = tmp_dir / "resume.pdf"
        if not produced.exists():
            raise RuntimeError(f"pdflatex produced no PDF:\n{log_tail}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, out_path)

    log.info("LaTeX PDF generated: %s", out_path)
    return out_path


# ── Orchestrator ───────────────────────────────────────────────────────────

def tailored_json_to_pdf(
    data: dict,
    job: dict,
    out_pdf_path: Path,
    *,
    default_role: str,
    auto_role: bool = False,
    save_tex: bool = True,
) -> tuple[Path, str]:
    """Render validated JSON to a PDF through the user's LaTeX template.

    Returns ``(pdf_path, role_used)``. Raises if templates aren't configured
    for the resolved role or if compilation fails — the caller is expected to
    fall back to the HTML renderer.
    """
    role = resolve_role(job, default_role=default_role, auto_role=auto_role)
    template_path = resume_template_path(role)
    if template_path is None:
        raise RuntimeError(
            f"No LaTeX template for role '{role}'. "
            "Set APPLYPILOT_RESUME_TEMPLATES_DIR (and ensure a <role>/*.tex exists)."
        )

    template_tex = template_path.read_text(encoding="utf-8")
    tex_source = render_latex(data, template_tex)

    if save_tex:
        Path(out_pdf_path).with_suffix(".tex").write_text(tex_source, encoding="utf-8")

    compile_pdf(tex_source, out_pdf_path, asset_dir=template_path.parent)
    return Path(out_pdf_path), role
