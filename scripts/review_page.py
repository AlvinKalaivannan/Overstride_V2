"""Render docs/REVIEW.md as a shareable page.

The markdown file is the single source of truth -- this only styles it, so the
page and the repository copy cannot drift apart. Design tokens match
scripts/phase6_page.py so the project's artifacts share one identity.

Run: .venv/Scripts/python.exe scripts/review_page.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parents[1]
FIG = REPO / "figures"

# Which markdown file to render. Both use the same styling so the review and the
# response to the validation plan read as one pair of documents.
PAGES = {
    "review": (REPO / "docs" / "REVIEW.md", REPO / "docs" / "review.html",
               "Overstride — review brief"),
    "validation": (REPO / "docs" / "validation-plan-review.md",
                   REPO / "docs" / "validation-plan-review.html",
                   "Overstride — notes on the validation plan"),
    "readiness": (REPO / "docs" / "second-cohort-readiness-review.md",
                  REPO / "docs" / "second-cohort-readiness-review.html",
                  "Overstride — notes on second-cohort readiness"),
}

# Dropped in after the named section, so a cold reader sees the finding before
# being asked to judge it.
AFTER_SECTION = {
    "1. The project, for a cold reader": (
        "phase4_degradation.png",
        "The deliverable: how the injured-limb signal degrades with monocular "
        "angular error. Diamonds are measured video error, not assumed noise."),
    "5. Current state by component": (
        "phase6_ceiling.png",
        "Every feature family tried. Everything carrying signal lands near "
        "0.61; the negative controls sit at chance."),
}


def plate(name: str, caption: str) -> str:
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure class="plate"><img src="data:image/png;base64,{b64}" '
            f'alt="{caption}"><figcaption>{caption}</figcaption></figure>')


def main() -> int:
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "review"
    if which not in PAGES:
        raise SystemExit(f"unknown page {which!r}; choose from {sorted(PAGES)}")
    SRC, OUT, PAGE_TITLE = PAGES[which]

    body = markdown.markdown(
        SRC.read_text(encoding="utf-8"),
        extensions=["tables", "attr_list", "sane_lists", "fenced_code"],
    )

    # python-markdown emits a bare <table>; wrap it so wide tables scroll inside
    # their own container instead of the page body scrolling sideways
    body = body.replace("<table>", '<div class="tablewrap"><table>')
    body = body.replace("</table>", "</table></div>")

    # insert each figure directly before the <h2> that follows its anchor section
    for anchor, (fig, cap) in (AFTER_SECTION.items() if which == "review" else ()):
        marker = f"<h2>{anchor}</h2>"
        if marker not in body:
            print(f"  WARNING: section not found, figure skipped: {anchor}")
            continue
        nxt = body.find("<h2>", body.find(marker) + len(marker))
        if nxt == -1:
            body += plate(fig, cap)
        else:
            body = body[:nxt] + plate(fig, cap) + body[nxt:]

    html = f"""<title>{PAGE_TITLE}</title>
<style>
:root {{
  --paper:#f6f7f9; --panel:#ffffff; --ink:#14171c; --muted:#66707c;
  --rule:#dde1e6; --accent:#1f6fb4; --accent-soft:#eaf1f8; --neg:#c0392b;
  --ok:#2e7d55;
  --serif:Georgia,'Iowan Old Style','Times New Roman',serif;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#101317; --panel:#171b21; --ink:#e7eaee; --muted:#98a2ae;
    --rule:#2a3038; --accent:#6aa6dd; --accent-soft:#1a2733; --neg:#e07b6c;
    --ok:#5cb98c;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#101317; --panel:#171b21; --ink:#e7eaee; --muted:#98a2ae;
  --rule:#2a3038; --accent:#6aa6dd; --accent-soft:#1a2733; --neg:#e07b6c;
  --ok:#5cb98c;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.68;-webkit-font-smoothing:antialiased}}
main{{max-width:1080px;margin:0 auto;padding:64px 28px 96px}}
h1,h2,h3{{font-family:var(--serif);font-weight:600;text-wrap:balance;
  letter-spacing:-.01em;line-height:1.2}}
h1{{font-size:clamp(2rem,4.4vw,3.1rem);margin:0 0 1.4rem;max-width:20ch}}
h2{{font-size:clamp(1.35rem,2.4vw,1.8rem);margin:3.4rem 0 .9rem;
  padding-top:1.5rem;border-top:1px solid var(--rule)}}
h2:first-of-type{{border-top:none;padding-top:0;margin-top:2.2rem}}
h3{{font-size:1.1rem;margin:2rem 0 .5rem;color:var(--accent)}}
p,ul,ol{{max-width:70ch}}
p{{margin:0 0 1.1rem}}
ul,ol{{margin:0 0 1.15rem;padding-left:1.3rem}}
li{{margin-bottom:.55rem}}
li>strong:first-child{{color:var(--ink)}}
strong{{font-weight:650}}
em{{color:var(--muted);font-style:italic}}
hr{{border:none;border-top:1px solid var(--rule);margin:2.6rem 0}}
code{{font-family:var(--mono);font-size:.875em;background:var(--accent-soft);
  padding:.1em .35em;border-radius:3px}}
pre{{background:var(--panel);border:1px solid var(--rule);padding:16px 18px;
  overflow-x:auto;font-size:.87rem;line-height:1.6}}
pre code{{background:none;padding:0}}
.tablewrap{{overflow-x:auto;margin:1.4rem 0;background:var(--panel);
  border:1px solid var(--rule)}}
table{{border-collapse:collapse;width:100%;font-size:.9rem;margin:0;
  min-width:520px}}
th,td{{text-align:left;padding:10px 14px;border-bottom:1px solid var(--rule);
  vertical-align:top}}
th{{font-family:var(--mono);font-size:.695rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);font-weight:500;
  border-bottom:1px solid var(--rule)}}
tbody tr:last-child td{{border-bottom:none}}
.plate{{margin:1.8rem 0;background:var(--panel);border:1px solid var(--rule);
  padding:14px}}
.plate img{{width:100%;height:auto;display:block}}
.plate figcaption{{font-size:.85rem;color:var(--muted);margin-top:11px;
  padding-top:10px;border-top:1px solid var(--rule);max-width:88ch}}
a{{color:var(--accent)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;
  transition:none!important}}}}
</style>
<main>
{body}
</main>
"""
    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    OUT.write_text(html, encoding="ascii")
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
