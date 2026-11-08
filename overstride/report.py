"""M5: the report. One self-contained HTML file, no JavaScript, no network.

`docs/phase16-demo-design.md` section 5 stage 10 and milestone M5.

**What the file is.** Inline CSS, inline `<svg>` traces drawn by this module,
overlay frames as inline base64 PNGs. No stylesheet, script, font or image is
fetched; the only `http` strings in the output are the attribution citations in
the footer, which are links a reader may follow, not resources the page loads.
It opens by double-clicking, reads correctly with JavaScript disabled, and can
be archived as a single file.

**What the document is.** Not a dashboard: a **calibration certificate for a
video**. The instrument states its own uncertainty before it states its
reading, so every number is rendered with its interval at the same weight, and
the two uncertainties in play are kept visibly apart and never added:

  * **stride-to-stride spread** -- how much this runner varied across the
    strides in this clip. A property of the running.
  * **measurement error** -- `overstride.budget` Row B, measured against mocap
    on five never-opened subjects. A property of the pipeline.

**What is not here.** `[A4]` no hip trace, no hip number, no hip axis -- not
collapsed, not greyed, not behind a toggle. The word does not appear in the
output at all, because the reason the hip is excluded is that its error is a
learned convention, which no amount of labelling fixes. `--debug` writes a
separate file; a normal run never produces one. The M6 guard asserts both.
"""

from __future__ import annotations

import base64
import html
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from overstride.budget import (NOT_MEASURED, ROW_B_CAVEATS,  # noqa: E402
                               ROW_LABELS, UNMEASURED, Budget, load_budget)
from overstride.strides import (STRIDE_COUNT_BAR,  # noqa: E402
                                STRIDE_COUNT_CLIPS,
                                STRIDE_COUNT_CLIPS_PASSED,
                                peaks_per_stride, stride_average)

# H36M-17 bones, for the overlay. Pairs of joint indices.
BONES = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
         (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15),
         (15, 16))
LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}

SIDE_NAME = {"l": "left", "r": "right"}
LIMB_PHRASE = {"near": "nearer the camera",
               "far": "further from the camera"}
LIMB_PHRASE = {"near": "nearer the camera", "far": "further from the camera"}
N_OVERLAY_FRAMES = 6
OVERLAY_WIDTH = 320

ATTRIBUTION = (
    ("MotionAGFormer (the 2D-to-3D lifter)",
     "Mehraban, Adeli &amp; Taati, WACV 2024",
     "https://arxiv.org/abs/2310.16288"),
    ("AthleticsPose (the checkpoint's training data)",
     "Suzuki et al., 2025. CC BY-NC-SA 4.0, non-commercial",
     "https://arxiv.org/abs/2507.12905"),
    ("BML MoVi (where the error was measured)",
     "Ghorbani et al., PLOS ONE 2021. Non-commercial research; no frame of it "
     "appears here",
     "https://doi.org/10.1371/journal.pone.0253157"),
    ("Keypoint R-CNN (the 2D detector)",
     "He et al., ICCV 2017, via torchvision",
     "https://arxiv.org/abs/1703.06870"),
)


# ------------------------------------------------------------- plumbing ----

def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _mark(text: str) -> str:
    """Escape, then give the UNMEASURED token its own weight on the page."""
    return _e(text).replace(
        UNMEASURED, f'<b class="unmeasured">{UNMEASURED}</b>')


def _f(x, digits: int = 2) -> str:
    """A number, or the UNMEASURED word. Never a blank cell."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return UNMEASURED
    return UNMEASURED if not np.isfinite(v) else f"{v:.{digits}f}"


def _deg(x, digits: int = 1) -> str:
    s = _f(x, digits)
    return s if s == UNMEASURED else f"{s}°"


# ----------------------------------------------------------- the overlay ---

def _bbox(kp):
    vis = kp[np.isfinite(kp[:, :2]).all(axis=1) & (np.abs(kp[:, :2]).sum(1) > 0)]
    if len(vis) < 4:
        return None
    return (vis[:, 0].min(), vis[:, 1].min(), vis[:, 0].max(), vis[:, 1].max())


def overlay_frames(frames, kp2d, detected=None, n: int = N_OVERLAY_FRAMES,
                   width: int = OVERLAY_WIDTH) -> list:
    """Evenly spaced frames with the detected skeleton drawn over them.

    Each thumbnail is cropped to the runner, because at a realistic filming
    distance the figure is a small part of an 800x600 frame and an uncropped
    thumbnail shows nothing. **One crop box, sized once across all the chosen
    frames and then centred on the runner in each**, so every thumbnail has the
    same dimensions and the figure's position is comparable between them.

    Returned as base64 PNG strings for inlining. Nothing is written to disk,
    which is what keeps a dataset frame inside `data/` even when a test
    renders a report.
    """
    import cv2

    frames = list(frames)
    if not frames or kp2d is None:
        return []
    kp2d = np.asarray(kp2d)
    ok = (np.ones(len(frames), bool) if detected is None
          else np.asarray(detected, bool)[:len(frames)])
    idx = [int(i) for i in np.linspace(0, len(frames) - 1, n).astype(int)
           if ok[i]]
    if not idx:
        return []
    fh, fw = frames[0].shape[:2]

    boxes = [b for b in (_bbox(kp2d[i]) for i in idx) if b is not None]
    if boxes:
        bw = max(b[2] - b[0] for b in boxes) * 1.5
        bh = max(b[3] - b[1] for b in boxes) * 1.22
        bw = max(bw, bh * 0.62)                  # a runner-shaped box
        cw, ch = int(min(fw, max(48, bw))), int(min(fh, max(48, bh)))
    else:
        cw, ch = fw, fh

    out = []
    for i in idx:
        img = np.ascontiguousarray(frames[i]).copy()
        kp = kp2d[i]
        for a, b in BONES:
            pa, pb = kp[a, :2], kp[b, :2]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            if abs(pa).sum() == 0 or abs(pb).sum() == 0:
                continue
            colour = ((236, 196, 92) if (a, b) in LEFT_BONES
                      else (92, 79, 29))          # BGR: left warm, right dark
            cv2.line(img, tuple(np.round(pa).astype(int)),
                     tuple(np.round(pb).astype(int)), colour, 2, cv2.LINE_AA)
        for j in range(kp.shape[0]):
            p = kp[j, :2]
            if np.isfinite(p).all() and abs(p).sum() > 0:
                cv2.circle(img, tuple(np.round(p).astype(int)), 2,
                           (247, 248, 246), -1, cv2.LINE_AA)

        box = _bbox(kp)
        if box is not None:
            cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            x0 = int(np.clip(cx - cw / 2, 0, fw - cw))
            y0 = int(np.clip(cy - ch / 2, 0, fh - ch))
            img = img[y0:y0 + ch, x0:x0 + cw]
        h, w = img.shape[:2]
        small = cv2.resize(img, (width, max(1, int(round(h * width / w)))),
                           interpolation=cv2.INTER_AREA)
        good, buf = cv2.imencode(".png", small)
        if good:
            out.append((i, base64.b64encode(buf.tobytes()).decode()))
    return out


# -------------------------------------------------------------- the SVG ----

def _path(xs, ys) -> str:
    pts = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
    if not pts:
        return ""
    return "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts)


def trace_svg(series: dict, title: str, y_label: str, x_label: str,
              width: int = 460, height: int = 210,
              bands: dict | None = None) -> str:
    """A line chart, drawn here, with an optional spread band per series.

    `series` maps a legend label to a 1-D array. `bands` maps the same labels
    to (lower, upper) arrays. Linear axes, a faint grid, the endpoint of each
    line marked -- no chart library, so there is nothing to pin and nothing to
    fetch.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 10, 22, 34
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    vals = [v for arr in series.values() for v in np.asarray(arr, float)
            if np.isfinite(v)]
    if bands:
        vals += [v for lu in bands.values() for arr in lu
                 for v in np.asarray(arr, float) if np.isfinite(v)]
    if not vals:
        return (f'<figure class="chart"><figcaption>{_e(title)}</figcaption>'
                f'<p class="unmeasured">{UNMEASURED}</p></figure>')
    lo, hi = float(min(vals)), float(max(vals))
    if hi - lo < 1e-6:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    lo, hi = lo - 0.08 * span, hi + 0.08 * span

    n = max(len(np.asarray(next(iter(series.values())))), 2)

    def X(i):
        return pad_l + iw * (i / (n - 1))

    def Y(v):
        return pad_t + ih * (1.0 - (v - lo) / (hi - lo))

    ticks = np.linspace(lo, hi, 4)
    grid = "".join(
        f'<line class="grid" x1="{pad_l}" y1="{Y(t):.1f}" '
        f'x2="{width - pad_r}" y2="{Y(t):.1f}"/>'
        f'<text class="tick" x="{pad_l - 6}" y="{Y(t) + 3.5:.1f}" '
        f'text-anchor="end">{t:.0f}</text>' for t in ticks)

    body = []
    for k, (label, arr) in enumerate(series.items()):
        arr = np.asarray(arr, float)
        cls = f"s{k}"
        if bands and label in bands:
            lower, upper = (np.asarray(a, float) for a in bands[label])
            fwd = _path([X(i) for i in range(len(upper))],
                        [Y(v) for v in upper])
            back = [(X(i), Y(v)) for i, v in reversed(list(enumerate(lower)))
                    if np.isfinite(v)]
            if fwd and back:
                body.append(
                    f'<path class="band {cls}" d="{fwd} L '
                    + " L ".join(f"{x:.2f} {y:.2f}" for x, y in back)
                    + ' Z"/>')
        d = _path([X(i) for i in range(len(arr))], [Y(v) for v in arr])
        if d:
            body.append(f'<path class="line {cls}" d="{d}"/>')
            last = [i for i, v in enumerate(arr) if np.isfinite(v)]
            if last:
                body.append(f'<circle class="end {cls}" cx="{X(last[-1]):.2f}"'
                            f' cy="{Y(arr[last[-1]]):.2f}" r="2.6"/>')

    legend = "".join(
        f'<span class="key s{k}">{_e(label)}</span>'
        for k, label in enumerate(series))
    return (
        f'<figure class="chart">'
        f'<figcaption>{_e(title)}</figcaption>'
        f'<div class="legend">{legend}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_e(title)}">'
        f'{grid}'
        f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" '
        f'y2="{pad_t + ih}"/>'
        f'<line class="axis" x1="{pad_l}" y1="{pad_t + ih}" '
        f'x2="{width - pad_r}" y2="{pad_t + ih}"/>'
        f'{"".join(body)}'
        f'<text class="axlabel" x="{pad_l}" y="{height - 6}">'
        f'{_e(x_label)}</text>'
        f'<text class="axlabel" x="{pad_l - 40}" y="{pad_t - 9}">'
        f'{_e(y_label)}</text>'
        f'</svg></figure>')


def uncertainty_bar(value: float, ci, width: int = 180) -> str:
    """The reading's interval, drawn to scale under the reading.

    The one piece of deliberate visual weight in the document: the interval is
    seen, not read out of a parenthesis.
    """
    if ci is None or not np.isfinite(value):
        return ""
    lo, hi = float(ci[0]), float(ci[1])
    span = max(hi, value) * 1.15 or 1.0
    x = lambda v: 2 + (width - 4) * (v / span)          # noqa: E731
    return (f'<svg class="ubar" viewBox="0 0 {width} 16" role="img" '
            f'aria-label="interval {lo:.2f} to {hi:.2f} degrees">'
            f'<line class="ubar-axis" x1="2" y1="12" x2="{width - 2}" '
            f'y2="12"/>'
            f'<line class="ubar-span" x1="{x(lo):.1f}" y1="8" '
            f'x2="{x(hi):.1f}" y2="8"/>'
            f'<line class="ubar-cap" x1="{x(lo):.1f}" y1="4" '
            f'x2="{x(lo):.1f}" y2="12"/>'
            f'<line class="ubar-cap" x1="{x(hi):.1f}" y1="4" '
            f'x2="{x(hi):.1f}" y2="12"/>'
            f'<circle class="ubar-dot" cx="{x(value):.1f}" cy="8" r="3"/>'
            f'</svg>')


# ------------------------------------------------------------- the CSS -----

CSS = """
:root{
  --paper:#f7f8f6; --ink:#15191c; --soft:#5a6361; --rule:#c9cfcb;
  --accent:#1d4f5c; --accent-2:#8a6a1e; --band:#9fb8bc; --band-2:#d8c99a;
  --caution:#8a5a1e; --caution-bg:#f3ead9; --exp:#6b6472; --exp-bg:#f1eff3;
  --stop:#7a2e2e; --stop-bg:#f6e4e2;
  --serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  --mono:"DejaVu Sans Mono","Cascadia Mono",Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#101314; --ink:#e7ebe9; --soft:#9aa5a2; --rule:#2c3335;
    --accent:#7fc3d1; --accent-2:#d6b36a; --band:#2a4b53; --band-2:#4a4228;
    --caution:#e0b070; --caution-bg:#2a2317; --exp:#b3a9bd; --exp-bg:#1d1a21;
    --stop:#e39a92; --stop-bg:#2b1a18;
  }
}
:root[data-theme="dark"]{
  --paper:#101314; --ink:#e7ebe9; --soft:#9aa5a2; --rule:#2c3335;
  --accent:#7fc3d1; --accent-2:#d6b36a; --band:#2a4b53; --band-2:#4a4228;
  --caution:#e0b070; --caution-bg:#2a2317; --exp:#b3a9bd; --exp-bg:#1d1a21;
  --stop:#e39a92; --stop-bg:#2b1a18;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--serif);font-size:16px;line-height:1.55;
  -webkit-text-size-adjust:100%}
main{max-width:860px;margin:0 auto;padding:0 16px 72px}
h1{font-size:1.5rem;font-weight:600;letter-spacing:-.01em;margin:0 0 .15em}
h2{font-size:1.12rem;font-weight:600;margin:0 0 .5em;
  padding-bottom:.3em;border-bottom:1px solid var(--rule)}
h3{font-size:.98rem;font-weight:600;margin:1.4em 0 .3em}
p{margin:.55em 0;max-width:66ch}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}
section{margin:2.4rem 0 0}
.masthead{padding:1.8rem 0 1.1rem;border-bottom:2px solid var(--ink)}
.masthead .meta{font-family:var(--mono);font-size:.76rem;color:var(--soft);
  margin:.6em 0 0;display:flex;flex-wrap:wrap;gap:.25em 1.4em}
.masthead .meta b{color:var(--ink);font-weight:500}
.masthead .meta span{white-space:nowrap}
.lede{font-size:1.02rem;color:var(--soft);max-width:62ch;margin:.5em 0 0}

/* the reading */
.readings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
  gap:1.1rem 2rem;margin:1rem 0 0}
.reading{border-top:1px solid var(--rule);padding-top:.7rem}
.reading .who{font-size:.86rem;color:var(--soft)}
.reading .who b{color:var(--ink);font-weight:600}
.reading .value{font-family:var(--mono);font-size:2.05rem;line-height:1.1;
  font-variant-numeric:tabular-nums;margin:.18em 0 0;letter-spacing:-.02em}
.reading .sub{font-family:var(--mono);font-size:.78rem;color:var(--soft);
  font-variant-numeric:tabular-nums;margin:.3em 0 0}
.reading .err{margin:.55em 0 0;font-size:.84rem}
.reading .err .n{font-family:var(--mono);font-variant-numeric:tabular-nums;
  color:var(--ink)}
.ubar{display:block;width:180px;max-width:100%;height:16px;margin:.2em 0 0}
.ubar-axis{stroke:var(--rule);stroke-width:1}
.ubar-span{stroke:var(--accent);stroke-width:2.5}
.ubar-cap{stroke:var(--accent);stroke-width:1.5}
.ubar-dot{fill:var(--accent)}
.unmeasured{font-family:var(--mono);font-size:.8rem;color:var(--caution)}

/* charts */
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:1.2rem}
.chart{margin:0}
.chart figcaption{font-size:.9rem;font-weight:600;margin-bottom:.1em}
.chart svg{width:100%;height:auto;overflow:visible}
.legend{font-family:var(--mono);font-size:.72rem;color:var(--soft);
  margin:0 0 .2em}
.key{margin-right:.9em;white-space:nowrap}
.key::before{content:"";display:inline-block;width:14px;height:0;
  border-top:2px solid currentColor;vertical-align:.25em;margin-right:.35em}
.key.s0{color:var(--accent)} .key.s1{color:var(--accent-2)}
.grid{stroke:var(--rule);stroke-width:.5;stroke-dasharray:2 3}
.axis{stroke:var(--rule);stroke-width:1}
.tick,.axlabel{font-family:var(--mono);font-size:9px;fill:var(--soft)}
.line{fill:none;stroke-width:1.9;stroke-linejoin:round}
.line.s0{stroke:var(--accent)} .line.s1{stroke:var(--accent-2)}
.band{stroke:none;opacity:.42}
.band.s0{fill:var(--band)} .band.s1{fill:var(--band-2)}
.end.s0{fill:var(--accent)} .end.s1{fill:var(--accent-2)}

/* frames */
.strip{display:grid;grid-template-columns:repeat(6,1fr);
  gap:10px;margin:.6rem 0 0}
.strip figure{margin:0}
.strip img{display:block;width:100%;height:auto;border:1px solid var(--rule);
  background:var(--paper)}
.strip figcaption{font-family:var(--mono);font-size:.68rem;color:var(--soft);
  margin-top:.2em}

/* checks */
.checks{list-style:none;padding:0;margin:.4rem 0 0}
.checks li{display:grid;grid-template-columns:5.2rem 1fr;gap:.8rem;
  padding:.45rem 0;border-top:1px solid var(--rule);font-size:.9rem}
.checks .state{font-family:var(--mono);font-size:.72rem;padding:.1rem .4rem;
  height:1.5em;text-align:center;align-self:start}
.state.pass{color:var(--accent)}
.state.warn{color:var(--caution);background:var(--caution-bg)}
.state.refuse{color:var(--stop);background:var(--stop-bg)}
.checks .what{font-weight:600}
.checks .why{color:var(--soft);font-size:.86rem}

/* tables */
table{border-collapse:collapse;width:100%;font-size:.86rem;margin:.5rem 0 0}
caption{text-align:left;color:var(--soft);font-size:.82rem;
  padding-bottom:.35em}
th,td{text-align:left;padding:.4rem .6rem .4rem 0;
  border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;font-size:.8rem;color:var(--soft)}
td.n,th.n{font-family:var(--mono);font-variant-numeric:tabular-nums;
  text-align:right;padding-right:1.2rem;white-space:nowrap}
td.src{font-family:var(--mono);font-size:.72rem;color:var(--soft);
  word-break:break-all}

/* panels */
.note{border-left:2px solid var(--rule);padding:.1rem 0 .1rem .9rem;
  color:var(--soft);font-size:.9rem;margin:.9rem 0}
.caution{border-left-color:var(--caution);color:var(--ink)}
.experimental{background:var(--exp-bg);border:1px solid var(--rule);
  padding:1rem 1.1rem;margin-top:2.4rem}
.experimental h2{color:var(--exp);border-bottom-color:var(--exp)}
.experimental .tag{font-family:var(--mono);font-size:.7rem;color:var(--exp);
  border:1px solid var(--exp);padding:.05rem .35rem;margin-left:.5rem;
  vertical-align:.15em}
.experimental .chart{max-width:430px}
.experimental .line.s0{stroke:var(--exp)}
.experimental .band.s0{fill:var(--exp);opacity:.22}
.experimental .end.s0{fill:var(--exp)}
.experimental .key.s0{color:var(--exp)}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--rule);
  font-size:.82rem;color:var(--soft)}
footer ul{list-style:none;padding:0}
footer li{margin:.35em 0}
@media (max-width:620px){
  .strip{grid-template-columns:repeat(3,1fr)}
  .readings{grid-template-columns:1fr}
  .reading .value{font-size:1.8rem}
  .checks li{grid-template-columns:4.4rem 1fr}
}
"""


# ----------------------------------------------------------- the sections --

def _reading_block(quantity_label: str, side: str, limb: str, peak,
                   spread, n_strides: int, quote, stride_ok: bool) -> str:
    err = (f'<span class="n">±{_f(quote.cell.value)}°</span> '
           f'measurement error, {quote.limb} limb '
           f'<span class="n">{quote.ci_text}</span>'
           if quote.cell is not None else
           f'<span class="unmeasured">{UNMEASURED}</span>')
    bar = (uncertainty_bar(quote.cell.value, quote.cell.ci)
           if quote.cell is not None else "")
    if stride_ok and n_strides:
        sub = (f"mean of {n_strides} stride{'s' if n_strides != 1 else ''}"
               f"; stride-to-stride spread ±{_f(spread)}°")
    else:
        sub = "highest value in the clip; strides were not segmented"
    return (
        f'<div class="reading">'
        f'<p class="who"><b>{_e(SIDE_NAME[side])} leg</b> '
        f'— {_e(LIMB_PHRASE.get(limb, limb))}</p>'
        f'<p class="who">{_e(quantity_label)}, peak</p>'
        f'<p class="value">{_deg(peak)}</p>'
        f'<p class="sub">{_e(sub)}</p>'
        f'<p class="err">{err}</p>{bar}'
        f'</div>')


def _checks_section(checks) -> str:
    if not checks:
        return ""
    items = []
    for c in checks:
        state = getattr(c, "status", "pass")
        items.append(
            f'<li><span class="state {_e(state)}">{_e(state)}</span>'
            f'<span><span class="what">{_e(getattr(c, "name", "check"))}'
            f'</span><br><span class="why">'
            f'{_e(getattr(c, "message", ""))}</span></span></li>')
    return ('<section id="checks"><h2>Input checks</h2>'
            '<p>Run before anything was measured. A warning means the clip is '
            'outside the conditions the error above was measured in, and the '
            'report says so rather than adjusting the number.</p>'
            f'<ul class="checks">{"".join(items)}</ul></section>')


def _budget_section(budget: Budget, quotes: list, fps: float) -> str:
    rows = []
    seen = set()
    for q in quotes:
        if q.cell is None or (q.quantity, q.limb) in seen:
            continue
        seen.add((q.quantity, q.limb))
        c = q.cell
        rows.append(
            f'<tr><td>{_e(c.label)}</td><td>{_e(c.limb)} limb</td>'
            f'<td class="n">{_f(c.value)}°</td>'
            f'<td class="n">{_f(c.ci[0])}, {_f(c.ci[1])}</td>'
            f'<td class="src">{_e(c.source_file)}<br>{_e(c.source_path)}</td>'
            f'</tr>')
    unmeasured = "".join(
        f'<tr><td colspan="2">{_e(what)}</td>'
        f'<td colspan="3">{_mark(status)}</td></tr>'
        for what, status in NOT_MEASURED)
    caveats = "".join(f'<p>{_e(t)}</p>' for t in ROW_B_CAVEATS)

    c_rows = []
    for rate in (50.0, 33.3, 30.0):
        cells = [budget.cell("C", q, "near", fps=rate)
                 for q in ("knee_from_segments", "thigh")]
        if not all(cells):
            continue
        c_rows.append(
            f'<tr><td class="n">{rate:.1f} fps</td>'
            + "".join(f'<td class="n">{c.value:+.2f}°</td>'
                      for c in cells)
            + f'<td class="src">{_e(cells[0].source_file)}</td></tr>')

    return (
        '<section id="budget"><h2>Where the error figures come from</h2>'
        f'<p>{_e(ROW_LABELS["B"].capitalize())}. Five subjects who had never '
        'been looked at, a pipeline pinned by content hash, and one run. Each '
        'figure below is read out of the result file named beside it when this '
        'report is built, so it cannot drift away from the measurement.</p>'
        '<table><caption>Measurement error, per quantity and limb</caption>'
        '<tr><th>quantity</th><th>limb</th><th class="n">mean abs. error</th>'
        '<th class="n">95% interval</th><th>source</th></tr>'
        f'{"".join(rows)}</table>'
        f'<div class="note">{caveats}</div>'
        '<h3>Why the filming instructions ask for 60 fps</h3>'
        f'<p>These are frame-rate penalties measured <b>in-domain on the tune '
        f'set</b>, against a native 100 Hz reference — not on new '
        f'footage. They are the reason a faster clip is worth filming: 67% of '
        f'the knee’s 30 fps penalty is incurred between 50 and 30 fps. '
        f'They are <b>not</b> a correction you can apply to the error above, '
        f'and this report never combines them with it.</p>'
        '<table><caption>Tune-set frame-rate penalty, near limb</caption>'
        '<tr><th class="n">clip rate</th><th class="n">knee</th>'
        '<th class="n">thigh</th><th>source</th></tr>'
        f'{"".join(c_rows)}</table>'
        '<h3>Conditions nobody measured</h3>'
        '<table><caption>Printed as they stand, never interpolated</caption>'
        f'{unmeasured}</table>'
        '</section>')


def _scope_section() -> str:
    return (
        '<section id="scope"><h2>What this does and does not measure</h2>'
        '<h3>It measures</h3>'
        '<p>Two sagittal angles per leg, from one side-on video: knee flexion, '
        'and thigh inclination against gravity. Both are reported with the '
        'error this pipeline carries on footage from a lab it had never seen. '
        'Left and right are always shown separately and never averaged.</p>'
        '<h3>It does not measure</h3>'
        '<p>Anything about health. This is a measurement of joint angles and '
        'nothing else: there is no score, no flag, no colour and no comparison '
        'against other people anywhere in this document. It is not a clinical '
        'assessment and makes no health claim of any kind.</p>'
        '<p>It also does not measure which leg was nearer the camera. You '
        'declared that with <code>--near-side</code>, because the pipeline '
        'inferred it correctly on only 9 of 11 calibrated clips with nothing '
        'to flag the failures. If the side was declared wrongly, the two error '
        'columns above are swapped.</p>'
        '<h3>What would make the next clip better</h3>'
        '<p>Film with the rear camera, from the side, with the runner crossing '
        'the whole frame at a steady pace, at 50 fps or faster, in good light. '
        'Everything the checks above warned about is a condition the error was '
        'not measured in.</p></section>')


def _footer(clip: str, fps: float, n_frames: int, width: int, height: int,
            resized_from, near_side: str, boundary: str) -> str:
    cites = "".join(
        f'<li>{_e(what)} — {who} '
        f'<a href="{_e(url)}">{_e(url)}</a></li>'
        for what, who, url in ATTRIBUTION)
    res = (f"{width}×{height}"
           + (f" (resized from {resized_from[0]}×{resized_from[1]})"
              if resized_from else ""))
    return (
        f'<footer><p><b>How this clip was processed.</b> '
        f'{_e(clip)}, {n_frames} frames at {fps:.3g} fps, {_e(res)}; detected '
        f'with Keypoint R-CNN on a tracked person crop, lifted to 3D with '
        f'MotionAGFormer over overlapping 81-frame windows, angles taken '
        f'against the level-camera vertical. Near side declared '
        f'<b>{_e(SIDE_NAME[near_side])}</b>. Stride cycles bounded at '
        f'{_e(boundary)}.</p>'
        f'<p>This report was generated on one computer and sent nowhere. '
        f'No frame of any research dataset appears in it.</p>'
        f'<ul>{cites}</ul></footer>')


# ------------------------------------------------------------- the report --

def render(result, strides=None, checks=None, budget: Budget | None = None,
           clip_name: str = "clip", frames=None,
           today: str | None = None) -> str:
    """Build the whole HTML document as a string.

    `result` is an `overstride.analyze.AnalysisResult`; `strides` an
    `overstride.strides.Strides` (None or `ok=False` falls back to per-frame
    traces, per M3); `checks` a list of `overstride.checks.CheckResult`.
    """
    budget = budget or load_budget()
    stride_ok = bool(strides is not None and strides.ok)
    quotes = {(q.quantity, q.side): q
              for q in budget.quotes(near_side=result.near_side,
                                     fps=result.fps)}

    # --- the reading -------------------------------------------------------
    blocks = []
    for quantity, label in (("knee_from_segments", "Knee flexion"),
                            ("thigh", "Thigh inclination against gravity")):
        for side in ("l", "r"):
            series = result.series(side, quantity)
            if stride_ok:
                pk = peaks_per_stride(series, strides.used[side])
                peak = float(np.nanmean(pk)) if pk.size else float("nan")
                spread = (float(np.nanstd(pk, ddof=1)) if pk.size > 1
                          else 0.0)
                n = int(pk.size)
            else:
                finite = np.asarray(series, float)
                finite = finite[np.isfinite(finite)]
                peak = float(finite.max()) if finite.size else float("nan")
                spread, n = float("nan"), 0
            q = quotes[(quantity, side)]
            blocks.append(_reading_block(
                label, side, q.limb, peak, spread, n, q, stride_ok))

    # --- the traces --------------------------------------------------------
    charts = []
    for quantity, label in (("knee_from_segments", "Knee flexion"),
                            ("thigh", "Thigh inclination")):
        series, bands = {}, {}
        for side in ("l", "r"):
            name = f"{SIDE_NAME[side]} leg"
            if stride_ok:
                avg = stride_average(result.series(side, quantity),
                                     strides.used[side])
                series[name] = avg["mean"]
                bands[name] = (avg["mean"] - avg["sd"], avg["mean"] + avg["sd"])
            else:
                series[name] = np.asarray(result.series(side, quantity),
                                          float)
        charts.append(trace_svg(
            series, label, "degrees",
            "% of stride cycle (0 = peak swing-phase knee flexion)"
            if stride_ok else "frame",
            bands=bands or None))

    if stride_ok:
        stride_note = (
            f'<p>Averaged over {strides.n_used["l"]} left and '
            f'{strides.n_used["r"]} right stride cycles, each stretched to its '
            f'own cycle length. The shaded band is the <b>stride-to-stride '
            f'spread of this run</b> — how much this runner varied. It is '
            f'a different quantity from the measurement error above, and the '
            f'two are never added. Cadence {_f(strides.cadence_spm, 0)} steps '
            f'per minute.</p>'
            f'<div class="note caution"><p><b>The stride count may be wrong '
            f'by a cycle or two.</b> Stride segmentation is the one part of '
            f'this pipeline with no prior measurement behind it, and its own '
            f'validation bar — {_e(STRIDE_COUNT_BAR)} against mocap '
            f'— came out {STRIDE_COUNT_CLIPS_PASSED} of '
            f'{STRIDE_COUNT_CLIPS}, so it is recorded as failed. The traces '
            f'above are therefore a convenience: the error figures in the '
            f'reading do not depend on them, and per-frame traces are what a '
            f'default run shows.</p></div>')
    else:
        reasons = ("; ".join(strides.reasons) if strides is not None
                   else "no stride segmentation was run")
        stride_note = (
            f'<div class="note caution"><p><b>Strides were not segmented, so '
            f'these are per-frame traces.</b> {_e(reasons)}. Stride averaging '
            f'is a convenience and is not part of the error budget, so '
            f'nothing above depends on it.</p></div>')

    # --- stride table ------------------------------------------------------
    if stride_ok:
        rows = "".join(
            f'<tr><td>{_e(SIDE_NAME[s])} leg</td>'
            f'<td class="n">{strides.count[s]}</td>'
            f'<td class="n">{strides.n_used[s]}</td>'
            f'<td class="n">{_f(np.mean(strides.durations_s[s]))} s</td>'
            f'<td class="n">{_f(strides.duration_cv(s))}</td></tr>'
            for s in ("l", "r"))
        stride_table = (
            '<table><caption>Stride cycles found, bounded at peak '
            'swing-phase knee flexion — not at foot contact, which this '
            'pipeline cannot see</caption>'
            '<tr><th>leg</th><th class="n">cycles</th>'
            '<th class="n">averaged</th><th class="n">mean duration</th>'
            '<th class="n">duration spread (CV)</th></tr>'
            f'{rows}</table>')
    else:
        stride_table = ""

    # --- the experimental panel -------------------------------------------
    trunk_q = quotes.get(("trunk", result.near_side))
    trunk_series = {}
    trunk_bands = {}
    side = result.near_side
    if stride_ok:
        avg = stride_average(result.series(side, "trunk"), strides.used[side])
        trunk_series["trunk lean"] = avg["mean"]
        trunk_bands["trunk lean"] = (avg["mean"] - avg["sd"],
                                     avg["mean"] + avg["sd"])
    else:
        trunk_series["trunk lean"] = np.asarray(
            result.series(side, "trunk"), float)
    trunk_err = (f'{_f(trunk_q.cell.value)}° {trunk_q.ci_text}'
                 if trunk_q and trunk_q.cell else UNMEASURED)
    experimental = (
        '<section class="experimental" id="trunk">'
        '<h2>Trunk lean<span class="tag">experimental</span></h2>'
        '<p>Read this as a direction, not a number. Trunk lean carries a '
        'measured −2.34° bias on side-on lab footage, is +1.92° '
        '[+0.583, +3.204] worse across labs than on the tune set, and is the '
        'worst-tracked quantity in the pipeline at any frame rate: r = +0.764, '
        'against +0.99 for the knee and thigh. The lifter also does not place '
        'its own upper-trunk point consistently in two different labs, and '
        'trunk lean is that axis.</p>'
        f'<p class="sub">Stated error {trunk_err}, {_e(SIDE_NAME[side])} leg '
        f'reference.</p>'
        + trace_svg(trunk_series, "Trunk lean", "degrees",
                    "% of stride cycle" if stride_ok else "frame",
                    bands=trunk_bands or None)
        + '</section>')

    # --- the frames --------------------------------------------------------
    strip = ""
    if frames is not None:
        shots = overlay_frames(frames, result.kp2d, result.detected)
        if shots:
            strip = (
                '<section id="frames"><h2>What the detector saw</h2>'
                '<p>Evenly spaced frames with the detected skeleton drawn '
                'over them. Warm lines are the left side. If these do not '
                'follow the runner, nothing below them means anything.</p>'
                '<div class="strip">'
                + "".join(f'<figure><img alt="detected skeleton, frame {i}" '
                          f'src="data:image/png;base64,{b64}">'
                          f'<figcaption>frame {i}</figcaption></figure>'
                          for i, b64 in shots)
                + '</div></section>')

    # --- travel warnings ---------------------------------------------------
    warn = ""
    notes = list(result.warnings or [])
    if notes:
        warn = ('<div class="note caution">'
                + "".join(f'<p>{_e(t)}</p>' for t in notes) + '</div>')

    detected = int(np.asarray(result.detected).sum())
    meta_items = (f'<b>{_e(clip_name)}</b>',
                  f'{result.n_frames} frames',
                  f'{result.fps:.3g} fps',
                  f'{result.width}×{result.height}',
                  f'runner found in {detected} of {result.n_frames}',
                  f'near side declared {_e(SIDE_NAME[result.near_side])}',
                  _e(today or date.today().isoformat()))
    meta = ('<p class="meta">'
            + "".join(f'<span>{m}</span>' for m in meta_items) + '</p>')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Overstride measurement report — {_e(clip_name)}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<header class="masthead">
<h1>Running kinematics, measured from one video</h1>
<p class="lede">Knee flexion and thigh inclination, both legs, with the error
this pipeline carries on footage from a lab it had never seen. The error is
part of the reading, not a footnote to it.</p>
{meta}
</header>

<section id="reading">
<h2>The reading</h2>
<div class="readings">{"".join(blocks)}</div>
{warn}
</section>

<section id="traces">
<h2>Stride-averaged traces</h2>
{stride_note}
<div class="charts">{"".join(charts)}</div>
{stride_table}
</section>

{strip}
{_checks_section(checks)}
{experimental}
{_scope_section()}
{_budget_section(budget, list(quotes.values()), result.fps)}
{_footer(clip_name, result.fps, result.n_frames, result.width,
         result.height, result.resized_from, result.near_side,
         strides.boundary if strides is not None
         else "peak swing-phase knee flexion")}
</main>
</body>
</html>
"""


def write_report(path, result, **kw) -> Path:
    """Render and write the report. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(result, **kw), encoding="utf-8")
    return path
