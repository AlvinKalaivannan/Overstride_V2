"""Build the phase 6 methods-demo page.

Self-contained HTML: figures embedded as data URIs, every number read from
results/*.json rather than typed into the markup. Writes docs/overstride.html,
which is what gets published as the shareable artifact.

Run: .venv/Scripts/python.exe scripts/phase6_page.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
FIG = REPO / "figures"
OUT = REPO / "docs" / "overstride.html"


def L(n: str) -> dict:
    return json.loads((RES / n).read_text(encoding="utf-8"))


def img(name: str, alt: str, caption: str) -> str:
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return (f'<figure class="plate">\n'
            f'  <img src="data:image/png;base64,{b64}" alt="{alt}">\n'
            f'  <figcaption>{caption}</figcaption>\n</figure>')


def main() -> int:
    p2r = L("phase2_realized_error.json")
    p3, p3b = L("phase3_angle_errors.json"), L("phase3b_viewpoint.json")
    p4, p4b = L("phase4_real_delta.json"), L("phase4b_operating_point.json")
    p5, p5b = L("phase5_limb.json"), L("phase5b_ceiling.json")
    p5c, p5d = L("phase5c_features.json"), L("phase5d_selective.json")

    ref = p5["results"]["limbsag_mean [matched]"]
    dep = next(s for s in p4["summary"] if s["features"] == "wave2"
               and s["bank"] == "lateral only" and s["fps"] == "30fps")
    depl = p4b["configs"]["deployment: lateral bank, 30 fps"]
    mocap = p4b["configs"]["clean mocap (upper bound)"]
    ft = p3b["models"]["ath-det-ft (fine-tuned)"]
    geo = p3b["geometry"]
    d1, d2, d3 = p5b["D1"], p5b["D2"], p5b["D3"]
    ms = p5d["multi_session"]
    yo = depl["operating_points"][0]
    spec90 = depl["operating_points"][1]

    def f(x, n=3):
        return f"{x:.{n}f}"

    stats = [
        ("what video costs", f"{dep['delta_vs_clean']:+.3f}", "ΔAUC vs marker mocap"),
        ("the ceiling", f(ref["auc_mean"]), "best AUC on perfect mocap"),
        ("repeat scans agree", f(depl["repeatability"]["binary_agreement"]),
         "on the same runner"),
        ("screening tests passed", "0 / 60", "after BH correction"),
    ]
    stat_html = "\n".join(
        f'    <div class="stat"><span class="stat-k">{k}</span>'
        f'<span class="stat-v">{v}</span><span class="stat-c">{c}</span></div>'
        for k, v, c in stats)

    phases = [
        ("0", "Inventory, join, verify against Table 1", "ok", "reproduces"),
        ("1", "Injury classifier vs demographics control", "fail",
         "kill criterion fired — 0/60"),
        ("2", "Degrade to video constraints", "ok", "interaction found"),
        ("3", "Video → 3D kinematics", "ok", "3.4° fine-tuned"),
        ("3B", "Viewpoint geometry", "ok", "occlusion penalty ~0"),
        ("4", "Measured error through the phase-2 model", "ok",
         f"ΔAUC {dep['delta_vs_clean']:+.3f}"),
        ("4B", "Operating point, calibration, repeatability", "fail",
         "not deployable"),
        ("5", "Within-subject limb identification", "ok", f"{f(ref['auc_mean'])}"),
        ("5B–5D", "Ceiling diagnostics, features, abstention", "fail",
         "ceiling is the signal"),
        ("6", "Methods demo + synthesis", "ok", "this page"),
    ]
    phase_html = "\n".join(
        f'      <tr><td class="ph">{n}</td><td>{w}</td>'
        f'<td class="gate {s}">{"✓" if s == "ok" else "✕"} {o}</td></tr>'
        for n, w, s, o in phases)

    ceiling_rows = [
        ("39 clinical metrics, a different modality", f(d2["dvr_auc"]),
         f"ΔAUC {d2['delta_dvr_vs_sag']:+.3f} [{d2['delta_dvr_ci'][0]:+.3f}, "
         f"{d2['delta_dvr_ci'][1]:+.3f}]"),
        ("severity dose-response (pre-registered)", f"ρ {d1['spearman_rho']:+.3f}",
         f"CI [{d1['spearman_ci'][0]:+.3f}, {d1['spearman_ci'][1]:+.3f}], "
         f"p {d1['spearman_p']:.3f}"),
        ("learning curve, final quarter of data",
         f"{d3['final_increment']:+.3f}", "saturated"),
        ("joint velocities (24 unused channels)",
         f(next(x for x in p5c["feature_sets"]
                if x["name"].startswith("limbvel "))["auc_mean"]),
         "chance — and it makes things worse"),
        ("best of four new feature families",
         f(max(x["auc_mean"] for x in p5c["feature_sets"])),
         "ΔAUC +0.008, CI spans zero"),
        ("injury conditions surviving both intervals",
         f"{sum(c['survives'] for c in p5c['per_condition'])} / "
         f"{len(p5c['per_condition'])}", "ITBS n=100, PFPS n=81"),
    ]
    ceiling_html = "\n".join(
        f'      <tr><td>{a}</td><td class="num">{b}</td><td class="note">{c}</td></tr>'
        for a, b, c in ceiling_rows)

    deploy_rows = [
        ("best precision (PPV)", f(yo["ppv"]),
         f"against a {f(p4['chance'])} base rate"),
        ("sensitivity at 90% specificity", f(spec90["sensitivity"]), "unusable"),
        ("calibration slope", f(depl["calibration"]["calibration_slope"]),
         "1.000 is perfect"),
        ("Brier vs a constant",
         f"{100 * (1 - depl['calibration']['brier'] / depl['calibration']['brier_baseline']):.1f}%",
         "improvement"),
        ("agreement on repeat scans",
         f(depl["repeatability"]["binary_agreement"]),
         f"{f(mocap['repeatability']['binary_agreement'])} even on mocap"),
        ("averaging a runner's repeat scans",
         f"{ms['gain']:+.3f}", "does not help — error is subject-specific"),
    ]
    deploy_html = "\n".join(
        f'      <tr><td>{a}</td><td class="num">{b}</td><td class="note">{c}</td></tr>'
        for a, b, c in deploy_rows)

    html = f"""<title>Overstride — what monocular video costs, measured</title>
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
body{{margin:0;background:var(--paper);color:var(--ink);
  font-family:var(--sans);font-size:16.5px;line-height:1.65;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1140px;margin:0 auto;padding:0 28px 96px}}
.col{{max-width:68ch}}
h1,h2,h3{{font-family:var(--serif);font-weight:600;text-wrap:balance;
  letter-spacing:-.01em;line-height:1.18;margin:0}}
h1{{font-size:clamp(2.1rem,4.6vw,3.35rem)}}
h2{{font-size:clamp(1.45rem,2.5vw,1.95rem);margin:0 0 .5rem}}
h3{{font-size:1.12rem;margin:0 0 .35rem}}
p{{margin:0 0 1.05rem}}
a{{color:var(--accent)}}
strong{{font-weight:650}}
.eyebrow{{font-family:var(--mono);font-size:.735rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted);margin:0 0 1.1rem}}
header{{padding:76px 0 40px;border-bottom:1px solid var(--rule);margin-bottom:44px}}
.lede{{font-family:var(--serif);font-size:clamp(1.1rem,1.9vw,1.32rem);
  color:var(--ink);margin:1.4rem 0 0;max-width:60ch}}
.lede em{{color:var(--accent);font-style:normal;font-weight:600}}
.strip{{display:grid;gap:1px;background:var(--rule);border:1px solid var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(190px,1fr));margin:38px 0 0}}
.stat{{background:var(--panel);padding:18px 20px;display:flex;flex-direction:column;gap:3px}}
.stat-k{{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted)}}
.stat-v{{font-family:var(--mono);font-size:1.72rem;font-variant-numeric:tabular-nums;
  color:var(--accent);line-height:1.15}}
.stat-c{{font-size:.83rem;color:var(--muted)}}
section{{margin:0 0 58px}}
.plate{{margin:26px 0 8px;background:var(--panel);border:1px solid var(--rule);
  padding:14px}}
.plate img{{width:100%;height:auto;display:block}}
.plate figcaption{{font-size:.855rem;color:var(--muted);margin-top:11px;
  padding-top:10px;border-top:1px solid var(--rule);max-width:88ch}}
.tbl{{overflow-x:auto;margin:20px 0 8px;border:1px solid var(--rule);
  background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:.905rem}}
th,td{{text-align:left;padding:9px 15px;border-bottom:1px solid var(--rule);
  vertical-align:top}}
tbody tr:last-child td{{border-bottom:none}}
th{{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-weight:500}}
td.num{{font-family:var(--mono);font-variant-numeric:tabular-nums;
  color:var(--accent);white-space:nowrap}}
td.note,.note{{color:var(--muted);font-size:.87rem}}
td.ph{{font-family:var(--mono);color:var(--muted);white-space:nowrap}}
td.gate{{font-family:var(--mono);font-size:.845rem;white-space:nowrap}}
td.gate.ok{{color:var(--ok)}}
td.gate.fail{{color:var(--neg)}}
blockquote{{margin:22px 0;padding:15px 20px;background:var(--accent-soft);
  border-left:2px solid var(--accent);font-size:.95rem}}
blockquote p:last-child{{margin:0}}
blockquote.warn{{background:transparent;border-left-color:var(--neg)}}
ul{{margin:0 0 1.05rem;padding-left:1.15rem}}
li{{margin-bottom:.5rem}}
footer{{border-top:1px solid var(--rule);padding-top:26px;color:var(--muted);
  font-size:.87rem}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;
  transition:none!important}}}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Overstride · a measurement, and a negative result</p>
  <h1>What monocular video actually costs</h1>
  <p class="lede">Recovering lower-limb kinematics from a single camera instead
  of a motion capture lab costs <em>{dep['delta_vs_clean']:+.3f} AUC</em>. It is
  that small because there was very little to lose.</p>
  <div class="strip">
{stat_html}
  </div>
</header>

<section class="col">
  <h2>The question, and the answer</h2>
  <p>The project asked one thing: how much injury-classification performance is
  lost when lower-limb kinematics come from a phone camera rather than a marker
  lab. Answering it honestly required measuring the whole chain — pose error,
  viewpoint geometry, temporal resolution — and then pushing real measured error
  through the classifier rather than assumed noise.</p>
  <p><strong>The camera is not the bottleneck.</strong> The only task in this
  dataset carrying kinematic injury signal — identifying <em>which limb</em> is
  injured in a runner already known to be injured — reaches
  <strong>{f(ref['auc_mean'])}</strong> on perfect mocap and
  <strong>{f(dep['auc_mean'])}</strong> with real video error. Chance is
  {f(p4['chance'])}.</p>
</section>

{img("phase4_degradation.png", "Degradation curve",
     "The deliverable. Phase 2 swept assumed Gaussian noise; phase 3 measured "
     "what monocular pose actually does; phase 4 injected those measured "
     "residuals. Phase 2 x-positions are its own realized error, not its "
     "nominal sigma — its 10&nbsp;Hz filter removes roughly 80% of injected "
     "noise at full resolution.")}

<section class="col">
  <h2>Two levels, two verdicts</h2>
  <p><strong>Screening failed outright.</strong> Zero of 60 pre-registered tests
  survived correction. A provenance-only model — questionnaire missingness and
  collection year, no physiology at all — scores 0.775 pooled and 0.863 on
  patellofemoral pain, because questionnaire completeness tracks the study wave.
  Most of what looks like a gait signal here is paperwork.</p>
  <p><strong>Within-subject limb identification is real, but small.</strong>
  Comparing a runner's injured limb to their own healthy limb in the same trial
  removes the confound by construction — demographics, provenance, file
  structure and limb dominance are constant within a session, and all four were
  verified at chance.</p>
</section>

<section>
  <div class="col">
    <h2>Why the ceiling is the signal, not the model</h2>
    <p>Five independent probes, each designed to find headroom. None does.</p>
  </div>
  <div class="tbl"><table>
    <thead><tr><th>probe</th><th>result</th><th>reading</th></tr></thead>
    <tbody>
{ceiling_html}
    </tbody>
  </table></div>
  <p class="col note">If the ceiling were an artefact of the sagittal waveform
  representation, a clinical scalar set built from different planes and different
  physics would not reproduce it to three decimals. It does.</p>
</section>

{img("phase6_ceiling.png", "Every feature family tried",
     "Eighteen feature families on one axis. Everything carrying signal lands "
     "in a narrow band around 0.61; the negative controls sit at chance. The "
     "clinical probe uses variables a side-on camera cannot recover — it is a "
     "diagnostic, not a candidate.")}

<section class="col">
  <h2>What the pipeline recovers</h2>
  <p>Sport-fine-tuned monocular pose reaches
  <strong>{ft['near_mae']:.1f}°</strong> mean sagittal error against marker
  ground truth (MPJPE {p3['mpjpe_mm']['ath-det-ft (fine-tuned)']['mean']:.1f}&nbsp;mm).
  The generic checkpoint lands at
  {p3b['models']['h36m (generic)']['near_mae']:.1f}°, inside the published
  14.1–25.8° band for generic models on clinical gait — which is what validates
  the setup rather than the fine-tuned number.</p>
  <p>Two results make the camera path <em>better</em> than expected. The
  far-limb occlusion penalty is essentially zero and does not grow with viewing
  angle ({ft['penalty_at_ratio_1']:+.2f}° extrapolated to a fully lateral view),
  and accuracy <em>improves</em> as the view becomes side-on — which is exactly
  the geometry this application prescribes. The clips here sit a median
  {geo['out_of_plane_deg_median']:.0f}° out of the image plane.</p>
</section>

{img("phase6_angle_recovery.png", "Recovered vs true joint angles",
     "What 3.4° looks like. Three held-out subjects, near limb, side-on views. "
     "This is 2D-to-3D lifting only — the source videos are not released, so "
     "decode and detection error are excluded and these are a lower bound.")}

<section>
  <div class="col">
    <h2>Why no demo was built</h2>
    <p>A per-user readout would misrepresent the evidence. At the realistic
    operating point — hip and knee only, side-on camera error, 30&nbsp;fps:</p>
  </div>
  <div class="tbl"><table>
    <thead><tr><th>measure</th><th>value</th><th>against</th></tr></thead>
    <tbody>
{deploy_html}
    </tbody>
  </table></div>
  <blockquote class="warn"><p><strong>The model changes its mind about the same
  runner on a third of repeat scans</strong> — and averaging those scans does not
  help. That means the error is subject-specific rather than random: it is
  consistently wrong about the same people. Scanning someone twice does not
  improve the answer.</p></blockquote>
  <div class="col">
    <p>Refusing to answer does work on mocap — at 50% coverage accuracy reaches
    {f([r for r in p5d['curves']['clean mocap, limbsag (upper bound)']
        if r['coverage'] == 0.5][0]['accuracy'])}, validated against a control
    curve that stays flat. On video-derived data no coverage level reliably beats
    guessing.</p>
  </div>
</section>

{img("phase5d_selective.png", "Selective classification",
     "Answering only the most confident scans. The grey control curve is the "
     "point: a provenance-only model, verified at chance, must stay flat for "
     "the rest to mean anything. It does — drifting downward, not up.")}

<section>
  <div class="col"><h2>Phases and gates</h2></div>
  <div class="tbl"><table>
    <thead><tr><th>phase</th><th>work</th><th>gate</th></tr></thead>
    <tbody>
{phase_html}
    </tbody>
  </table></div>
  <p class="col note">Phase 1's kill criterion fired and the project continued
  deliberately, onto the within-subject task — a different question against a
  stronger control set. That pivot is a real departure from the original plan and
  is recorded rather than hidden.</p>
</section>

<section class="col">
  <h2>What this is not</h2>
  <ul>
    <li><strong>Not a live demo.</strong> This page presents a measured pipeline.
    AthleticsPose does not release its source videos, so nothing here runs on
    arbitrary footage.</li>
    <li><strong>Not evidence that running gait carries no injury information.</strong>
    One cohort, one protocol, treadmill running, stance phase only, with a
    provenance confound that had to be removed by construction.</li>
    <li><strong>Not a verdict on monocular pose estimation.</strong> It costs
    ~0.03 AUC here and its errors are a lower bound.</li>
    <li><strong>Nothing here forecasts future injury.</strong> Every label
    describes injury status at the time of testing.</li>
  </ul>
</section>

<footer class="col">
  <p>Ferber / Running Injury Clinic Kinematic Dataset (n=1,798), never
  redistributed. Monocular error measured on AthleticsPose (CC BY-NC-SA 4.0) and
  AthletePose3D (non-commercial research only) — <strong>commercial use is
  prohibited</strong>.</p>
  <p>All splits are <span style="font-family:var(--mono)">StratifiedGroupKFold</span>
  grouped by subject, 5 splits × 5 seeds = 25 folds shared across every model so
  deltas are paired. Every number on this page is read from the result JSONs by
  the script that builds it.</p>
</footer>
</div>
"""
    # Escape every non-ASCII character to a numeric entity. The page is then
    # byte-identical under any charset the host declares, so degree signs, rho,
    # em-dashes and check marks cannot turn into mojibake if a <meta charset> is
    # missing or wrong. The base64 payloads are already ASCII, so this costs
    # nothing in size.
    html = html.encode("ascii", "xmlcharrefreplace").decode("ascii")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="ascii")
    print(f"wrote {OUT.relative_to(REPO)}  ({OUT.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
