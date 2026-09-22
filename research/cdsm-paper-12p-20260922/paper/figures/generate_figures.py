"""Extract frozen summaries and write editable PGFPlots figures; never runs ANN.

Run from any directory: python generate_figures.py
Then compile each .tex twice from this directory with pdflatex.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
RESEARCH = HERE.parents[2]
STRENGTH_RELATIVE = Path("cdsm-review-strengthening-20260922/SUMMARY.json")
NATIVE_RELATIVE = Path("hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json")


def resolve_source(relative: Path) -> Path:
    """Prefer the portable package copy; fall back to the research workspace."""
    bundled = PACKAGE_ROOT / "source-data" / relative
    if bundled.is_file():
        return bundled
    return RESEARCH / relative


STRENGTH = resolve_source(STRENGTH_RELATIVE)
NATIVE = resolve_source(NATIVE_RELATIVE)
GRAPHS = ["Faiss-M32", "Imported-s42", "Imported-s777", "Imported-s1234"]
NATIVE_GRAPHS = ["Faiss-M32", "Imported-s777"]
METHODS = ["C", "F", "MEP_DEV", "FRONTIER"]
NATIVE_METHODS = ["C", "F", "NATIVE", "ABS", "ABS_CAP"]
STYLES = {"C": "cstyle", "F": "fstyle", "MEP_DEV": "mstyle", "FRONTIER": "rstyle", "NATIVE": "nstyle", "ABS": "astyle", "ABS_CAP": "acstyle"}
LEGENDS = {"C": "C", "F": "CDSM-F", "MEP_DEV": "MEP-DEV", "FRONTIER": "FRONTIER", "NATIVE": "Native HNSW", "ABS": r"ABS ($\gamma<0.06$)", "ABS_CAP": "ABS with cap"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name: str, rows: list[dict]) -> None:
    with (HERE / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def begin(height: int) -> list[str]:
    return [
        "% Generated from external CSV files by generate_figures.py.",
        r"\documentclass[border=0pt]{standalone}",
        r"\input{preamble.tex}",
        "% ===== TUNABLES: physical page, panel geometry, font, limits =====",
        r"\def\figwidth{174mm}",
        rf"\def\figheight{{{height}mm}}",
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\fontsize{8}{9.2}\selectfont]",
        r"\path[use as bounding box] (0,0) rectangle (\figwidth,\figheight);",
    ]


def axis(x: float, y: float, width: float, height: float, options: str, paths: list[tuple[str, str]], metric: str) -> list[str]:
    out = [rf"\begin{{axis}}[paperaxis,at={{({x}mm,{y}mm)}},anchor=south west,width={width}mm,height={height}mm,{options}]"]
    for method, name in paths:
        out.append(rf"\addplot[{STYLES[method]}] table[x={'cap_k' if name.startswith('budget-') else 'qps'},y={metric},col sep=comma] {{{name}}};")
    out.append(r"\end{axis}")
    return out


def legend(methods: list[str], y: float, positions: list[float]) -> list[str]:
    out = []
    for method, x in zip(methods, positions):
        style = STYLES[method]
        if method != "ABS_CAP":
            out.append(rf"\draw[{style}] ({x}mm,{y}mm) -- ++(6mm,0) node[pos=.5]{{}};")
        # A plot puts the same marker into the legend; ABS_CAP intentionally has no connecting line.
        out.append(rf"\draw[{style},only marks] plot coordinates {{({x+3}mm,{y}mm)}};")
        out.append(rf"\node[anchor=west,inner sep=0pt] at ({x+7.5}mm,{y}mm) {{{LEGENDS[method]}}};")
    return out


def main() -> None:
    strength = json.loads(STRENGTH.read_text(encoding="utf-8"))
    native = json.loads(NATIVE.read_text(encoding="utf-8"))
    assert strength['all_actual_dc_equal']
    budget_rows = []
    for graph in GRAPHS:
        for method in METHODS:
            records = []
            for cap in [8192,10240,12800]:
                policy = strength['selection'][graph+'-'+str(cap)] if method=='MEP_DEV' else method
                records.append(next(r for r in strength['cells'] if (r['graph'],r['policy'],r['cap'])==(graph,policy,cap)))
            assert len(records) == 3
            rows = []
            for r in records:
                assert r["queries"] == 8000 and r['underfilled']==0 and r['mean_dc']==r['cap']
                row = dict(graph=graph, method=method, actual_policy=r['policy'], cap=r["cap"], cap_k=r["cap"] / 1000, recall_pct=100*r["recall"], severe=r["severe"], queries=r["queries"])
                rows.append(row)
            write_csv(f"budget-{graph}-{method}.csv", rows)
            budget_rows.extend(rows)
    write_csv("t2i-budget-data.csv", budget_rows)

    native_rows = []
    for graph in NATIVE_GRAPHS:
        for method in NATIVE_METHODS:
            records = native["evaluation"][f"{graph}-{method}"]
            if method == "ABS":
                records = [r for r in records if r["parameter"] < 60]
            # Keep every allowed evaluation point, including dominated points.
            rows = []
            for r in sorted(records, key=lambda r: r["qps"]):
                assert r["queries"] == 8000
                rows.append(dict(graph=graph, method=method, parameter=r["parameter"], qps=r["qps"], recall_pct=100*r["recall"], severe=r["severe"], mean_dc=r["mean_dc"], queries=r["queries"]))
            write_csv(f"native-{graph}-{method}.csv", rows)
            native_rows.extend(rows)
    write_csv("native-tradeoff-data.csv", native_rows)

    out = begin(82)
    for row, metric in enumerate(["recall_pct", "severe"]):
        for col, graph in enumerate(GRAPHS):
            x, y = 14 + 40*col, 47.5 if row == 0 else 16
            common = "xmin=7.85,xmax=13.1,xtick={8.192,10.24,12.8},xticklabels={8.192,10.24,12.8},"
            common += "ymin=98,ymax=99.8,ytick={98,98.5,99,99.5}" if row == 0 else "ymin=0,ymax=66,ytick={0,20,40,60}"
            if row == 0:
                common += ",xticklabels=\\empty"
            if col > 0:
                common += ",yticklabels=\\empty"
            else:
                label = r"Mean Recall@10 (\%)" if row == 0 else "Severe / 8,000"
                out.append(rf"\node[rotate=90,inner sep=0pt] at (2.2mm,{y+12.75}mm) {{{label}}};")
            paths = [(method, f"budget-{graph}-{method}.csv") for method in ["C", "FRONTIER", "MEP_DEV", "F"]]
            out.extend(axis(x, y, 35.5, 25.5, common, paths, metric))
            letter = chr(97 + row*4 + col)
            tag_x, tag_anchor = (x+1, "north west") if row == 0 else (x+34.5, "north east")
            out.append(rf"\node[anchor={tag_anchor},inner sep=1pt] at ({tag_x}mm,{y+25}mm) {{({letter})}};")
            if row == 0:
                out.append(rf"\node[anchor=south,inner sep=0pt] at ({x+17.75}mm,75mm) {{{graph}}};")
    out.append(r"\node[inner sep=0pt] at (92mm,7.1mm) {Actual distance evaluations ($\times 10^3$)};")
    out.extend(legend(METHODS, 2, [22, 44, 86, 128]))
    out += [r"\end{tikzpicture}", r"\end{document}"]
    (HERE / "t2i-budget.tex").write_text("\n".join(out)+"\n", encoding="utf-8")

    out = begin(84)
    for row, metric in enumerate(["recall_pct", "severe"]):
        for col, graph in enumerate(NATIVE_GRAPHS):
            x, y = 14 + 81*col, 47.5 if row == 0 else 15
            common = "xmin=100,xmax=1500,xtick={200,600,1000,1400},y label style={at={(axis description cs:-.14,.5)},xshift=2.5mm},"
            common += "ymin=92,ymax=100,ytick={92,94,96,98,100}" if row == 0 else "ymin=0,ymax=205,ytick={0,50,100,150,200}"
            if row == 0:
                common += ",xticklabels=\\empty"
            if col > 0:
                common += ",yticklabels=\\empty"
            else:
                label = r"Mean Recall@10 (\%)" if row == 0 else "Severe / 8,000"
                out.append(rf"\node[rotate=90,inner sep=0pt] at (2.2mm,{y+12.75}mm) {{{label}}};")
            paths = [(method, f"native-{graph}-{method}.csv") for method in ["C", "NATIVE", "ABS", "ABS_CAP", "F"]]
            out.extend(axis(x, y, 74, 25.5, common, paths, metric))
            letter = chr(97 + row*2 + col)
            tag_x, tag_anchor = (x+73, "north east") if row == 0 else (x+1, "north west")
            out.append(rf"\node[anchor={tag_anchor},inner sep=1pt] at ({tag_x}mm,{y+25}mm) {{({letter})}};")
            if row == 0:
                out.append(rf"\node[anchor=south,inner sep=0pt] at ({x+37}mm,74.5mm) {{{graph}}};")
    out.append(r"\node[inner sep=0pt] at (92mm,6.4mm) {Throughput (queries/s)};")
    out.extend(legend(NATIVE_METHODS, 81.5, [10, 29, 67, 95, 140]))
    out += [r"\end{tikzpicture}", r"\end{document}"]
    (HERE / "native-tradeoff.tex").write_text("\n".join(out)+"\n", encoding="utf-8")
    # Preserve the complete range, then use identical data in a readable zoom.
    full = "\n".join(out).replace(r"\def\figheight{84mm}",r"\def\figheight{82mm}").replace("81.5mm","79.5mm")
    (HERE / "native-full.tex").write_text(full+"\n", encoding="utf-8")
    zoom = "\n".join(out).replace(
        "xmin=100,xmax=1500,xtick={200,600,1000,1400}",
        "xmin=190,xmax=340,xtick={200,250,300}").replace(
        "ymin=92,ymax=100,ytick={92,94,96,98,100}",
        "ymin=98.2,ymax=99.8,ytick={98.2,98.6,99,99.4,99.8}").replace(
        "ymin=0,ymax=205,ytick={0,50,100,150,200}",
        "ymin=0,ymax=65,ytick={0,20,40,60}")
    (HERE / "native-tradeoff.tex").write_text(zoom+"\n", encoding="utf-8")

    provenance = {
        # Logical research-relative identities stay stable for either location.
        "sources": [{"path": relative.as_posix(), "sha256": sha(actual)} for relative, actual in [(STRENGTH_RELATIVE, STRENGTH), (NATIVE_RELATIVE, NATIVE)]],
        "t2i_budget": {"graphs": GRAPHS, "methods": METHODS, "caps": [8192, 10240, 12800], "cells": len(budget_rows), "exact_dc_validated": True},
        "native_tradeoff": {"graphs": NATIVE_GRAPHS, "methods": NATIVE_METHODS, "cells": len(native_rows), "abs_filter": "parameter < 60 means gamma < 0.06", "grid": "all available evaluation points after the explicit ABS filter; no Pareto filtering", "abs_cap_encoding": "scatter, because cap and gamma both vary", "qps_provenance": "only the historical native timing wave; never mixed with later controlled experiments"},
        "data_sha256": {p.name: sha(p) for p in sorted(HERE.glob("*.csv"))},
        "editable_state": "TeX/CSV authoritative; no SVG fork",
        "native_zoom": {"x_qps":[190,340],"recall_percent":[98.2,99.8],"severe":[0,65],"full_range_file":"native-full.tex","same_data":True},
        "method_diagram": {"file":"method-feedback.tex","kind":"conceptual execution organization, not measured data"},
    }
    (HERE / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"budget_cells":len(budget_rows),"native_cells":len(native_rows),"source_sha256":provenance["sources"]},indent=2))


if __name__ == "__main__":
    main()
