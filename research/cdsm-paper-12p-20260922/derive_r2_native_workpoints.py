"""Describe existing native/F operating points; never run ANN or select per query."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source-data/hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json"
GRAPHS = ["Faiss-M32", "Imported-s42", "Imported-s777", "Imported-s1234"]


def metrics(row):
    return {key: row[key] for key in (
        "parameter", "queries", "recall", "severe", "zero", "qps",
        "mean_ms", "p99_ms", "mean_dc"
    ) if key in row}


def compare(reference, candidates):
    eligible = [row for row in candidates
                if row["recall"] + 1e-12 >= reference["recall"]
                and row["severe"] <= reference["severe"]]
    if not eligible:
        return {"reference": metrics(reference), "candidate": None,
                "status": "No qualifying point in the measured grid; not an impossibility result."}
    best = max(eligible, key=lambda row: row["qps"])
    return {
        "reference": metrics(reference), "candidate": metrics(best),
        "qps_change_percent": 100.0 * (best["qps"] / reference["qps"] - 1.0),
        "recall_change_pp": 100.0 * (best["recall"] - reference["recall"]),
        "severe_change": best["severe"] - reference["severe"],
    }


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    output = {
        "scope": "Retrospective whole-grid descriptive comparison; no new ANN runs, new tests, interpolation, or equal-work/equal-time claim.",
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "selection_rule": "For every reference grid point, choose the fastest measured candidate with mean recall >= reference and severe count <= reference; retain all references including unmatched ones.",
        "timing": "All points from the same earlier native wave; qps is its recorded median round-level throughput. p99 is a quantile of per-query median latency, not raw service latency.",
        "evaluation_exposure": "Historical evaluation queries; these point comparisons are not independent confirmation or development-selected deployment choices.",
        "graphs": {},
    }
    for graph in GRAPHS:
        native = data["evaluation"][graph + "-NATIVE"]
        scouts = data["evaluation"][graph + "-F"]
        forward = [compare(row, scouts) for row in native]
        backward = [compare(row, native) for row in scouts]
        output["graphs"][graph] = {
            "native_references_to_F": forward,
            "F_references_to_native": backward,
        }
    destination = ROOT / "R2-NATIVE-WORKPOINTS.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination.name)
    for graph, rows in output["graphs"].items():
        for row in rows["native_references_to_F"]:
            if row.get("qps_change_percent", -100) > 1.5:
                print(graph, "native", row["reference"]["parameter"],
                      "F", row["candidate"]["parameter"],
                      "QPS gain %", round(row["qps_change_percent"], 5))


if __name__ == "__main__":
    main()
