"""Verify published paper/evidence bytes without running ANN or needing datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "research/cdsm-paper-12p-20260922"
ARCHIVE_SHA256 = "4b0eebb60296b18f006f98134f423d8606dac48b34ba9b17bc31cbea1bc96ee3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative.replace("\\", "/"))
    require(not path.is_absolute() and ".." not in path.parts and ":" not in relative,
            f"Unsafe manifest path: {relative}")
    return root.joinpath(*path.parts)


def verify_records(root: Path, records: list[dict]) -> int:
    for record in records:
        payload = safe_path(root, record["path"]).read_bytes()
        require(sha256(payload) == record["sha256"], f"Hash mismatch: {record['path']}")
        if "bytes" in record:
            require(len(payload) == record["bytes"], f"Size mismatch: {record['path']}")
    return len(records)


def main() -> None:
    package = json.loads((PAPER / "PACKAGE-MANIFEST.json").read_text(encoding="utf-8"))
    evidence = json.loads((PAPER / "SOURCE-DATA.json").read_text(encoding="utf-8"))
    qa = json.loads((PAPER / "QA.json").read_text(encoding="utf-8"))
    package_count = verify_records(PAPER, package["files"])
    evidence_count = verify_records(PAPER / "source-data", evidence["files"])

    pdf = safe_path(PAPER, qa["file"]).read_bytes()
    require(pdf.startswith(b"%PDF-"), "Delivered file is not a PDF")
    require(sha256(pdf) == qa["sha256"] == package["delivered_pdf_sha256"],
            "Delivered PDF differs from the reviewed edition")
    require(qa["passed"] and qa["pages"] == 12, "Expected a reviewed 12-page edition")

    archive = PAPER / "output/cdsm-paper-source.zip"
    require(sha256(archive.read_bytes()) == ARCHIVE_SHA256, "Source archive hash mismatch")
    with zipfile.ZipFile(archive) as bundle:
        require(bundle.testzip() is None, "Corrupt source archive")
        for member in bundle.infolist():
            require(not member.is_dir(), f"Unexpected directory entry: {member.filename}")
            require(safe_path(PAPER, member.filename).read_bytes() == bundle.read(member),
                    f"Checkout differs from frozen archive: {member.filename}")
        archive_count = len(bundle.infolist())

    audit = qa["latest_server_audit"]
    require(audit["all_actual_dc_equal"], "Server audit did not verify actual distance equality")
    print(json.dumps({
        "passed": True,
        "package_files_verified": package_count,
        "evidence_files_verified": evidence_count,
        "archive_entries_verified": archive_count,
        "reviewed_pdf_pages": qa["pages"],
        "pdf_sha256": qa["sha256"],
        "server_evaluation_records": audit["evaluation_records"],
        "server_timing_records": audit["timing_records"],
        "scope": "Published file integrity; does not rerun ANN, timing, or visual review."
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
