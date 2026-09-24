"""Audit candidate pristine face sources without modifying raw data.

This stage intentionally does not generate scars. It produces a row-level audit
manifest and a summary that must be reviewed before any counterfactual pipeline
is allowed to consume the candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


LOGGER = logging.getLogger("pristine_source_audit")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LEGACY_ROOT_NAMES = {
    "faces_clean",
    "faces_synth_scar",
    "scar_masks",
    "mm10k",
    "mm10k_unbiased",
    "src_faces",
}
DEFAULT_MIN_WIDTH = 224
DEFAULT_MIN_HEIGHT = 224
DEFAULT_MIN_FACE_PX = 64


KNOWN_METADATA_FILES = {
    "CelebA": [
        "list_attr_celeba.txt",
        "identity_CelebA.txt",
        "list_eval_partition.txt",
        "list_bbox_celeba.txt",
        "list_landmarks_align_celeba.txt",
    ],
    "FFHQ": [
        "ffhq-dataset-v1.json",
        "ffhq-dataset-v2.json",
    ],
}


@dataclass(frozen=True)
class AuditConfig:
    min_width: int = DEFAULT_MIN_WIDTH
    min_height: int = DEFAULT_MIN_HEIGHT
    min_face_px: int = DEFAULT_MIN_FACE_PX
    detect_faces: bool = True
    max_images: Optional[int] = None
    run_id: str = ""


@dataclass
class AuditRow:
    source_id: str
    source_domain: str
    relative_path: str
    absolute_path: str
    file_extension: str
    file_size_bytes: int
    file_sha256: str
    pixel_sha256: str
    perceptual_hash: str
    width: Optional[int]
    height: Optional[int]
    channels: Optional[int]
    face_count: Optional[int]
    largest_face_width: Optional[int]
    largest_face_height: Optional[int]
    decode_status: str
    license_status: str
    provenance_status: str
    status: str
    exclusion_reasons: str


class ManifestWriter:
    """Write deterministic tabular output and a content hash for audit artifacts."""

    FIELDNAMES = list(AuditRow.__dataclass_fields__.keys())

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = output_dir / f"source_audit_{run_id}.csv"
        self.summary_path = output_dir / f"source_audit_{run_id}.json"

    def write(self, rows: list[AuditRow], summary: dict[str, Any], config: AuditConfig) -> None:
        with self.manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

        manifest_sha256 = sha256_file(self.manifest_path)
        summary = dict(summary)
        summary["manifest_sha256"] = manifest_sha256
        summary["manifest_path"] = str(self.manifest_path)
        summary["config"] = asdict(config)
        summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self.summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def perceptual_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] >= resized[:, :-1]
    return "".join("1" if value else "0" for value in bits.ravel())


def domain_for_root(root: Path) -> str:
    normalized = root.name.lower()
    if normalized == "ffhq":
        return "FFHQ"
    if normalized == "img_align_celeba":
        return "CelebA"
    return root.name


def discover_metadata(domain: str, root: Path) -> list[dict[str, str]]:
    """Scan root and its immediate parents for known metadata files."""
    found = []
    if domain not in KNOWN_METADATA_FILES:
        return found
    
    search_dirs = [root, root.parent, root.parent.parent]
    search_dirs = list(dict.fromkeys(d.resolve() for d in search_dirs if d.exists()))
    
    for candidate_dir in search_dirs:
        for file_name in KNOWN_METADATA_FILES[domain]:
            candidate_path = candidate_dir / file_name
            if candidate_path.is_file():
                found.append({
                    "file_name": file_name,
                    "absolute_path": str(candidate_path),
                    "sha256": sha256_file(candidate_path)
                })
    
    unique_found = {m["absolute_path"]: m for m in found}
    return list(unique_found.values())


def reject_unsafe_root(root: Path, project_root: Path) -> None:
    resolved = root.resolve()
    if not resolved.exists() or not resolved.is_dir():
        raise FileNotFoundError(f"Source root does not exist or is not a directory: {resolved}")
    if project_root.resolve() not in resolved.parents:
        raise ValueError(f"Source root must be inside the project: {resolved}")
    if any(part.lower() in LEGACY_ROOT_NAMES for part in resolved.parts):
        raise ValueError(f"Legacy/downstream source root is forbidden: {resolved}")


def image_paths(root: Path, max_images: Optional[int]) -> Iterable[Path]:
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def face_measurements(image: np.ndarray, detector: Any) -> tuple[int, Optional[int], Optional[int]]:
    if detector is None:
        return (None, None, None)  # type: ignore[return-value]
    # detector.get expects BGR
    faces = detector.get(image)
    if len(faces) == 0:
        return 0, None, None
    
    def box_area(box):
        return max(0, int(box[2] - box[0])) * max(0, int(box[3] - box[1]))
        
    largest = max(faces, key=lambda f: box_area(f.bbox))
    w = max(0, int(largest.bbox[2] - largest.bbox[0]))
    h = max(0, int(largest.bbox[3] - largest.bbox[1]))
    return len(faces), w, h


def license_and_provenance(domain: str) -> tuple[str, str]:
    if domain == "FFHQ":
        return "verify_per_image_metadata", "dataset_metadata_required"
    if domain == "CelebA":
        return "restricted_noncommercial_pending_review", "dataset_metadata_required"
    return "unknown_pending_review", "unknown_pending_review"


def audit_image(
    path: Path,
    root: Path,
    domain: str,
    config: AuditConfig,
    detector: Any,
    seen_file_hashes: dict[str, str],
    seen_pixel_hashes: dict[str, str],
    seen_perceptual_hashes: dict[str, list[str]],
    has_metadata: bool = False,
) -> AuditRow:
    relative_path = path.relative_to(root).as_posix()
    source_id = f"{domain.lower()}:{relative_path}"
    reasons: list[str] = []
    file_hash = ""
    pixel_hash = ""
    perceptual = ""
    width = height = channels = None
    face_count = largest_width = largest_height = None
    decode_status = "decode_failed"

    try:
        file_hash = sha256_file(path)
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            reasons.append("decode_failed")
        else:
            decode_status = "ok"
            height, width = image.shape[:2]
            channels = 3
            pixel_hash = pixel_sha256(image)
            perceptual = perceptual_hash(image)
            if width < config.min_width or height < config.min_height:
                reasons.append("resolution_below_threshold")
            if config.detect_faces:
                face_count, largest_width, largest_height = face_measurements(image, detector)
                if face_count == 0:
                    reasons.append("no_face_detected")
                elif face_count != 1:
                    reasons.append("face_count_not_one")
                elif (largest_width or 0) < config.min_face_px or (largest_height or 0) < config.min_face_px:
                    reasons.append("face_below_size_threshold")
    except (OSError, ValueError, cv2.error) as exc:
        LOGGER.debug("Failed to audit %s: %s", path, exc)
        reasons.append("read_error")

    if file_hash and file_hash in seen_file_hashes:
        reasons.append("exact_file_duplicate")
    elif file_hash:
        seen_file_hashes[file_hash] = source_id
    if pixel_hash and pixel_hash in seen_pixel_hashes:
        reasons.append("exact_pixel_duplicate")
    elif pixel_hash:
        seen_pixel_hashes[pixel_hash] = source_id
    if perceptual:
        for prior_id in seen_perceptual_hashes[perceptual]:
            if prior_id != source_id:
                reasons.append("perceptual_duplicate_candidate")
                break
        seen_perceptual_hashes[perceptual].append(source_id)

    license_status, provenance_status = license_and_provenance(domain)
    reasons.append("license_review_required")
    
    if has_metadata and provenance_status == "dataset_metadata_required":
        provenance_status = "dataset_metadata_linked"
    else:
        reasons.append("provenance_review_required")
        
    if "decode_failed" in reasons or "read_error" in reasons:
        status = "rejected"
    elif any(reason in reasons for reason in ("resolution_below_threshold", "no_face_detected", "face_count_not_one", "face_below_size_threshold")):
        status = "rejected"
    else:
        status = "pending_review"

    return AuditRow(
        source_id=source_id,
        source_domain=domain,
        relative_path=relative_path,
        absolute_path=str(path.resolve()),
        file_extension=path.suffix.lower(),
        file_size_bytes=path.stat().st_size,
        file_sha256=file_hash,
        pixel_sha256=pixel_hash,
        perceptual_hash=perceptual,
        width=width,
        height=height,
        channels=channels,
        face_count=face_count,
        largest_face_width=largest_width,
        largest_face_height=largest_height,
        decode_status=decode_status,
        license_status=license_status,
        provenance_status=provenance_status,
        status=status,
        exclusion_reasons=";".join(sorted(set(reasons))),
    )


def build_summary(
    rows: list[AuditRow], 
    roots: list[Path], 
    metadata_records: dict[str, list[dict[str, str]]]
) -> dict[str, Any]:
    by_domain: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_domain[row.source_domain][row.status] += 1
    return {
        "source_roots": [str(root.resolve()) for root in roots],
        "metadata_files": metadata_records,
        "rows_written": len(rows),
        "status_counts": dict(Counter(row.status for row in rows)),
        "domain_status_counts": {domain: dict(counts) for domain, counts in sorted(by_domain.items())},
        "decode_counts": dict(Counter(row.decode_status for row in rows)),
        "reason_counts": dict(
            Counter(reason for row in rows for reason in row.exclusion_reasons.split(";") if reason)
        ),
        "warning": "pending_review rows are not approved pristine sources and must not enter generation",
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        required=True,
        help="Allowlisted pristine candidate root; repeat for FFHQ and CelebA.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/source_audits"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--min-width", type=int, default=DEFAULT_MIN_WIDTH)
    parser.add_argument("--min-height", type=int, default=DEFAULT_MIN_HEIGHT)
    parser.add_argument("--min-face-px", type=int, default=DEFAULT_MIN_FACE_PX)
    parser.add_argument("--no-face-detection", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    project_root = args.project_root.resolve()
    roots = [(root if root.is_absolute() else project_root / root).resolve() for root in args.source_root]
    for root in roots:
        reject_unsafe_root(root, project_root)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    config = AuditConfig(
        min_width=args.min_width,
        min_height=args.min_height,
        min_face_px=args.min_face_px,
        detect_faces=not args.no_face_detection,
        max_images=args.max_images,
        run_id=run_id,
    )
    detector = None
    if config.detect_faces:
        try:
            from insightface.app import FaceAnalysis
            detector = FaceAnalysis(name='buffalo_l')
            detector.prepare(ctx_id=0, det_size=(640, 640))
        except ImportError:
            raise RuntimeError("insightface not installed. pip install insightface onnxruntime-gpu")

    rows: list[AuditRow] = []
    seen_file_hashes: dict[str, str] = {}
    seen_pixel_hashes: dict[str, str] = {}
    seen_perceptual_hashes: dict[str, list[str]] = defaultdict(list)
    metadata_records: dict[str, list[dict[str, str]]] = {}
    
    for root in roots:
        domain = domain_for_root(root)
        domain_meta = discover_metadata(domain, root)
        
        if domain not in metadata_records:
            metadata_records[domain] = []
        for m in domain_meta:
            if m not in metadata_records[domain]:
                metadata_records[domain].append(m)
                
        has_metadata = len(metadata_records[domain]) > 0
        
        paths = list(image_paths(root, config.max_images))
        LOGGER.info("Auditing %s: %d image files (metadata found: %s)", domain, len(paths), has_metadata)
        for index, path in enumerate(paths, start=1):
            rows.append(
                audit_image(
                    path,
                    root,
                    domain,
                    config,
                    detector,
                    seen_file_hashes,
                    seen_pixel_hashes,
                    seen_perceptual_hashes,
                    has_metadata=has_metadata,
                )
            )
            if index % 1000 == 0:
                LOGGER.info("Audited %d/%d files in %s", index, len(paths), domain)

    writer = ManifestWriter(output_dir, run_id)
    writer.write(rows, build_summary(rows, roots, metadata_records), config)
    LOGGER.info("Wrote audit manifest: %s", writer.manifest_path)
    LOGGER.info("Wrote audit summary: %s", writer.summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
