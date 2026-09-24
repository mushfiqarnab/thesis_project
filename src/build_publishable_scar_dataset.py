"""Build paired, provenance-tracked scar-like artifacts from raw faces and WESAD windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis


MORPHOLOGIES = {
    "linear_mature": {"length": 0.72, "width": 0.85, "tint": (0.82, 0.76, 0.80)},
    "linear_immature": {"length": 0.78, "width": 1.25, "tint": (0.88, 0.70, 0.72)},
    "irregular_atrophic_like": {"length": 0.62, "width": 0.75, "tint": (0.78, 0.78, 0.82)},
    "short_clustered": {"length": 0.45, "width": 0.95, "tint": (0.84, 0.74, 0.76)},
}
GENERATOR_VERSION = "scar-like-renderer-2.0"
MIN_MASK_AREA = 0.0003
MAX_MASK_AREA = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


def render_scar(image: np.ndarray, keypoints: np.ndarray, seed: int, morphology: str):
    height, width = image.shape[:2]
    config = MORPHOLOGIES[morphology]
    left_eye, right_eye, _, left_mouth, _ = keypoints
    rng = random.Random(seed)
    face_scale = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    anchor = (left_eye + left_mouth) / 2.0
    anchor[1] += face_scale * 0.08
    angle = rng.uniform(-0.35, 0.35)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    length = face_scale * rng.uniform(0.55, 0.85) * config["length"]
    points = []
    for fraction in np.linspace(-1.0, 1.0, 7):
        offset = direction * (fraction * length / 2.0)
        offset += normal * rng.uniform(-face_scale * 0.035, face_scale * 0.035)
        points.append(anchor + offset)
    points = np.rint(np.asarray(points)).astype(np.int32)

    hard_mask = np.zeros((height, width), dtype=np.uint8)
    width_px = max(2, int(round(face_scale * rng.uniform(0.035, 0.06) * config["width"])))
    cv2.polylines(hard_mask, [points.reshape(-1, 1, 2)], False, 255, width_px, cv2.LINE_AA)
    if morphology == "short_clustered":
        branch = points.copy()
        branch[:, 1] += int(round(face_scale * 0.08))
        cv2.polylines(hard_mask, [branch.reshape(-1, 1, 2)], False, 255, max(1, width_px - 1), cv2.LINE_AA)

    feather = max(3, int(round(face_scale * 0.035)))
    feather += feather % 2 == 0
    soft_mask = cv2.GaussianBlur(hard_mask, (feather, feather), 0)
    soft = soft_mask.astype(np.float32) / 255.0
    image_float = image.astype(np.float32) / 255.0
    local_color = cv2.GaussianBlur(image_float, (0, 0), max(face_scale * 0.08, 1.0))
    texture = rng.normalvariate(0.0, 0.01) * np.ones_like(soft)
    texture += np.random.default_rng(seed).normal(0.0, 0.006, soft.shape).astype(np.float32)
    scar_tint = local_color * np.asarray(config["tint"], dtype=np.float32)
    result_float = image_float * (1.0 - soft[..., None]) + scar_tint * soft[..., None]
    result_float[..., 2] = np.clip(result_float[..., 2] + 0.025 * soft, 0.0, 1.0)
    result_float += soft[..., None] * texture[..., None]
    result = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
    result[soft_mask == 0] = image[soft_mask == 0]
    parameters = {
        "morphology": morphology,
        "seed": seed,
        "angle_radians": angle,
        "length_pixels": float(length),
        "width_pixels": int(width_px),
        "anchor_xy": [float(anchor[0]), float(anchor[1])],
        "mask_area_fraction": float((soft_mask > 0).mean()),
    }
    return result, soft_mask, parameters


def mask_passes_geometry(mask: np.ndarray, keypoints: np.ndarray) -> bool:
    area = float((mask > 0).mean())
    if not MIN_MASK_AREA <= area <= MAX_MASK_AREA:
        return False
    height, width = mask.shape
    if np.any(mask[0] > 0) or np.any(mask[-1] > 0) or np.any(mask[:, 0] > 0) or np.any(mask[:, -1] > 0):
        return False
    exclusion = np.zeros_like(mask, dtype=np.uint8)
    eye_distance = max(float(np.linalg.norm(keypoints[0] - keypoints[1])), 1.0)
    radius = max(3, int(round(eye_distance * 0.12)))
    for point in keypoints:
        cv2.circle(exclusion, tuple(np.rint(point).astype(int)), radius, 255, -1)
    return not bool(np.any((mask > 0) & (exclusion > 0)))


def assign_balanced_morphologies(face_records: list[dict], face_splits: dict[str, str], seed: int) -> None:
    names = list(MORPHOLOGIES)
    for split in ("train", "val", "test"):
        records = sorted((record for record in face_records if face_splits[record["face_id"]] == split), key=lambda item: item["face_id"])
        offset = stable_seed(seed, split, "morphology") % len(names)
        for index, record in enumerate(records):
            record["morphology"] = names[(offset + index) % len(names)]


def assign_stratified_scar_labels(dataset: pd.DataFrame, rho: float, seed: int) -> pd.DataFrame:
    result = dataset.copy()
    result["scar"] = 0
    for split in ("train", "val", "test"):
        split_indices = result.index[result["split"] == split]
        for threat, probability in ((1, rho), (0, 1.0 - rho)):
            indices = result.index[(result["split"] == split) & (result["threat"] == threat)].tolist()
            count = int(round(len(indices) * probability))
            ordered = sorted(indices, key=lambda index: stable_seed(seed, split, threat, int(index)))
            result.loc[ordered[:count], "scar"] = 1
    result["image_path"] = np.where(result["scar"] == 1, result["scarred_path"], result["clean_path"])
    result["counterfactual_image_path"] = np.where(result["scar"] == 1, result["clean_path"], result["scarred_path"])
    return result


def detect_face(app: FaceAnalysis, image: np.ndarray):
    faces = app.get(image)
    if len(faces) != 1 or getattr(faces[0], "kps", None) is None:
        return None
    face = faces[0]
    age = getattr(face, "age", None)
    if age is None or not 18 <= float(age) <= 65:
        return None
    eye_delta = np.asarray(face.kps[1] - face.kps[0], dtype=np.float32)
    roll_degrees = abs(float(np.degrees(np.arctan2(eye_delta[1], eye_delta[0]))))
    if roll_degrees > 20.0:
        return None
    return face


def allocate_subjects(values: list[str], seed: int) -> dict[str, str]:
    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    train_end = int(round(len(shuffled) * 0.70))
    val_end = train_end + int(round(len(shuffled) * 0.15))
    return {value: "train" if i < train_end else "val" if i < val_end else "test" for i, value in enumerate(shuffled)}


def select_windows(frame: pd.DataFrame, subjects: set[str], count: int, seed: int) -> pd.DataFrame:
    candidates = frame[frame["subject"].astype(str).isin(subjects)]
    if candidates.empty:
        raise RuntimeError("No physiology rows are available for the requested split.")
    per_class = max(1, count // 2)
    selected = []
    for threat in (0, 1):
        group = candidates[candidates["threat"].astype(int) == threat]
        if group.empty:
            raise RuntimeError(f"Physiology split is missing threat class {threat}.")
        selected.append(group.sample(n=min(per_class, len(group)), random_state=seed))
    result = pd.concat(selected)
    remaining = candidates.drop(index=result.index, errors="ignore")
    if len(result) < count and not remaining.empty:
        result = pd.concat([result, remaining.sample(n=min(count - len(result), len(remaining)), random_state=seed)])
    return result.sample(frac=1.0, random_state=seed).head(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", default="data/raw/FFHQ")
    parser.add_argument("--wesad", default="data/csv/wesad_windows.csv")
    parser.add_argument("--out", default="data/publishable_scar")
    parser.add_argument("--max_faces", type=int, default=0)
    parser.add_argument("--windows_per_face", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.85)
    parser.add_argument("--claim_scope", choices=["controlled_benchmark", "expert_validated_realism"], default="controlled_benchmark")
    args = parser.parse_args()
    if not 0.0 <= args.rho <= 1.0:
        raise ValueError("rho must be between 0 and 1")
    if args.claim_scope == "expert_validated_realism":
        raise RuntimeError("Expert ratings and a licensed real-scar reference set are required before this claim scope is allowed.")

    face_paths = sorted(path for path in Path(args.faces).rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.max_faces:
        face_paths = face_paths[: args.max_faces]
    if not face_paths:
        raise FileNotFoundError("No raw face images found.")
    physiology = pd.read_csv(args.wesad).reset_index(drop=True)
    required = {"hrv_rmssd", "gsr_mean", "threat", "subject"}
    missing = sorted(required - set(physiology.columns))
    if missing:
        raise ValueError(f"WESAD CSV missing columns: {missing}")

    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new path for an immutable build.")
    clean_dir, scar_dir, mask_dir = output / "clean", output / "scarred", output / "masks"
    for directory in (clean_dir, scar_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "genderage"], providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces, rejected = [], []
    seen_source_hashes = set()
    for source_path in face_paths:
        source_hash = sha256_file(source_path)
        if source_hash in seen_source_hashes:
            rejected.append({"source_path": str(source_path), "reason": "duplicate_source_content"})
            continue
        seen_source_hashes.add(source_hash)
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        face = None if image is None else detect_face(app, image)
        if face is None:
            rejected.append({"source_path": str(source_path), "reason": "not_exactly_one_adult_face_with_keypoints"})
            continue
        face_id = f"face_{source_hash[:16]}"
        morphology = "unassigned"
        clean_path, scar_path, mask_path = clean_dir / f"{face_id}.png", scar_dir / f"{face_id}.png", mask_dir / f"{face_id}.png"
        keypoints = np.asarray(face.kps, dtype=np.float32)
        morphology_seed = stable_seed(args.seed, face_id)
        all_passed = True
        for morph_name in MORPHOLOGIES:
            _, test_mask, _ = render_scar(image, keypoints, morphology_seed, morph_name)
            if not mask_passes_geometry(test_mask, keypoints):
                all_passed = False
                break
        if not all_passed:
            rejected.append({"source_path": str(source_path), "reason": "mask_geometry_gate"})
            continue
        cv2.imwrite(str(clean_path), image)
        faces.append({
            "face_id": face_id,
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "estimated_age": float(face.age),
            "estimated_gender": int(face.gender),
            "keypoints": keypoints.round(4).tolist(),
            "clean_path": str(clean_path),
            "scarred_path": str(scar_path),
            "mask_path": str(mask_path),
            "morphology": morphology,
            "render_seed": morphology_seed,
            "render_parameters": None,
        })

    if not faces:
        raise RuntimeError("No face passed the landmark gate.")
    face_frame = pd.DataFrame(faces)
    face_splits = allocate_subjects(face_frame["face_id"].tolist(), args.seed)
    assign_balanced_morphologies(faces, face_splits, args.seed)
    face_frame = pd.DataFrame(faces)
    for split in ("train", "val", "test"):
        if sum(face_splits[face_id] == split for face_id in face_splits) < len(MORPHOLOGIES):
            raise RuntimeError(f"{split} split needs at least one face per morphology preset.")
    for face in faces:
        image = cv2.imread(face["source_path"], cv2.IMREAD_COLOR)
        keypoints = np.asarray(face["keypoints"], dtype=np.float32)
        scarred, mask, render_parameters = render_scar(
            image,
            keypoints,
            face["render_seed"],
            face["morphology"],
        )
        if not mask_passes_geometry(mask, keypoints):
            raise RuntimeError(f"Morphology assignment failed mask gate for {face['face_id']}")
        cv2.imwrite(face["scarred_path"], scarred)
        cv2.imwrite(face["mask_path"], mask)
        face["render_parameters"] = render_parameters
        face["clean_sha256"] = sha256_file(Path(face["clean_path"]))
        face["scarred_sha256"] = sha256_file(Path(face["scarred_path"]))
        face["mask_sha256"] = sha256_file(Path(face["mask_path"]))
    phys_subjects = sorted(physiology["subject"].astype(str).unique())
    phys_splits = allocate_subjects(phys_subjects, args.seed)
    rows = []
    for face in faces:
        split = face_splits[face["face_id"]]
        subjects = {subject for subject, value in phys_splits.items() if value == split}
        windows = select_windows(physiology, subjects, args.windows_per_face, stable_seed(args.seed, face["face_id"], split))
        for index, (_, phys) in enumerate(windows.iterrows()):
            threat = int(phys["threat"])
            rows.append({
                "face_id": face["face_id"],
                "physiology_subject": str(phys["subject"]),
                "window_id": int(phys.name),
                "split": split,
                "clean_path": face["clean_path"],
                "scarred_path": face["scarred_path"],
                "mask_path": face["mask_path"],
                "face_source_sha256": face["source_sha256"],
                "clean_sha256": face["clean_sha256"],
                "scarred_sha256": face["scarred_sha256"],
                "mask_sha256": face["mask_sha256"],
                "render_seed": face["render_seed"],
                "keypoints": json.dumps(face["keypoints"]),
                "render_parameters": json.dumps(face["render_parameters"]),
                "morphology": face["morphology"],
                "hrv": float(phys["hrv_rmssd"]),
                "gsr": float(phys["gsr_mean"]),
                "threat": threat,
                "rho_target": args.rho,
                "estimated_age": float(face["estimated_age"]),
                "estimated_gender": int(face["estimated_gender"]),
            })

    dataset = pd.DataFrame(rows)
    for split in ("train", "val", "test"):
        subset = dataset[dataset["split"] == split]
        if subset.empty or subset["threat"].nunique() < 2:
            raise RuntimeError(f"{split} split is missing one or more threat classes.")
    dataset = assign_stratified_scar_labels(dataset, args.rho, args.seed)
    dataset_path, manifest_path = output / "multimodal_publishable.csv", output / "manifest.json"
    dataset.to_csv(dataset_path, index=False)
    realized = {}
    for split, subset in dataset.groupby("split"):
        realized[split] = {
            "rows": int(len(subset)),
            "scar_rate": float(subset["scar"].mean()),
            "threat_rate": float(subset["threat"].mean()),
            "scar_rate_given_threat": float(subset.loc[subset["threat"] == 1, "scar"].mean()),
            "scar_rate_given_no_threat": float(subset.loc[subset["threat"] == 0, "scar"].mean()),
            "scar_threat_correlation": float(np.corrcoef(subset["scar"], subset["threat"])[0, 1]),
        }
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "claim_scope": args.claim_scope,
        "seed": args.seed,
        "rho_target": args.rho,
        "face_count": len(faces),
        "rejected_face_count": len(rejected),
        "physiology_subject_count": len(phys_subjects),
        "rows": len(dataset),
        "face_splits": {split: sorted(value for value, assignment in face_splits.items() if assignment == split) for split in ("train", "val", "test")},
        "physiology_splits": {split: sorted(value for value, assignment in phys_splits.items() if assignment == split) for split in ("train", "val", "test")},
        "rejected_faces": rejected,
        "morphology_counts": face_frame["morphology"].value_counts().to_dict(),
        "realized_by_split": realized,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved dataset: {dataset_path}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Accepted faces: {len(faces)}")
    print(f"Rejected faces: {len(rejected)}")
    print(f"Rows: {len(dataset)}")
    print(dataset.groupby(["split", "scar", "threat"]).size().to_string())


if __name__ == "__main__":
    main()
