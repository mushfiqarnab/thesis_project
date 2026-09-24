"""Build a deterministic subject- and content-disjoint dataset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_components(frame: pd.DataFrame) -> tuple[list[list[int]], list[str]]:
    subjects = frame["subject"].astype(str).tolist()
    image_hashes = [sha256_file(Path(str(value))) for value in frame["image_path"]]

    subject_ids = {value: index for index, value in enumerate(sorted(set(subjects)))}
    hash_ids = {
        value: len(subject_ids) + index
        for index, value in enumerate(sorted(set(image_hashes)))
    }
    union_find = UnionFind(len(subject_ids) + len(hash_ids))

    for subject, image_hash in zip(subjects, image_hashes):
        union_find.union(subject_ids[subject], hash_ids[image_hash])

    component_rows: dict[int, list[int]] = {}
    for row_index, subject in enumerate(subjects):
        root = union_find.find(subject_ids[subject])
        component_rows.setdefault(root, []).append(row_index)

    return list(component_rows.values()), image_hashes


def score_assignment(
    frame: pd.DataFrame,
    components: list[list[int]],
    assignment: np.ndarray,
    targets: np.ndarray,
) -> float:
    overall_threat = float(frame["threat"].mean())
    overall_scar = float(frame["scar"].mean())
    score = 0.0

    for split_index in range(3):
        rows = [row for component, split in zip(components, assignment) if split == split_index for row in component]
        if not rows:
            return float("inf")
        subset = frame.iloc[rows]
        row_fraction = len(rows) / len(frame)
        score += 4.0 * abs(row_fraction - targets[split_index])
        score += abs(float(subset["threat"].mean()) - overall_threat)
        score += abs(float(subset["scar"].mean()) - overall_scar)
        if subset["threat"].nunique() < 2 or subset["scar"].nunique() < 2:
            score += 100.0

    return score


def choose_assignment(
    frame: pd.DataFrame,
    components: list[list[int]],
    seed: int,
    attempts: int = 5000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    targets = np.array([0.70, 0.15, 0.15], dtype=float)
    component_order = sorted(range(len(components)), key=lambda index: len(components[index]), reverse=True)
    best_assignment = None
    best_score = float("inf")

    for _ in range(attempts):
        assignment = np.full(len(components), -1, dtype=int)
        for split_index, component_index in enumerate(component_order[:3]):
            assignment[component_index] = split_index
        for component_index in component_order[3:]:
            assignment[component_index] = int(rng.integers(0, 3))

        score = score_assignment(frame, components, assignment, targets)
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None or not np.isfinite(best_score) or best_score >= 100.0:
        raise RuntimeError("Could not construct valid train/val/test partitions.")
    return best_assignment


def validate_manifest(frame: pd.DataFrame, manifest: dict, image_hashes: list[str]) -> None:
    split_names = ["train", "val", "test"]
    index_sets = [set(manifest[f"{name}_idx"]) for name in split_names]
    if set.union(*index_sets) != set(range(len(frame))):
        raise RuntimeError("Manifest does not cover every CSV row exactly once.")
    if any(index_sets[left] & index_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Manifest contains overlapping row indices.")

    subject_sets = [set(frame.iloc[sorted(indices)]["subject"].astype(str)) for indices in index_sets]
    hash_sets = [set(image_hashes[index] for index in indices) for indices in index_sets]
    if any(subject_sets[left] & subject_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Manifest contains subject overlap.")
    if any(hash_sets[left] & hash_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("Manifest contains exact-content overlap.")

    for name, indices in zip(split_names, index_sets):
        subset = frame.iloc[sorted(indices)]
        if subset["threat"].nunique() < 2 or subset["scar"].nunique() < 2:
            raise RuntimeError(f"{name} split does not contain both threat and scar classes.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    frame = pd.read_csv(csv_path)
    required = {"image_path", "subject", "scar", "threat"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    frame["scar"] = pd.to_numeric(frame["scar"], errors="raise").astype(int).clip(0, 1)
    frame["threat"] = pd.to_numeric(frame["threat"], errors="raise").astype(int).clip(0, 1)
    components, image_hashes = build_components(frame)
    assignment = choose_assignment(frame, components, args.seed)

    manifest = {
        "source_csv": str(csv_path),
        "seed": args.seed,
        "train_idx": sorted(row for component, split in zip(components, assignment) if split == 0 for row in component),
        "val_idx": sorted(row for component, split in zip(components, assignment) if split == 1 for row in component),
        "test_idx": sorted(row for component, split in zip(components, assignment) if split == 2 for row in component),
        "train_subjects": sorted(set(frame.iloc[assignment == 0]["subject"].astype(str))),
        "val_subjects": sorted(set(frame.iloc[assignment == 1]["subject"].astype(str))),
        "test_subjects": sorted(set(frame.iloc[assignment == 2]["subject"].astype(str))),
        "content_component_count": len(components),
    }
    validate_manifest(frame, manifest, image_hashes)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Validated strict manifest: {output_path}")
    for name in ["train", "val", "test"]:
        indices = manifest[f"{name}_idx"]
        subset = frame.iloc[indices]
        print(
            f"{name}: rows={len(indices)} subjects={len(manifest[f'{name}_subjects'])} "
            f"threat_rate={subset['threat'].mean():.4f} scar_rate={subset['scar'].mean():.4f}"
        )


if __name__ == "__main__":
    main()