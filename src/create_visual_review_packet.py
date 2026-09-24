"""Create a blinded, stratified visual-review packet from a validated pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image


RATING_COLUMNS = [
    "anatomical_plausibility",
    "boundary_naturalness",
    "pigmentation_or_vascular_naturalness",
    "texture_and_maturation_consistency",
    "unintended_change_free",
    "overall_visual_plausibility",
]


def stable_id(seed: int, face_id: str) -> str:
    digest = hashlib.sha256(f"{seed}|{face_id}".encode("utf-8")).hexdigest()
    return f"review_{digest[:12]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/publishable_scar_strict_pilot2/multimodal_publishable.csv")
    parser.add_argument("--out", default="outputs/visual_review_packet")
    parser.add_argument("--max_per_morphology", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    frame = pd.read_csv(dataset_path)
    required = {"face_id", "image_path", "scar", "morphology", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    scarred = frame[frame["scar"] == 1].drop_duplicates("face_id").copy()
    selected = []
    for morphology, group in scarred.groupby("morphology", sort=True):
        if len(group) < args.max_per_morphology:
            raise RuntimeError(f"Morphology {morphology} has only {len(group)} eligible faces.")
        selected.append(group.sample(n=args.max_per_morphology, random_state=args.seed))
    selected_frame = pd.concat(selected, ignore_index=True)
    selected_frame = selected_frame.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    output = Path(args.out)
    images_dir = output / "images"
    if output.exists():
        shutil.rmtree(output)
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, row in selected_frame.iterrows():
        review_id = stable_id(args.seed, str(row["face_id"]))
        destination = images_dir / f"{review_id}.png"
        with Image.open(row["image_path"]).convert("RGB") as image:
            image.save(destination, format="PNG")
        rows.append({
            "review_id": review_id,
            "display_order": index + 1,
            "image_file": str(destination),
        })

    review_frame = pd.DataFrame(rows)
    review_frame.to_csv(output / "review_index.csv", index=False)
    rating_frame = review_frame.copy()
    rating_frame["rater_id"] = ""
    rating_frame["rater_expertise"] = ""
    for column in RATING_COLUMNS:
        rating_frame[column] = ""
    rating_frame["comments"] = ""
    rating_frame = rating_frame[["rater_id", "rater_expertise", "review_id", "display_order", *RATING_COLUMNS, "comments"]]
    rating_frame.to_csv(output / "ratings_template.csv", index=False)

    answer_key = selected_frame[["face_id", "split", "morphology", "image_path", "face_source_sha256"]].copy()
    answer_key.insert(0, "review_id", [row["review_id"] for row in rows])
    answer_key.to_csv(output / "answer_key_private.csv", index=False)
    protocol = {
        "scale": "1-5 ordinal; 1=very poor, 5=very plausible",
        "blinding": "Raters receive only images/ratings_template.csv. Do not provide answer_key_private.csv.",
        "dimensions": RATING_COLUMNS,
        "interpretation": "This review measures visual plausibility, not clinical diagnosis or biological validity.",
        "seed": args.seed,
        "source_dataset": str(dataset_path),
        "image_count": len(review_frame),
    }
    (output / "review_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved review packet: {output}")
    print(f"Images: {len(review_frame)}")
    print(f"Morphologies: {sorted(selected_frame['morphology'].unique().tolist())}")
    print("Private answer key: answer_key_private.csv")


if __name__ == "__main__":
    main()