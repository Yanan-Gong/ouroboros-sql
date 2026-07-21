"""Validate the committed golden set: schema, split integrity, no leakage."""

import json
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parents[1] / "data" / "golden"
SPLITS = ["train", "val", "holdout"]

REQUIRED_KEYS = {
    "id",
    "source",
    "db_id",
    "question",
    "evidence",
    "gold_sql",
    "difficulty",
    "order_matters",
    "adversarial",
    "golden_split",
}

pytestmark = pytest.mark.skipif(
    not (GOLDEN_DIR / "train.json").is_file(),
    reason="golden set not built (run scripts/build_golden_set.py)",
)


def load(split: str) -> list[dict]:
    with (GOLDEN_DIR / f"{split}.json").open() as f:
        return json.load(f)


@pytest.mark.parametrize("split", SPLITS)
def test_schema_and_split_field(split: str):
    examples = load(split)
    assert examples, f"{split} is empty"
    for ex in examples:
        assert REQUIRED_KEYS.issubset(ex.keys()), f"missing keys in {ex['id']}"
        assert ex["golden_split"] == split
        if ex["adversarial"]:
            assert ex["gold_sql"] is None
        else:
            assert ex["gold_sql"], f"non-adversarial {ex['id']} lacks gold SQL"
            assert "select" in ex["gold_sql"].lower()


def test_no_leakage_between_splits():
    ids: dict[str, str] = {}
    for split in SPLITS:
        for ex in load(split):
            assert ex["id"] not in ids, f"{ex['id']} in both {ids[ex['id']]} and {split}"
            ids[ex["id"]] = split


def test_adversarial_share_present():
    for split in SPLITS:
        examples = load(split)
        n_adv = sum(1 for ex in examples if ex["adversarial"])
        assert 0 < n_adv <= len(examples) * 0.15


def test_manifest_matches_files():
    with (GOLDEN_DIR / "manifest.json").open() as f:
        manifest = json.load(f)
    for split in SPLITS:
        assert manifest["splits"][split] == len(load(split))
    assert manifest["seed"] == 20260721
