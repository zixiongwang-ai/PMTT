"""Integrity checks for the portable PMTT reproduction assets."""

import csv
import tomllib
from pathlib import Path

from pmtt.assets import destroyed_seed, observation_seed


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_model_registry_has_exact_paper_checkpoints() -> None:
    registry = tomllib.loads((ASSETS / "model_registry.toml").read_text(encoding="utf-8"))
    models = registry["models"]
    assert set(models) == {"S", "B", "L", "H+"}
    for record in models.values():
        assert record["filename"].endswith(".pth")
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64
        int(record["sha256"], 16)


def test_nyuv2_public_ids_and_seed_lookup() -> None:
    manifest = _rows(ASSETS / "manifests" / "nyuv2.csv")
    assert manifest
    assert all(row["id"] == row["image"] for row in manifest)
    assert all(row["id"].startswith("nyu/") for row in manifest)
    assert all("dataset/prepared" not in row["id"] for row in manifest)

    training = next(row for row in manifest if row["split"] == "train")
    calibration = next(row for row in manifest if row["calibration_split"])
    evaluation = next(row for row in manifest if row["split"] == "test")

    assert isinstance(observation_seed(("nyu", training["id"], "axis", 0)), int)
    assert isinstance(
        observation_seed(("nyu", calibration["id"], "calibration", 0)),
        int,
    )
    assert isinstance(observation_seed(("nyu", evaluation["id"], "dense", 0)), int)
    assert isinstance(destroyed_seed(("nyu", evaluation["id"], "B"), 0), int)
    assert isinstance(destroyed_seed(("nyu", evaluation["id"], "H+"), 0), int)


def test_nyuv2_fitting_identities_use_public_paths() -> None:
    fitting = _rows(ASSETS / "manifests" / "fitting" / "nyuv2_fit_ids.csv")
    donors = _rows(ASSETS / "manifests" / "fitting" / "dense_donors.csv")
    nyu_donors = [row for row in donors if row["task"] == "nyu"]

    assert fitting
    assert nyu_donors
    assert all(row["id"].startswith("nyu/") for row in fitting)
    assert all(row["image_id"].startswith("nyu/") for row in nyu_donors)
    assert all("dataset/prepared" not in row["id"] for row in fitting)
    assert all("dataset/prepared" not in row["image_id"] for row in nyu_donors)
