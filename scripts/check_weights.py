"""Verify external DINOv3 checkpoint identities without loading the models."""

from __future__ import annotations

import argparse
import hashlib
import tomllib
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "paper_reproduction.toml"
MODEL_REGISTRY = REPOSITORY / "assets" / "model_registry.toml"
SCALE_KEYS = {
    "S": "weights_s",
    "B": "weights_b",
    "L": "weights_l",
    "H+": "weights_hplus",
}


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_config_path(config_path: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (config_path.resolve().parent / path).resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Check DINOv3 checkpoint filenames, sizes, and SHA-256 digests."
    )
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--weights-s", type=Path)
    result.add_argument("--weights-b", type=Path)
    result.add_argument("--weights-l", type=Path)
    result.add_argument("--weights-hplus", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    registry = tomllib.loads(MODEL_REGISTRY.read_text(encoding="utf-8"))["models"]
    configured_paths = config.get("paths", {})

    failures = []
    for scale, config_key in SCALE_KEYS.items():
        override = getattr(args, config_key)
        path = override or resolve_config_path(
            args.config,
            configured_paths.get(config_key, ""),
        )
        if path is None:
            failures.append(f"{scale}: checkpoint path is not configured")
            continue

        expected = registry[scale]
        if not path.is_file():
            failures.append(f"{scale}: file not found: {path}")
            continue

        size = path.stat().st_size
        digest = sha256(path)
        checks = {
            "filename": path.name == expected["filename"],
            "bytes": size == expected["bytes"],
            "sha256": digest == expected["sha256"],
        }
        status = "PASS" if all(checks.values()) else "FAIL"
        print(
            f"{scale}: {status} | {path.name} | "
            f"{size} bytes | sha256={digest}"
        )
        if status == "FAIL":
            failed = ", ".join(name for name, ok in checks.items() if not ok)
            failures.append(f"{scale}: mismatched {failed}")

    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
