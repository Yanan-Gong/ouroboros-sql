"""Download and arrange the BIRD mini-dev dataset.

Downloads the official mini-dev bundle (questions + SQLite databases), verifies
its checksum, and arranges databases under ``data/databases/<db_id>/<db_id>.sqlite``.
Databases are never committed to git — only the derived golden-set JSON is.

Dataset: https://github.com/bird-bench/mini_dev (CC BY-SA 4.0)
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import settings

MINIDEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
# SHA256 of minidev.zip as downloaded on 2026-07-21. If the upstream file
# changes, verify manually and update.
MINIDEV_SHA256 = "cc48ba16838204e4e214512030cb572eeb5f7bcdd999bae4b9b6ff12ec13b92f"
EXPECTED_SIZE = 800_943_648


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest} (~{EXPECTED_SIZE / 1e6:.0f} MB)")

    def report(blocks: int, block_size: int, total: int) -> None:
        done = blocks * block_size
        if total > 0 and blocks % 200 == 0:
            sys.stdout.write(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def arrange(extract_dir: Path) -> tuple[int, Path]:
    """Move databases and the question file into the canonical layout."""
    db_sources = list(extract_dir.rglob("dev_databases"))
    if not db_sources:
        raise FileNotFoundError(f"No dev_databases directory found under {extract_dir}")
    databases_root = db_sources[0]

    settings.databases_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for db_dir in sorted(databases_root.iterdir()):
        sqlite_file = db_dir / f"{db_dir.name}.sqlite"
        if not sqlite_file.is_file():
            continue
        target_dir = settings.databases_dir / db_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sqlite_file, target_dir / sqlite_file.name)
        moved += 1

    question_files = list(extract_dir.rglob("mini_dev_sqlite.json"))
    if not question_files:
        raise FileNotFoundError("mini_dev_sqlite.json not found in the bundle")
    questions_dest = settings.data_dir / "raw" / "mini_dev_sqlite.json"
    questions_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(question_files[0], questions_dest)

    return moved, questions_dest


def main(*, keep_zip: bool = True, skip_checksum: bool = False) -> None:
    raw_dir = settings.data_dir / "raw"
    zip_path = raw_dir / "minidev.zip"

    if not zip_path.is_file():
        download(MINIDEV_URL, zip_path)

    if not skip_checksum:
        actual = sha256_of(zip_path)
        if MINIDEV_SHA256 != "PLACEHOLDER" and actual != MINIDEV_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for {zip_path}:\n  expected {MINIDEV_SHA256}\n  got {actual}\n"
                "Upstream file may have changed — verify before trusting it."
            )

    extract_dir = raw_dir / "minidev_extracted"
    if not extract_dir.is_dir():
        print(f"Extracting to {extract_dir} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

    moved, questions_dest = arrange(extract_dir)
    print(f"Arranged {moved} databases under {settings.databases_dir}")
    print(f"Questions at {questions_dest}")

    if not keep_zip:
        zip_path.unlink()


if __name__ == "__main__":
    main()
