"""Discovery of downloaded SQLite databases: db_id -> file path."""

from pathlib import Path


class DatabaseNotFoundError(KeyError):
    def __init__(self, db_id: str, available: list[str]):
        self.db_id = db_id
        self.available = available
        hint = ", ".join(sorted(available)[:10]) or "none — run `ouroboros download-data`"
        super().__init__(f"Unknown database {db_id!r}. Available: {hint}")


class Catalog:
    """Maps db_id to a .sqlite file under a root directory.

    Layout matches BIRD/Spider releases: <root>/<db_id>/<db_id>.sqlite,
    with a flat <root>/<db_id>.sqlite fallback for hand-made test databases.
    """

    def __init__(self, root: Path):
        self.root = root

    def db_ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        found = []
        for child in self.root.iterdir():
            if child.is_dir() and (child / f"{child.name}.sqlite").is_file():
                found.append(child.name)
            elif child.suffix == ".sqlite" and child.is_file():
                found.append(child.stem)
        return sorted(set(found))

    def path_for(self, db_id: str) -> Path:
        nested = self.root / db_id / f"{db_id}.sqlite"
        if nested.is_file():
            return nested
        flat = self.root / f"{db_id}.sqlite"
        if flat.is_file():
            return flat
        raise DatabaseNotFoundError(db_id, self.db_ids())
