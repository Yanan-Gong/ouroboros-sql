"""Central configuration. Everything model- or path-related is overridable via env vars."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OUROBOROS_", env_file=".env", extra="ignore")

    # Worker agents run on a small model; judging and optimizing need a stronger one.
    agent_model: str = "gpt-5-mini"
    judge_model: str = "gpt-5"
    optimizer_model: str = "gpt-5"

    data_dir: Path = REPO_ROOT / "data"
    runs_dir: Path = REPO_ROOT / "runs"

    max_turns: int = 25
    sql_timeout_seconds: float = 15.0
    sql_row_limit: int = 500

    @property
    def databases_dir(self) -> Path:
        return self.data_dir / "databases"

    @property
    def golden_dir(self) -> Path:
        return self.data_dir / "golden"


settings = Settings()
