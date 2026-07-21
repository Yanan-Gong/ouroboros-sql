#!/usr/bin/env python3
"""Thin wrapper: `uv run python scripts/download_data.py` == `ouroboros download-data`."""

from ouroboros_sql.data_setup import main

if __name__ == "__main__":
    main()
