#!/usr/bin/env python3
"""Thin wrapper: build data/golden/{train,val,holdout}.json from BIRD mini-dev."""

from ouroboros_sql.golden_builder import build

if __name__ == "__main__":
    build()
