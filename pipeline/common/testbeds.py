#!/usr/bin/env python3
"""One population seed of a robustness testbed: from the extracted ``runs_testbed_<name>/`` directory (the Zenodo archives)
or, failing that, extracted on the fly from a ``runs_testbed_<name>.tar.gz`` next to the package (replay/per_round.py testbeds,
replay/logspace_grids.py and common/pairings.py read one seed at a time)."""
from __future__ import annotations

import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMBER_DIR = "runs_testbed_{name}"  # top-level directory inside each archive


def archive_path(name: str) -> Path:
    """``runs_testbed_<name>.tar.gz`` next to the package (or inside it)."""
    for p in (ROOT.parent / f"runs_testbed_{name}.tar.gz", ROOT / f"runs_testbed_{name}.tar.gz"):
        if p.exists():
            return p
    raise FileNotFoundError(f"runs_testbed_{name}.tar.gz not found next to {ROOT}")


def extract_seed(name: str, seed: int, work: Path) -> Path:
    """Return ``run_popseed_<seed>.checkpoints`` of testbed ``name``: the extracted directory if present, else extract it into ``work``."""
    for p in (ROOT.parent / f"runs_testbed_{name}" / f"run_popseed_{seed}.checkpoints", ROOT / f"runs_testbed_{name}" / f"run_popseed_{seed}.checkpoints"):
        if (p / "metadata.json").exists():
            return p
    member_prefix = f"{MEMBER_DIR.format(name=name)}/run_popseed_{seed}.checkpoints"
    with tarfile.open(archive_path(name), "r:gz") as tar:
        def members():
            for m in tar:
                if m.name.startswith(member_prefix + "/") or m.name == member_prefix:
                    yield m
        tar.extractall(work, members=members(), filter="data")
    return work / member_prefix
