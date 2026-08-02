"""Save/load a `NetworkGenotype` as plain JSON (handoff §14, §23: never pickle
model objects into the search archive -- the genotype is the only thing that
needs to survive to rebuild the architecture in a fresh process).
"""

from __future__ import annotations

from pathlib import Path

from brainmri_nas.search_space.genotype import NetworkGenotype
from brainmri_nas.utils.serialization import dump_json, load_json


def save_genotype(genotype: NetworkGenotype, path: str | Path) -> None:
    dump_json(genotype.to_dict(), path)


def load_genotype(path: str | Path) -> NetworkGenotype:
    return NetworkGenotype.from_dict(load_json(path))
