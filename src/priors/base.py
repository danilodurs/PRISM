"""The `Prior` interface: everything downstream of graph construction (mask
building, structural controls, encoder, CV, significance testing, permutation
null) takes a `Prior` and never branches on which one it is. A prior module's
entire surface is producing one of these from a shared gene universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Prior:
    name: str          # e.g. "tf_target", "epigenomic" -- used in condition names and file naming
    signed: bool        # gates whether the sign-flip control is run at all
    edges: pd.DataFrame  # columns: source, target, weight -- already filtered to the shared gene universe
