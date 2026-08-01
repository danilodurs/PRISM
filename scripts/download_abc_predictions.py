"""Download the genome-wide ABC (Activity-by-Contact) enhancer-gene
predictions (Nasser et al. 2021, Nature) and cache it under data/. This is
the raw resource src/priors/epigenomic.py filters down to PBMC biosamples.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.priors.epigenomic import ABC_RAW_PATH

URL = "https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz"


def main() -> None:
    if ABC_RAW_PATH.exists():
        print(f"Already present: {ABC_RAW_PATH} ({ABC_RAW_PATH.stat().st_size} bytes) -- skipping download.")
        return

    print(f"Downloading {URL} (~340MB)")
    ABC_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, ABC_RAW_PATH)

    size = ABC_RAW_PATH.stat().st_size
    assert size > 100_000_000, f"downloaded file suspiciously small: {size} bytes"
    print(f"Saved to {ABC_RAW_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
