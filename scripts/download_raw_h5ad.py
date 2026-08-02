"""Download a target dataset's original source h5ad and cache it under
data/. `--dataset ra` (default) is this repo's primary target (~260MB);
`--dataset sle` fetches the much larger SLE cohort (~12GB) for anyone who
wants to reproduce that run instead.

Census's SOMA obs/var metadata queries are lightweight, but each dataset's
raw counts are read from its own pre-Census-ingestion h5ad rather than
Census's bulk X array -- both because that array can be slower to read at
scale from this network and because several fields this harness needs
(per-cell library size, computed here directly from raw counts; RA's DAS28/
treatment/batch fields; SLE's disease_state/Processing_Cohort) either aren't
in Census's standardized schema or, in library size's case, aren't provided
by either submission at all.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import DATASETS, DatasetConfig

S3_BUCKET_HTTPS = "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com"


def main() -> None:
    import cellxgene_census

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="ra")
    args = parser.parse_args()
    config: DatasetConfig = DATASETS[args.dataset]

    if config.raw_h5ad_path.exists():
        print(f"Already present: {config.raw_h5ad_path} ({config.raw_h5ad_path.stat().st_size} bytes) -- skipping download.")
        return

    locator = cellxgene_census.get_source_h5ad_uri(config.dataset_id, census_version=config.census_version)
    relative_uri = locator["relative_uri"]
    filename = locator["uri"].rsplit("/", 1)[-1]
    url = f"{S3_BUCKET_HTTPS}{relative_uri}{filename}"

    print(f"Downloading {url}")
    config.raw_h5ad_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, config.raw_h5ad_path)

    size = config.raw_h5ad_path.stat().st_size
    assert size > 100_000_000, f"downloaded file suspiciously small: {size} bytes"
    print(f"Saved to {config.raw_h5ad_path} ({size} bytes)")


if __name__ == "__main__":
    main()
