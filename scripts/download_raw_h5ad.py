"""Download the SLE dataset's original source h5ad and cache it under data/.

Census's SOMA obs/var metadata queries are lightweight, but this dataset's
raw counts (1.26M cells) are read from its own pre-Census-ingestion h5ad
rather than Census's bulk X array -- both because that array can be slow to
read at scale from this network (see TRACE's equivalent script) and because
several fields this harness needs (per-cell library size, computed here
directly from raw counts; disease_state; Processing_Cohort) either aren't
in Census's standardized schema or, in library size's case, aren't provided
by this submission at all.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import CENSUS_VERSION, DATASET_ID, RAW_H5AD_PATH

S3_BUCKET_HTTPS = "https://cellxgene-census-public-us-west-2.s3.us-west-2.amazonaws.com"


def main() -> None:
    import cellxgene_census

    if RAW_H5AD_PATH.exists():
        print(f"Already present: {RAW_H5AD_PATH} ({RAW_H5AD_PATH.stat().st_size} bytes) -- skipping download.")
        return

    locator = cellxgene_census.get_source_h5ad_uri(DATASET_ID, census_version=CENSUS_VERSION)
    relative_uri = locator["relative_uri"]
    filename = locator["uri"].rsplit("/", 1)[-1]
    url = f"{S3_BUCKET_HTTPS}{relative_uri}{filename}"

    print(f"Downloading {url} (~12GB, may take a while)")
    RAW_H5AD_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, RAW_H5AD_PATH)

    size = RAW_H5AD_PATH.stat().st_size
    assert size > 1_000_000_000, f"downloaded file suspiciously small: {size} bytes"
    print(f"Saved to {RAW_H5AD_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
