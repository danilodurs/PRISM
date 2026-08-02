# Graph construction summary (ra, prior-mode=combined)

Shared gene universe: 14931 genes

## tf_target (signed=True)
Real graph: 28003 edges, 268 hidden units
Controls generated per seed (n_seeds=5): C1 degree-preserving, C2 fully random
C3 sign-flipped (single version, deterministic given real graph)

## epigenomic (signed=False)
Real graph: 30278 edges, 362 hidden units
Controls generated per seed (n_seeds=5): C1 degree-preserving, C2 fully random
C3 sign-flipped: not applicable (unsigned prior)

## combined (tf_target::* + epigenomic::*)
Real joint mask: 630 hidden units (268 tf_target + 362 epigenomic)
Controls generated per seed (n_seeds=5): joint C1 degree-preserving (per source prior), joint C2 fully random (per source prior), plus exploratory only_tf_real/only_epi_real ablations (uncorrected, see MEMO_RA_COMBINED.md)
Joint C3 sign-flipped (TF-target block only, single deterministic version)