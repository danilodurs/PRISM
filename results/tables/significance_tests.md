# Paired significance tests -- donor-level disease-status AUROC


## tf_target (Bonferroni alpha = 0.05 / 4)
    prior                                               comparison  n_paired_splits  mean_a   mean_b  mean_diff  diff_std  wilcoxon_stat  wilcoxon_p  ttest_stat  ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
tf_target                          tf_target: real vs baseline_pca               30 0.52963 0.485185   0.044444  0.211625           89.5    0.228525    1.150302 0.259417            0.0125                 False              False    0.505556    0.544444
tf_target tf_target: real vs C1 degree-preserving (pooled 5 seeds)               30 0.52963 0.488889   0.040741  0.189740          174.0    0.346524    1.176065 0.249133            0.0125                 False              False    0.505556    0.491111
tf_target      tf_target: real vs C2 fully random (pooled 5 seeds)               30 0.52963 0.451852   0.077778  0.195805          132.5    0.039666    2.175663 0.037873            0.0125                 False              False    0.505556    0.465556
tf_target                       tf_target: real vs C3 sign-flipped               30 0.52963 0.529630   0.000000  0.000000            0.0         NaN         NaN      NaN            0.0125                 False              False    0.505556    0.505556

Permutation null (n=100 donor-level label permutations, same CV splits): real AUROC 0.5296 is at the 52.0th percentile of its own null (range 0.209-0.737); baseline_pca AUROC 0.4852 is at the 37.0th percentile of its null (range 0.168-0.754).

## epigenomic (Bonferroni alpha = 0.05 / 3)
     prior                                                comparison  n_paired_splits  mean_a   mean_b  mean_diff  diff_std  wilcoxon_stat  wilcoxon_p  ttest_stat  ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
epigenomic                          epigenomic: real vs baseline_pca               30 0.52963 0.459259   0.070370  0.207323          132.5    0.065473    1.859101 0.073187          0.016667                 False              False         0.5    0.477778
epigenomic epigenomic: real vs C1 degree-preserving (pooled 5 seeds)               30 0.52963 0.425926   0.103704  0.221326          115.5    0.027374    2.566384 0.015705          0.016667                 False               True         0.5    0.443333
epigenomic      epigenomic: real vs C2 fully random (pooled 5 seeds)               30 0.52963 0.488148   0.041481  0.208032          163.5    0.368371    1.092155 0.283756          0.016667                 False              False         0.5    0.515556

Permutation null (n=100 donor-level label permutations, same CV splits): real AUROC 0.5296 is at the 53.0th percentile of its own null (range 0.209-0.790); baseline_pca AUROC 0.4593 is at the 26.0th percentile of its null (range 0.201-0.724).

## combined (Bonferroni alpha = 0.05 / 6)
   prior                                                    comparison  n_paired_splits   mean_a   mean_b     mean_diff  diff_std  wilcoxon_stat  wilcoxon_p    ttest_stat  ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
combined                                combined: real vs baseline_pca               30 0.514815 0.448148  6.666667e-02  0.229015          111.0    0.100890  1.594430e+00 0.121682          0.008333                 False              False    0.533333    0.461111
combined combined: real vs joint C1 degree-preserving (pooled 5 seeds)               30 0.514815 0.429630  8.518519e-02  0.209122          121.0    0.061798  2.231129e+00 0.033568          0.008333                 False              False    0.533333    0.444444
combined      combined: real vs joint C2 fully random (pooled 5 seeds)               30 0.514815 0.481481  3.333333e-02  0.204317          182.5    0.303664  8.935831e-01 0.378901          0.008333                 False              False    0.533333    0.495556
combined            combined: real vs joint C3 sign-flipped (TF block)               30 0.514815 0.514815  1.850372e-18  0.029179            1.0    0.654721  3.473331e-16 1.000000          0.008333                 False              False    0.533333    0.533333
combined               combined: real vs tf_target_real (single prior)               30 0.514815 0.529630 -1.481481e-02  0.166943          114.0    0.683610 -4.860582e-01 0.630579          0.008333                 False              False    0.533333    0.505556
combined              combined: real vs epigenomic_real (single prior)               30 0.514815 0.529630 -1.481481e-02  0.181600          118.5    0.551876 -4.468286e-01 0.658318          0.008333                 False              False    0.533333    0.500000

## combined -- exploratory ablation (isolates which prior drives the effect; uncorrected, NOT part of the 6-comparison family above)
   prior                                       comparison  n_paired_splits   mean_a   mean_b  mean_diff  diff_std  wilcoxon_stat  wilcoxon_p  ttest_stat  ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
combined  combined: real vs only_tf_real (pooled 5 seeds)               30 0.514815 0.502222   0.012593  0.162325          191.5    0.398835    0.424904 0.674043              0.05                 False              False    0.533333    0.498889
combined combined: real vs only_epi_real (pooled 5 seeds)               30 0.514815 0.437037   0.077778  0.213447          146.5    0.076868    1.995838 0.055419              0.05                 False              False    0.533333    0.464444

Permutation null (n=100 donor-level label permutations, same CV splits): real AUROC 0.5148 is at the 46.0th percentile of its own null (range 0.207-0.770); baseline_pca AUROC 0.4481 is at the 27.0th percentile of its null (range 0.219-0.738).
