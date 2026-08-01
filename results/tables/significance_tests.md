# Paired significance tests -- donor-level disease-status AUROC


## tf_target (Bonferroni alpha = 0.05 / 4)
    prior                                               comparison  n_paired_splits   mean_a   mean_b  mean_diff  diff_std  wilcoxon_stat   wilcoxon_p  ttest_stat      ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
tf_target                          tf_target: real vs baseline_pca               30 0.959318 0.983669  -0.024351  0.018744            2.0 3.152627e-06   -7.115548 7.883201e-08            0.0125                  True               True    0.909549    0.947093
tf_target tf_target: real vs C1 degree-preserving (pooled 5 seeds)               30 0.959318 0.963375  -0.004057  0.019283          193.0 4.279546e-01   -1.152338 2.585931e-01            0.0125                 False              False    0.909549    0.905433
tf_target      tf_target: real vs C2 fully random (pooled 5 seeds)               30 0.959318 0.929040   0.030278  0.022165           12.0 1.303852e-07    7.482119 3.021101e-08            0.0125                  True               True    0.909549    0.854817
tf_target                       tf_target: real vs C3 sign-flipped               30 0.959318 0.958737   0.000581  0.003760           16.0 7.772974e-01    0.846284 4.043239e-01            0.0125                 False              False    0.909549    0.909514

Permutation null (n=100 donor-level label permutations, same CV splits): real AUROC 0.9593 is at the 100.0th percentile of its own null (range 0.404-0.612); baseline_pca AUROC 0.9837 is at the 100.0th percentile of its null (range 0.390-0.603).

## epigenomic (Bonferroni alpha = 0.05 / 3)
     prior                                                comparison  n_paired_splits   mean_a   mean_b  mean_diff  diff_std  wilcoxon_stat  wilcoxon_p  ttest_stat      ttest_p  bonferroni_alpha  significant_wilcoxon  significant_ttest  accuracy_a  accuracy_b
epigenomic                          epigenomic: real vs baseline_pca               30 0.926507 0.984227  -0.057721  0.028379            0.0    0.000002  -11.140107 5.394235e-12          0.016667                  True               True    0.818534    0.952502
epigenomic epigenomic: real vs C1 degree-preserving (pooled 5 seeds)               30 0.926507 0.937371  -0.010864  0.020953          112.0    0.013192   -2.839911 8.165078e-03          0.016667                  True               True    0.818534    0.865599
epigenomic      epigenomic: real vs C2 fully random (pooled 5 seeds)               30 0.926507 0.933466  -0.006959  0.029699          180.0    0.280202   -1.283402 2.095125e-01          0.016667                 False              False    0.818534    0.859408

Permutation null (n=100 donor-level label permutations, same CV splits): real AUROC 0.9265 is at the 100.0th percentile of its own null (range 0.383-0.582); baseline_pca AUROC 0.9842 is at the 100.0th percentile of its null (range 0.377-0.598).
