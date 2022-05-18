S = MPF_Log();

Q{1} = MPF_RangeQuery( ...
  '../../../../output/index_by_application/mpf_application_04_uncov', ...
  '../../../../data/exact_inverse/banded_covariance', ...
  'uncov_128_128_3_5', ...
  'stride', ...
  [1 3]);

mpf_add_query(S, Q{1});
mpf_plot(S, 1, true);
