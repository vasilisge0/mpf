close all;
root = '/home/vasilis';
filename = strcat(root, strcat('/mpf/tests/automation/dataset_uncov_256_3_5_A_stride.txt'));
%filename_gko = strcat(root, '/mpf/tests/automation/dataset_lapl_2D_shift_4.8_128_128_A_gko_cuda.txt');
path_meta = strcat(root, '/mpf_output/meta/');
prefix = 'stride_exp2_';

S = MPF_Log();
T = MPF_LogIds();

mpf_read_log_by_range(filename, path_meta, prefix, T, S, [1 2], 'stride');
legend_names = string(mpf_get_strides(S));
f = mpf_runtimes_plot(S, "exp", legend_names, 'uncov-256-3-5');
f_bar = mpf_bar_plot(S, legend_names, 'uncov-256-3-5');
f_nrhs = mpf_plot_nrhs(S, legend_names, 'uncov-256-3-5');
f_perf = mpf_perf_plot(S, legend_names, 'uncov-256-3-5');
mpf_get_speedup(S);
f_speedup = mpf_speedup_plot(S, "exp", legend_names, 'uncov-256');

%S.n_max_B(:, 1)
