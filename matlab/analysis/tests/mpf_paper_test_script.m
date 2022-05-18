close all;
root = '/home/vasilis';
filename = strcat(root, '/mpf/tests/automation/dataset_lapl_2D_shift_4.8_128_128_A.txt');
filename_gko = strcat(root, '/mpf/tests/automation/dataset_lapl_2D_shift_4.8_128_128_A_gko_cuda.txt');
path_meta = strcat(root, '/mpf_output/meta/');

S = MPF_Log();
T = MPF_LogIds();

mpf_read_log_by_range(filename, path_meta, '', T, S, [1 7], 'solver_inner_nthreads');
mpf_read_log_by_range(filename_gko, path_meta, '', T, S, [1 1], 'solver_inner_nthreads');

legend_names = mpf_get_solver_inner_nthreads(S);
f = mpf_runtimes_plot(S, "exp", legend_names, '2D-128-128');
f_bar = mpf_bar_plot(S, legend_names, '2D-128-128');
f_perf = mpf_perf_plot(S, legend_names, '2D-128-128');
mpf_get_speedup(S);
f_speedup = mpf_speedup_plot(S, "exp", legend_names, '2D-128-128');

%saveas(f, '~/mpf_plots/mpf_runtimes_lapl_2D_4.8_128_128.png');
%saveas(f_bar, '~/mpf_plots/mpf_bar_lapl_2D_4.8_128_128.png');
%saveas(f_perf, '~/mpf_plots/mpf_perf_lapl_2D_4.8_128_128.png');
%saveas(f_speedup(1), '~/mpf_plots/mpf_seedup_probe_lapl_2D_4.8_128_128.png');
%saveas(f_speedup(2), '~/mpf_plots/mpf_seedup_solver_lapl_2D_4.8_128_128.png');
%saveas(f_speedup(3), '~/mpf_plots/mpf_seedup_total_lapl_2D_4.8_128_128.png');

% uncov

filename = strcat(root, '/mpf/tests/automation/dataset_uncov_3_5_256_256.txt');
filename_gko = strcat(root, '/mpf/tests/automation/dataset_uncov_3_5_256_256_gko.txt');

S2 = MPF_Log();
T2 = MPF_LogIds();

mpf_read_log_by_range(filename, path_meta, '', T2, S2, [1 7], 'solver_inner_nthreads');
mpf_read_log_by_range(filename_gko, path_meta, '', T2, S2, [1 1], 'solver_inner_nthreads');
legend_names2 = mpf_get_solver_inner_nthreads(S2);

f2 = mpf_runtimes_plot(S2, "exp", legend_names2, 'uncov-256-256');
f_bar2 = mpf_bar_plot(S2, legend_names2, 'uncov-256-256');
f_perf2 = mpf_perf_plot(S2, legend_names2, 'uncov-256-256');
mpf_get_speedup(S2);
f_speedup2 = mpf_speedup_plot(S2, "exp", legend_names2, 'uncov-256-256');

%saveas(f2, '~/mpf_plots/mpf_uncov.png');
%saveas(f_bar2, '~/mpf_plots/mpf_bar_uncov.png');
%saveas(f_perf2, '~/mpf_plots/mpf_perf_uncovpng');
%saveas(f_speedup2(1), '~/mpf_plots/mpf_speedup_probe_uncov.png');
%saveas(f_speedup2(2), '~/mpf_plots/mpf_speedup_solver_uncov.png');
%saveas(f_speedup2(3), '~/mpf_plots/mpf_speedup_total_uncov.png');

% uncov clustered

S3 = MPF_Log();
T3 = MPF_LogIds();

filename = strcat(root, '/mpf/tests/automation/clust_c100_p100000_n125.txt');
filename_gko = strcat(root, '/mpf/tests/automation/clust_c100_p100000_n125_gko_cuda.txt');

mpf_read_log_by_range(filename, path_meta, '', T3, S3, [1 7], 'solver_inner_nthreads');
mpf_read_log_by_range(filename_gko, path_meta, '', T3, S3, [1 1], 'solver_inner_nthreads');
legend_names3 = mpf_get_solver_inner_nthreads(S3);

f3 = mpf_runtimes_plot(S3, "exp", legend_names3, 'clust-c100-p100000');
f_bar3 = mpf_bar_plot(S3, legend_names3, 'clust-c100-p100000');
f_perf3 = mpf_perf_plot(S3, legend_names3, 'clust-c100-p100000');
mpf_get_speedup(S3);
f_speedup3 = mpf_speedup_plot(S3, "exp", legend_names3,  'clust-c100-p100000');

%saveas(f3, '~/mpf_plots/mpf_clust.png');
%saveas(f_bar3, '~/mpf_plots/mpf_bar_clust.png');
%saveas(f_perf3, '~/mpf_plots/mpf_perf_clust.png');
%saveas(f_speedup3(1), '~/mpf_plots/mpf_speedup_probe_clust.png');
%saveas(f_speedup3(2), '~/mpf_plots/mpf_speedup_solver_clust.png');
%saveas(f_speedup3(3), '~/mpf_plots/mpf_speedup_total_clust.png');
