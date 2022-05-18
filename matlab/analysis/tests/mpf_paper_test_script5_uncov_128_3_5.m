close all;
root = '/home/vasilis';
filename = strcat(root, strcat('/mpf/tests/automation/dataset_uncov_128_3_5_A_stride.txt'));
%filename_gko = strcat(root, '/mpf/tests/automation/dataset_lapl_2D_shift_4.8_128_128_A_gko_cuda.txt');
path_meta = strcat(root, '/mpf_output/meta/');
path_approx = strcat(root, '/mpf_output/approx/');
prefix = 'stride_exp3_';

S = MPF_Log();
T = MPF_LogIds();

S.filename_Ai = {'/home/vasilis/mpf_output/exact/uncov_128_3_5_Di1.mtx'};

mpf_read_log_by_range(filename, path_meta, prefix, T, S, [1 2], 'stride');
legend_names = string(mpf_get_strides(S));

f = mpf_runtimes_plot(S, "exp", legend_names, 'uncov-128-3-5');

f_bar = mpf_bar_plot(S, legend_names, 'uncov-128-3-5');

f_nrhs = mpf_plot_nrhs(S, legend_names, 'uncov-128-3-5');

f_perf = mpf_perf_plot(S, legend_names, 'uncov-128-3-5');

mpf_get_speedup(S);
f_speedup = mpf_speedup_plot(S, "exp", legend_names, 'uncov-128');

S.filename_A = cellfun(@(A, B, C) replace(A, B, C), S.filename_A, repmat({'georgiov'}, 1, size(S.filename_A, 2)), ...
  repmat({'vasilis'}, 1, size(S.filename_A, 2)), 'UniformOutput', false)
S.filename_A = cellfun(@(A, B, C) replace(A, B, C), S.filename_A, repmat({'mpf_data'}, 1, size(S.filename_A, 2)),  repmat({'data_mpf/data'}, 1, size(S.filename_A, 2)), 'UniformOutput', false)
S.filename_output = cellfun(@(A, B, C) replace(A, B, C), S.filename_output, repmat({'georgiov'}, 1, size(S.filename_output, 2)),  repmat({'vasilis'}, 1, size(S.filename_output, 2)), 'UniformOutput', false)
S.filename_Ai = repmat(S.filename_Ai, 1, S.nlogs);

mpf_eval_error_by_range(filename, path_approx, prefix, T, S, [1 2], 'stride');

f_error = mpf_error_plot(S, "exp", legend_names, 'uncov-128');

S.n_max_B(:, 1)

S.filename_plots = 'test_script5';
%mkdir(strcat('matlab/analysis/output/', string(S.filename_output)));

mpf_save_runtimes(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_0_runtimes'), f);
mpf_save_bar(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_1_bar'), f_bar);
mpf_save_nrhs(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_2_nrhs'), f_nrhs);
mpf_save_perf(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_3_perf'), f_perf);
mpf_save_speedup(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_4_speedup'), f_speedup);
mpf_save_error(strcat('matlab/analysis/output/', S.filename_plots, 'mpf_plot_5_error'), f_error);
