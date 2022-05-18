close all;

% access directories 
filename_approx_prefix = '';  % do that 
filename_meta = '../../output/mpf_test7_blk_probing_cg/meta';
filename_exact_inverse = '../../data/exact_inverses/test' % do that
path_plots = '../../output/mpf_test7_blk_probing_cg/plots'
path_log = '../../output/mpf_test7_blk_probing_cg/log';
path_approx = '../../output/mpf_test7_blk_probing_cg/approx/';

% initialize analysis objects
S = MPF_Log();
T = MPF_LogIds();
S.filename_Ai = {filename_exact_inverse};
S.title = 'uncov-128-3-5';
query_range = [1 1];

% read logs by stride
mpf_read_log_by_range(filename_meta, path_log, filename_approx_prefix, T, S, query_range, 'stride');
legend_names = string(mpf_get_strides(S));
S.nlogs

% plots
f = mpf_runtimes_plot(S, "exp", legend_names, S.title);
f_bar = mpf_bar_plot(S, legend_names, S.title);
f_nrhs = mpf_plot_nrhs(S, legend_names, S.title);
f_perf = mpf_perf_plot(S, legend_names, S.title);

mpf_get_speedup(S);
f_speedup = mpf_speedup_plot(S, "exp", legend_names, S.title);

%mpf_eval_error_by_range(filename_meta, path_approx, filename_approx_prefix, T, S, query_range, 'stride');
%f_error = mpf_error_plot(S, "exp", legend_names, S.title);

% save plots
S.path_plots = '/mpf_test7_';

mpf_save_runtimes(strcat(path_plots, S.path_plots, 'plot_0_runtimes'), f);
mpf_save_bar(strcat(path_plots, S.path_plots, 'plot_1_bar'), f_bar);
mpf_save_nrhs(strcat(path_plots, S.path_plots, 'plot_2_nrhs'), f_nrhs);
mpf_save_perf(strcat(path_plots, S.path_plots, 'plot_3_perf'), f_perf);
mpf_save_speedup(strcat(path_plots, S.path_plots, 'plot_4_speedup'), f_speedup);
%mpf_save_error(strcat(path_plots, S.path_plots, 'plot_5_error'), f_error);
