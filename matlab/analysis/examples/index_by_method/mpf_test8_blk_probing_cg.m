close all;

% access directories 
filename_approx_prefix = '../../output/index_by_method/mpf_test8_blk_probing_cg/approx/';  % do that 
filename_meta = '../../output/index_by_method/mpf_test8_blk_probing_cg/meta';
filename_exact_inverse = '../../data/exact_inverse/clustered_covariance/Di16_clust_c100_p50000_n125.mtx'
path_plots = '../../output/index_by_method/mpf_test8_blk_probing_cg/plots'
path_log = '../../output/index_by_method/mpf_test8_blk_probing_cg/log';
path_approx = '../../output/index_by_method/mpf_test8_blk_probing_cg/approx/';

% initialize analysis objects
S = MPF_Log();
T = MPF_LogIds();
S.filename_Ai = {filename_exact_inverse};
S.title = 'clust';
query_range = [1 1];

% read logs by stride
mpf_read_log_by_range(filename_meta, path_log, '', T, S, query_range, 'stride');
legend_names = string(mpf_get_strides(S));
mpf_get_speedup(S);

mpf_eval_error_by_range(filename_meta, path_approx, filename_approx_prefix, T, S, query_range, 'stride');
S.filename_output = strcat('../', S.filename_output); % corrects output

%mpf_eval_error(S);

% plots
%f = mpf_runtimes_plot(S, "exp", legend_names, S.title);
%f_bar = mpf_bar_plot(S, legend_names, S.title);
%f_nrhs = mpf_plot_nrhs(S, legend_names, S.title);
%f_perf = mpf_perf_plot(S, legend_names, S.title);
%f_speedup = mpf_speedup_plot(S, "exp", legend_names, S.title);
%f_error = mpf_error_plot(S, "exp", legend_names, S.title);

% save plots
%S.path_plots = '/mpf_test8_';
%mpf_save_runtimes(strcat(path_plots, S.path_plots, 'plot_0_runtimes'), f);
%mpf_save_bar(strcat(path_plots, S.path_plots, 'plot_1_bar'), f_bar);
%mpf_save_nrhs(strcat(path_plots, S.path_plots, 'plot_2_nrhs'), f_nrhs);
%mpf_save_perf(strcat(path_plots, S.path_plots, 'plot_3_perf'), f_perf);
%mpf_save_speedup(strcat(path_plots, S.path_plots, 'plot_4_speedup'), f_speedup);
%mpf_save_error(strcat(path_plots, S.path_plots, 'plot_5_error'), f_error);
