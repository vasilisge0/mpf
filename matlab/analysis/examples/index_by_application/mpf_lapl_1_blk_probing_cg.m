close all;

% access directories 
filename_approx_prefix = '../../../../output/index_by_application/mpf_lapl_1_blocking_cg/approx/';  % do that 
filename_meta = '../../../../output/index_by_application/mpf_lapl_1_blocking_cg/meta';
filename_exact_inverse = '../../../../data/exact_inverse/standard_laplacian/dataset_lapl_2D_shift_2.0_128_128_exact_Di2.mtx'
                                                                            %dataset_lapl_2D_shift_2.0_128_128_exact_Di2.mtx
path_plots = '../../../../output/index_by_application/mpf_lapl_1_blocking_cg/plots'
path_log = '../../../../output/index_by_application/mpf_lapl_1_blocking_cg/log';
path_approx = '../../../../output/index_by_application/mpf_lapl_1_blocking_cg/approx/';

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
f = mpf_runtimes_plot(S, "exp", legend_names, S.title);
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
