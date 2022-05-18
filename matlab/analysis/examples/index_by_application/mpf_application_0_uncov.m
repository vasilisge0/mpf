close all;

data_prefix='uncov_128_128_3_5_';

% access directories 
filename_approx_prefix = strcat('../../../../output/index_by_application/mpf_application_0_uncov/approx/', data_prefix);  % do that
filename_meta = strcat('../../../../output/index_by_application/mpf_application_0_uncov/', data_prefix, 'meta');
filename_exact_inverse = '../../../../data/exact_inverse/banded_covariance/dataset_uncov_128_128_3_5_Di1.mtx'
                                                                            %dataset_lapl_2D_shift_2.0_128_128_exact_Di2.mtx
path_plots = '../../../../output/index_by_application/mpf_application_0_uncov/plots'
path_log = '../../../../output/index_by_application/mpf_application_0_uncov/log';
path_approx = '../../../../output/index_by_application/mpf_application_0_uncov/approx/';

% initialize analysis objects
S = MPF_Log();
T = MPF_LogIds();
S.filename.Ai = {filename_exact_inverse};
S.title = 'uncov-128-128-3-5';
query_range = [1 3];

% read logs by stride
mpf_read_log_by_range(filename_meta, path_log, data_prefix, T, S, query_range, 'stride');
legend_names = string(mpf_get_strides(S));
mpf_get_speedup(S);

mpf_eval_error_by_range(filename_meta, path_approx, filename_approx_prefix, T, S, query_range, 'stride');
S.filename_output = strcat('../', S.filename_output); % corrects output

% plots
f = mpf_runtimes_plot(S, "exp", legend_names, S.title);
f_bar = mpf_bar_plot(S, legend_names, S.title);
f_nrhs = mpf_plot_nrhs(S, legend_names, S.title);
f_perf = mpf_perf_plot(S, legend_names, S.title);
f_speedup = mpf_speedup_plot(S, "exp", legend_names, S.title);
[f_error, f_error_entrywise] = mpf_error_plot(S, "exp", legend_names, S.title);

% save plots
S.path_plots = 'mpf_application_0_uncov';
mpf_save_runtimes(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_0_runtimes'), f);
mpf_save_bar(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_1_bar'), f_bar);
mpf_save_nrhs(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_2_nrhs'), f_nrhs);
mpf_save_perf(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_3_perf'), f_perf);
mpf_save_speedup(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_4_speedup'), f_speedup);
mpf_save_error(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_5_error'), f_error);
mpf_save_entrywise_error(strcat(path_plots, '/', data_prefix, S.path_plots, '_', 'plot_6_entrywise_error'), f_error_entrywise, S);
