function [S] = mpf_plot(S, query_id, varargin)

  % low memory configuration
  save_mem = false
  if nargin == 3
    save_mem = varargin{1}
  end

  % read logs by query type
  mpf_read_log_by_range(S, query_id);

  % compute speedup
  mpf_get_speedup(S, query_id);

  % computes error
  %mpf_eval_error_by_range(S, query_id, save_mem);

  % extracts legends for bar-plots
  mpf_get_legend(S, query_id);

  % plots
  mpf_runtimes_plot(S, "exp", query_id);
  mpf_bar_plot(S, query_id);
  mpf_plot_nrhs(S, query_id);
  mpf_perf_plot(S, query_id);
  mpf_speedup_plot(S, "exp", query_id);
  %mpf_error_plot(S, "exp", query_id);

  % save plots
  mpf_save_runtimes(S, query_id);
  mpf_save_bar(S, query_id);
  mpf_save_nrhs(S, query_id);
  mpf_save_perf(S, query_id);
  mpf_save_speedup(S, query_id);
  %mpf_save_error(S, query_id);
  %mpf_save_entrywise_error(S, query_id);

end
