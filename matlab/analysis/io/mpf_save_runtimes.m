function [f] = mpf_save_runtimes(S, range_query_id)
  saveas(S.fig.f_runtimes(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_00_runtimes_probe.png'));
  saveas(S.fig.f_runtimes(2), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_01_runtimes_solver_total.png'));
  saveas(S.fig.f_runtimes(3), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_02_runtimes_solver_inner.png'));
  saveas(S.fig.f_runtimes(4), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_03_runtimes_total.png'));
end
