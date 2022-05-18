function [f] = mpf_save_speedup(S, range_query_id)
  saveas(S.fig.f_speedup(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_40_speedup_probe.png'));
  saveas(S.fig.f_speedup(2), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_41_speedup_solver.png'));
  saveas(S.fig.f_speedup(3), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_42_speedup_total.png'));
end
