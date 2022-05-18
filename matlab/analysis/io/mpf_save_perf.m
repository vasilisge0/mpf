function [f] = mpf_save_perf(S, range_query_id)
  saveas(S.fig.f_perf(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_30_perf.png'));
end
