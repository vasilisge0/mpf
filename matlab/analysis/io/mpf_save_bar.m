function [f] = mpf_save_bar(S, range_query_id)
  saveas(S.fig.f_bar(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_10_work_distribution.png'));
end
