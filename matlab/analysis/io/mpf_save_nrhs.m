function [f] = mpf_save_nrhs(S, range_query_id)
  saveas(S.fig.f_nrhs(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_20_nrhs.png'));
end
