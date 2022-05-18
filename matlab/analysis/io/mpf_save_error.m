function [] = mpf_save_error(S, range_query_id)
  saveas(S.fig.f_error(1), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_50_error_fro.png'));
  saveas(S.fig.f_error(2), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_51_error_fro_rel.png'));
  saveas(S.fig.f_error(3), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_52_error_1.png'));
  saveas(S.fig.f_error(4), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_53_error_1_rel.png'));

  if length(S.fig.f_error) >= 5
    saveas(S.fig.f_error(5), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_54_error_inf.png'));
  end

  if length(S.fig.f_error) >= 6
    saveas(S.fig.f_error(6), strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_55_error_inf_rel.png'));
  end
end
