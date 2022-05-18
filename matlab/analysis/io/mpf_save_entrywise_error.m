function [f] = mpf_save_entrywise_error(S, range_query_id)

  r = 0;
  if length(S.fig.f_error_entrywise) > 0
    for i = 1:S.n_logs
      for j = 1:length(S.probe.n_levels(:, i))
        saveas(S.fig.f_error_entrywise{j, i}, strcat(S.filename.path_plots{range_query_id}, '/', S.filename.data_prefix{range_query_id}, 'plot_6', string(i-1), string(j-1), '_error_entrywise', '.png'));
        r = r + 1;
      end
    end
  end
end
