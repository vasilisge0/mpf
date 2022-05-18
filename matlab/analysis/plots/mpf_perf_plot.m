function [f] = mpf_perf_plot(S, query_id)

  legend_names = S.legend_names{:};
  dataname = S.title;

  S.fig.f_perf(1, query_id) = figure('Units','normalized','Position',[0 0 1 1]);
  T = [];
  for i = 1:S.n_logs

    for j = 1:length(S.runtime.probe_total(:, i))
      T(j, :) = [S.runtime.probe_total(j, i), S.runtime.solver_total(j, i)];
      T(j, :) = T(j, :)/sum(T(j, :));
    end

    subplot(1, S.n_logs, i);
    bar(T, 'stacked');
    title(legend_names{1, i});
    set(gca,'xticklabel', S.probe.n_max_B(:, i));
    xlabel('number of rhs')
    ylabel('work distribution');
    legend('probe', 'solver', 'Location', 'NorthWest');
  end

  sgtitle(dataname);

  hold off;
  grid on;
end
