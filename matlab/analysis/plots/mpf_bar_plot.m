function [f] = mpf_bar_plot(S, query_id)

  legend_names = S.legend_names{:};
  dataname = S.title;

  S.fig.f_bar(1, query_id) = figure('Units','normalized','Position',[0 0 1 1]);
  T = [];
  for i = 1:S.n_logs
    for j = 1:length(S.runtime.probe_total(:, i))
      T(j, :) = [S.runtime.solver_inner(j, i); ...
                 S.runtime.solver_reconstruct(j, i)];
      T(j, :) = T(j, :)/sum(T(j, :));
    end

    subplot(1, S.n_logs, i);
    bar(T, 'stacked');
    set(gca,'xticklabel', S.probe.n_max_B(:, i));
    title(legend_names(1, i));
    xlabel('number of rhs')
    ylabel('work distribution');
    legend('inner-solver', 'reconstruct', 'Location', 'South');

  end

  sgtitle(dataname);

  hold off;
  grid on;

end
