function [f] = mp_plot_nrhs(S, query_id)

  legend_names = S.legend_names{:};
  dataname = S.title;

  p = size(S.runtime.probe_total, 1);
  x = (2.^(1:p).');
  xname = "Expansion exponent k";
  S.fig.f_nrhs = figure();
  for i = 1:S.n_logs
    semilogy(x, S.probe.n_max_B(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('number of right-hand sides');
  title(strcat(dataname, ' / Number of right-hand sides'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)
end
