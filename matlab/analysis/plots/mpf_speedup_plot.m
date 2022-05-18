function [f] = mpf_speedup_plot(S, xname, query_id)

  legend_names = S.legend_names{:};
  dataname = S.title;

  if xname == "nrhs"
    x = S.probe.n_max_B(:, 1);
    xname = "Number of right-hand sides";

  elseif xname == "exp"
    p = size(S.runtime.probe_total, 1);
    x = (2.^(1:p).');
    xname = "Expansion exponent k";
  end

  S.fig.f_speedup(1) = figure();
  larray = [];
  for i = 1:S.n_logs
    plot(x, S.speedup.probe_total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' - probe speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_speedup(2) = figure();
  for i = 1:S.n_logs
    plot(x, S.speedup.solver_total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / solver speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_speedup(3) = figure();
  for i = 1:S.n_logs
    plot(x, S.speedup.total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / total speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize', 12)
end
