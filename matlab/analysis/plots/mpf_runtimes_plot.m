function [f] = mpf_runtimes_plot(S, xname, query_id)

  legend_names = S.legend_names{:};
  dataname = S.title;

  if xname == "nrhs"
    x = S.n_max_B(:, 1);
    xname = "Number of right-hand sides";

  elseif xname == "exp"
    p = size(S.runtime.probe_total, 1);
    x = (2.^(1:p).');
    xname = "Expansion exponent k";
  end

  S.fig.f_runtimes(1, query_id) = figure();
  for i = 1:S.n_logs
    semilogy(x, S.runtime.probe_total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / Probe total'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_runtimes(2, query_id) = figure();
  for i = 1:S.n_logs
    semilogy(x, S.runtime.solver_total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / Solver Total'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_runtimes(3, query_id) = figure();
  for i = 1:S.n_logs
    semilogy(x, S.runtime.solver_inner(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / Solver Inner'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_runtimes(4, query_id) = figure();
  for i = 1:S.n_logs
    semilogy(x, S.runtime.probe_total(:, i)+S.runtime.solver_total(:, i), '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / Total'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
  set(gca,'fontsize',12)
end
