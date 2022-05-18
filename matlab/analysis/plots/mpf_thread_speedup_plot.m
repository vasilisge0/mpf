function [f] = mpf_thread_speedup_plot(S, legend_names, dataname)

  f(1) = figure();
  larray = [];
  for i = 1:S.nlogs
    plot(S.n_max_B(:, 1), S.thread_speedup_probe_total(:, i), '--x', 'LineWidth', 2);
    hold on;
  end
  xlabel('number of right-hand sides');
  ylabel('runtime (sec)');
  title(strcat(dataname, ' - probe thread-speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;

  f(2) = figure();
  for i = 1:S.nlogs
    plot(S.n_max_B(:, 1), S.thread_speedup_solver_total(:, i), '--x', 'LineWidth', 2);
    hold on;
  end
  xlabel('number of right-hand sides');
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / solver thread-speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;

  f(3) = figure();
  for i = 1:S.nlogs
    plot(S.n_max_B(:, 1), S.thread_speedup_total(:, i), '--x', 'LineWidth', 2);
    hold on;
  end
  xlabel('number of right-hand sides');
  ylabel('runtime (sec)');
  title(strcat(dataname, ' / total thread-speedup'));
  legend(legend_names, 'Location', 'NorthWest');
  hold off;
  grid on;
end
