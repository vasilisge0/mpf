function [] = mpf_get_thread_speedup(S)

  for i = 1:S.nlogs
    S.thread_speedup_solver_total(:, i) = S.runtime_solver_total(:, 1) ./ S.runtime_solver_total(:, i);
    S.thread_speedup_probe_total(:, i) = S.runtime_probe_total(:, 1) ./ S.runtime_probe_total(:, i);
    S.thread_speedup_total(:, i) = (S.runtime_solver_total(:, 1) + S.runtime_probe_total(:, 1)) ./ (S.runtime_solver_total(:, i) + S.runtime_probe_total(:, i));
  end

end
