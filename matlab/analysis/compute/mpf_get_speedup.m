function [] = mpf_get_speedup(S, query_id)
  for i = 1:S.n_logs
    S.speedup.solver_total(:, i) = S.runtime.solver_total(:, 1) ./ S.runtime.solver_total(:, i);
    S.speedup.probe_total(:, i) = S.runtime.probe_total(:, 1) ./ S.runtime.probe_total(:, i);
    S.speedup.total(:, i) = (S.runtime.solver_total(:, 1) + S.runtime.probe_total(:, 1)) ./ (S.runtime.solver_total(:, i) + S.runtime.probe_total(:, i));
  end
end
