function [] = mpf_get_total_probe_runtime(S, i)
  S.runtime.probe_total(:, i) = ...
    + S.runtime.probe_contract(:, i) ...
    + S.runtime.probe_expand(:, i)...
    + S.runtime.probe_color(:, i) ...
    + S.runtime.probe_other(:, i);
end
