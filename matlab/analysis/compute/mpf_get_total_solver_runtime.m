function [] = mpf_get_total_solver_runtime(S, i)
  S.runtime.solver_total(:, i) = ...
    + S.runtime.solver_pre_process(:, i) ...
    + S.runtime.solver_alloc(:, i) ... =
    + S.runtime.solver_generate_rhs(:, i) ...
    + S.runtime.solver_inner(:, i) ...
    + S.runtime.solver_reconstruct(:, i) ...
    + S.runtime.solver_post_process(:, i);
end

