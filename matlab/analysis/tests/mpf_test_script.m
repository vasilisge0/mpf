S = MPF_Log();
T = MPF_LogIds(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
%T = MPF_LogIds();

filename = '/home/georgiov/mpf/tests/automation/dataset_lapl_2D_shift_4.8_128_128_A.txt';
path_meta = '/home/georgiov/mpf_output/meta/';

mpf_read_log_by_range(filename, path_meta, T, S, [1 1], 'solver_inner_nthreads')

figure, loglog(S.n_max_B(:, 1), S.runtime_solver_inner(:, 1), '--x', 'LineWidth', 2); hold on;
        loglog(S.n_max_B(:, 2), S.runtime_solver_inner(:, 2), '--o', 'LineWidth', 2); hold off;
grid on;
xlabel('number of right-hand sides');
ylabel('runtime (sec)');
title('Inner Solver')
legend('mpf-omp-1', 'mpf-omp-2', 'Location', 'NorthEast');

figure, loglog(S.n_max_B(:, 1), S.runtime_solver_total(:, 1), '--x', 'LineWidth', 2); hold on;
        loglog(S.n_max_B(:, 2), S.runtime_solver_total(:, 2), '--o', 'LineWidth', 2); hold off;
grid on;
xlabel('number of right-hand sides');
ylabel('runtime (sec)');
title('Outer Solver')
legend('mpf-omp-1', 'mpf-omp-2', 'Location', 'NorthEast');
