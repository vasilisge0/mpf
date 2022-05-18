% 1D modeling
niters = 12;
n_disp = 50;
N = 4;
n = 100;
di = zeros(n, niters);
di_off = zeros(n, niters);
max_diff = zeros(niters, 1);
max_off = zeros(niters, 1);

gamma = 0;
beta = 0.1;
alpha = [gamma; gamma+(1:niters-1)'*beta];


f1 = figure();
close all;

for i = 1:niters
  % initialize A and compute Ai
  [~,~,A] = laplacian([10 10], {'DD' 'DD'});
  A = A + speye(nx*ny)*alpha(i);
  Ai = inv(A);

  % evaluate diagonal and off-diagonal
  di(:, i) = diag(abs(Ai));
  di_off(:, i) = sum(abs(Ai)-diag(di(:, i)), 2);
  max_off(i) = max(di_off(:, i));
  diff = di_off(:, i)./di(:, i);
  disp("max_off")
  max_off(i)
  disp("max_diff")
  max_diff(i) = max(diff)
  disp("di")
  di(1:10, i)
  disp("di_off")
  di_off(1:10, i)

  % plots
  subplot(ceil(niters/N), N, i);
  surf(Ai(1:n_disp, 1:n_disp));
  title(sprintf('alpha(%d): %d, max-diff(%d): %1.2f', i, alpha(i), i, max_diff(i)));
end

max_diff
max_off
di(1:10, 1:niters)
di_off(1:10, 1:niters)
