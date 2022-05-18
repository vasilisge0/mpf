% 3D modeling
n_disp = 50;
niters = 12;
N = 4;

I = 3;
nx = [4, 8, 16];
ny = [4, 8, 16];
nz = [4, 8, 16];

n = nx(I)*ny(I)*nz(I);
di = zeros(n, niters);
di_off = zeros(n, niters);
max_diff = zeros(niters, 1);
max_off = zeros(niters, 1);

gamma = 0;
beta = 0.01;
alpha = [gamma; gamma+(1:niters-1)'*beta];
kA = zeros(niters, 1);

close all;
f1 = figure();
f2 = figure();
f3 = figure();

for i = 1:niters
  % initialize A and compute Ai
  [~,~,A] = laplacian([nx(I) ny(I) nz(I)], {'DD' 'DD', 'DD'});
  A = A + speye(n)*alpha(i);
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
  dap = 1./diag(A);
  diff_ap = abs(dap - di(:, i))./abs(di(:, i));
  max_diff_ap(i) = max(diff_ap);
  disp('max-diff-ap');
  max_diff_ap(i);
  kA(i) = cond(A);

  % plots
  set(0, 'CurrentFigure', f1)
  subplot(ceil(niters/N), N, i);
  surf(Ai(1:n_disp, 1:n_disp));
  title(sprintf('alpha(%d): %d, max-diff: %1.3f, max-diff-a: %1.3f', i, alpha(i), max_diff(i), max_diff_ap(i)));

  set(0, 'CurrentFigure', f2)
  subplot(ceil(niters/N), N, i);
  hold on;
  h1 = plot(di, '--xb');
  h2 = plot(1./(diag(A)), 'or');
  h3 = plot(sum(Ai, 2), '*g');
  hold off;
  legend([h1(1) h2(1) h3(1)], 'di', '1./d(A)', 'Ai*e')
  title(sprintf('alpha(%d): %d, max-diff: %1.3f, max-diff-a: %1.3f', i, alpha(i), max_diff(i), max_diff_ap(i)));

  set(0, 'CurrentFigure', f3)
  surf(Ai(1:n_disp, 1:n_disp));
  saveas(f3, sprintf('fig_3D_%1.2f.png', alpha(i)));
end

f3 = figure()
plot(alpha, kA, '--o')
title('\kappa(A, alpha)')
max_diff
max_off
di(1:10, 1:niters)
di_off(1:10, 1:niters)
