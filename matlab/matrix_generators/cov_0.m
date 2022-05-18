f_samples = 'test/mp_cov_0.mtx';
n = 16384
n_samples = 500;

% mean value
mu = 0;

% inverse covariance
Sinv = toeplitz([1.25, -0.5, zeros(1, n-2)]);
U = chol(Sinv);

% generate random samples generated via precision sampling and write in file
fid = fopen(f_samples, 'w');
comment = '%%Matrix Market matrix array real symmetric';
fprintf(fid, '%s\n', comment)
fprintf(fid, '%d %d\n', n, n_samples);
Z = randn(n, n_samples);
X = mu + U\Z;
format = [repmat('%e ', 1, n_samples), '\n'];
for i = 1:n
  fprintf(fid, format, X(i, :));
end
fclose(fid);
