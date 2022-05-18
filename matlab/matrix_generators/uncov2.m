function [A, X, Y] = uncov2(alpha, beta, dims)

    % indices in x dimensions
    cx = (dims{1}+1)/2;
    X  = kron(((1:dims{1})-cx)', ones(dims{2}, 1));

    % indices in y dimensions
    cy = (dims{2}+1)/2;
    Y  = kron(ones(dims{2}, 1), ((1:dims{1})-cy)');

    % initialize A
    n = dims{1}*dims{2};
    A = sparse(n, n);

    % setup columns of A
    for i = 1:n

        [I, J, M] = find(sqrt((X-X(i)).^2 + (Y-Y(i)).^2) <= alpha);
        A(I, i) = M;
        fprintf('%d/%d -> %d\n', i, n, nnz(I));
        %d = sqrt((X-X(i)).^2 + (Y-Y(i)).^2);                                    % euclidean distance from (X(i), Y(i)) - all

        % select entries
        %sel = d > alpha;
        %d(sel)  = 0;
        %d(~sel) = (1 - d(~sel)/alpha).^beta;

        %% apply A
        %A(:, i) = d;
    end

end
