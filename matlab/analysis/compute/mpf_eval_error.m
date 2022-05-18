function [E] = mpf_eval_error(S, j, varargin)

  mem_save = false;
  if nargin == 3
    mem_save = varargin{1};
  end

  S.data.Ai = mmread(S.filename.Ai{1});
  for i = 1:length(S.probe.n_levels(:, j))
    S.data.diff{i, j} = S.data.Ai - S.data.Ai_approx{i, j};
    if size(S.data.Ai_approx{i, j}, 1) ~= size(S.data.Ai_approx{i, j}, 2)
      S.data.diff{i, j} = abs(S.data.diff{i, j});
    end

    S.num_error.norm_fro{i, j} = norm(S.data.diff{i, j}, 'fro');
    S.num_error.norm_fro_rel{i, j} = norm(S.data.diff{i, j}, 'fro')/norm(S.data.Ai, 'fro');
    S.num_error.norm_1{i, j} = norm(S.data.diff{i, j}, 1);
    S.num_error.norm_1_rel{i, j} = S.num_error.norm_1{i, j}/norm(S.data.Ai, 1);

    if ~mem_save

      S.num_error.norm_inf{i, j} = norm(S.data.diff{i, j}, 'inf'); 
      temp = norm(S.data.Ai, 'inf');
      S.num_error.norm_inf_rel{i, j} = S.num_error.norm_inf{i, j}/temp;

      if size(S.data.Ai, 1) == size(S.data.Ai, 2)
        B = S.data.Ai_approx{i, j};
        C = S.data.Ai;
        C(abs(B > 0)) = 1;

        S.data.Ai_s = S.data.Ai .* C;
        S.data.diff_s = abs(S.data.Ai_s - S.data.Ai_approx);
        S.num_error.norm_fro_s{i, j} = norm(S.data.diff, 'fro');
        S.num_error.norm_fro_rel_s{i, j} = norm(S.data.diff, 'fro')/norm(S.data.Ai, 'fro');
        S.num_error.norm_1_s{i, j} = norm(S.data.diff, 1);
        S.num_error.norm_1_rel_s{i, j} = norm(S.data.diff, 1)/norm(S.data.Ai, 1);
        S.num_error.norm_inf_s{i, j} = norm(S.data.diff, 'inf');
        S.num_error.norm_inf_rel_s{i, j} = norm(S.data.diff, 'inf')/norm(S.data.Ai, 'inf');

        S.data.Ai_s{i} = [];
      end
    end
  end
end
