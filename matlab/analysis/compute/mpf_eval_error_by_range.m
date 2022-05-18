function [] = mpf_eval_error_by_range(S, query_id, varargin)

  mem_save = false;
  if nargin == 3
    mem_save = varargin{1};
  end

  range = S.query.range{query_id, 1};

  if strcmp(S.query.type, 'stride')
    S.index.stride_id = range(1);
    for i = range(1):range(2)
      mpf_read_error_by_id(S, query_id, i);

      % computes error
      mpf_eval_error(S, i, mem_save);
    end
  end

end
