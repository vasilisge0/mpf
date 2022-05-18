classdef MPF_Query < handle
  properties
    type;
    range;
    n_logs;
    n_range_queries;
  end

  methods
    function num_error = MPF_Runtime()
      num_error.norm_fro = [];
      num_error.norm_fro_rel = [];
      num_error.norm_1 = [];
      num_error.norm_1_rel = [];
      num_error.norm_inf = [];
      num_error.norm_inf_rel = [];
    end
  end
end
