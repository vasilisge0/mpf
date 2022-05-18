classdef MPF_Error < handle
  properties
    norm_fro;
    norm_fro_rel;
    norm_1;
    norm_1_rel;
    norm_inf;
    norm_inf_rel;

    norm_fro_s;
    norm_fro_rel_s;
    norm_1_s;
    norm_1_rel_s;
    norm_inf_s;
    norm_inf_rel_s;

    entrywise;
    entrywise_rel;
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
