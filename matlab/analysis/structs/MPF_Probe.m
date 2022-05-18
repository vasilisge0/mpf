classdef MPF_Probe < handle
  properties
    type;
    expand_degree;
    n_colors;
    n_levels;
    stride;
    blk_fA;
    max_blk_fA;
    n_max_B;
    m_P;
    n_P;
    nz_P;
  end

  methods
    function probe = MPF_Probe()
      type = [];
      expand_degree = [];
      n_colors = [];
      n_levels = [];
      stride = [];
      blk_fA = [];
      max_blk_fA = [];
      n_max_B = [];
      m_P = [];
      n_P = [];
      nz_P = [];
    end
  end
end
