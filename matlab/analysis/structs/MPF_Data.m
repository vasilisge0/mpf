classdef MPF_Data < handle
  properties
    m_A;
    n_A;
    nz_A;
    diff;
    diff_s;
    Ai;
    Ai_approx;
  end

  methods
    function data = MPF_Data()
      data.m_A = [];
      data.n_A = [];
      data.nz_A = [];
      data.diff = [];
      data.diff_s = [];
      data.Ai = [];
      data.Ai_approx = [];
    end
  end
end
