classdef MPF_Filename < handle
  properties
    file_log = [];
    data_prefix = [];
    approx_prefix = [];
    exact_inverse = [];
    path_log = [];
    path_plots = [];
    path_approx = [];
    path_exact = [];
    title = [];
    caller = [];
    A = [];
    Ai = [];
    output = [];
    meta = [];
  end

  methods
    function filename = MPF_Filename()
      filename.caller = [];
      filename.A = [];
      filename.output = [];
      filename.meta = [];
    end
  end
end
