classdef MPF_Fig < handle
  properties
    f_runtimes;
    f_bar;
    f_nrhs;
    f_perf;
    f_speedup;
    f_error;
    f_error_entrywise;
  end

  methods
    function fig = MPF_Fig()
      fig.f_runtimes = [];
      fig.f_bar = [];
      fig.f_nrhs = [];
      fig.f_perf = [];
      fig.f_speedup = [];
      fig.f_error = [];
      f_error_entrywise = [];
    end
  end
end
