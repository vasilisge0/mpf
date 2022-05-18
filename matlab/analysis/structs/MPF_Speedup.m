classdef MPF_Speedup < handle
  properties
    solver_total;
    probe_total;
    total;
  end

  methods
    function speedup = MPF_Speedup(speedup)
      speedup.probe_total = [];
      speedup.solver_total = [];
      speedup.total = [];
    end
  end
end
