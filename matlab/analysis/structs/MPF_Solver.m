classdef MPF_Solver < handle
  properties
    batch;
    framework;
    outer_type;
    inner_type;
    outer_nthreads;
    inner_nthreads;
    restarts;
    iterations;
    tolerance;
    defl;
    precond;
    blk_fA;
  end

  methods
    function solver = MPF_Solver()
      batch = [];
      framework = [];
      outer_type = [];
      inner_type = [];
      outer_nthreads = [];
      inner_nthreads = [];
      restarts = [];
      iterations = [];
      tolerance = [];
      blk_fA = [];
    end
  end

end
