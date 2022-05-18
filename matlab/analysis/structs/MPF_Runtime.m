classdef MPF_Runtime < handle
  properties
      probe_contract;
      probe_expand;
      probe_color;
      probe_other;
      probe_total;
      solver_pre_process;
      solver_alloc;
      solver_generate_rhs;
      solver_inner;
      solver_reconstruct;
      solver_post_process;
      solver_total;
  end

  methods
    function runtime = MPF_Runtime()
      runtime.probe_contract = [];
      runtime.probe_expand = [];
      runtime.probe_color = [];
      runtime.probe_other = [];
      runtime.probe_total = [];
      runtime.solver_pre_process = [];
      runtime.solver_alloc = [];
      runtime.solver_generate_rhs = [];
      runtime.solver_inner = [];
      runtime.solver_reconstruct = [];
      runtime.solver_post_process = [];
      runtime.solver_total = [];
    end
  end
end
