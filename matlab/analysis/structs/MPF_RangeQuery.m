classdef MPF_RangeQuery < handle
  properties
    root_path;
    exact_path;
    title;
    type;
    range;
  end

  methods
    function this = MPF_RangeQuery(root_path, exact_path, title, type, range)
      this.root_path = root_path;
      this.exact_path = exact_path;
      this.title = title;
      this.type = type;
      this.range = range;
    end
  end

end
