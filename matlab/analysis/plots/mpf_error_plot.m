function [] = mpf_num_error_plot(S, xname, range_query_id, varargin)

  save_mem = false;
  if nargin == 4
    save_mem = varargin{1};
  end

  legend_names = S.legend_names{:};
  dataname = S.title;

  if xname == "nrhs"
    x = S.n_max_B(:, 1);
    xname = "Number of right-hand sides";

  elseif xname == "exp"
    p = size(S.runtime.probe_total, 1);
    x = (2.^(1:p).');
    xname = "Expansion exponent k";
  end

  S.fig.f_error(1) = figure();
  larray = [];
  for i = 1:S.n_logs
    semilogy(x, [S.num_error.norm_fro{:, i}], '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('num_error (frobenious norm)');
  title(strcat(dataname, ' / num\_error (fro-norm)'));
  legend(legend_names, 'Location', 'NorthEast');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_error(2) = figure();
  for i = 1:S.n_logs
    semilogy(x, [S.num_error.norm_fro_rel{:, i}], '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('relative num_error (frobenious norm)');
  title(strcat(dataname, ' / relative num\_error (fro-norm)'));
  legend(legend_names, 'Location', 'NorthEast');
  hold off;
  grid on;
  set(gca,'fontsize',12)

  S.fig.f_error(3) = figure();
  for i = 1:S.n_logs
    semilogy(x, [S.num_error.norm_1{:, i}], '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('num_error (1-norm)');
  title(strcat(dataname, ' / num\_error (1-norm)'));
  legend(legend_names, 'Location', 'NorthEast');
  hold off;
  grid on;
  set(gca,'fontsize', 12)

  S.fig.f_error(4) = figure();
  for i = 1:S.n_logs
    semilogy(x, [S.num_error.norm_1_rel{:, i}], '-x', 'LineWidth', 2);
    hold on;
  end
  xlabel(xname);
  ylabel('relative num_error (1-norm)');
  title(strcat(dataname, ' / relative num_error (1-norm)'));
  legend(legend_names, 'Location', 'NorthEast');
  hold off;
  grid on;
  set(gca,'fontsize', 12)

  for i = 1:S.n_logs
    for j = 1:length(S.probe.stride)
      if size(S.data.diff{j, i}, 1) ~= size(S.data.diff{j, i}, 2) && (size(S.data.diff{j, i}, 2) == 1)
        g(j, i) = figure();
        semilogy(S.data.diff{j, i}./S.data.Ai, '*');
        grid on;
      end
    end
  end

  if ~save_mem
    S.fig.f_error(5) = figure();
    for i = 1:S.n_logs
      semilogy(x, [S.num_error.norm_inf{:, i}], '-x', 'LineWidth', 2);
      hold on;
    end
    xlabel(xname);
    ylabel('num_error (inf-norm)');
    title(strcat(dataname, ' / num\_error (inf-norm)'));
    legend(legend_names, 'Location', 'NorthEast');
    hold off;
    grid on;
    set(gca,'fontsize', 12)

    S.fig.f_error(6) = figure();
    for i = 1:S.n_logs
      semilogy(x, [S.num_error.norm_inf_rel{:, i}], '-x', 'LineWidth', 2);
      hold on;
    end
    xlabel(xname);
    ylabel('relative num_error (inf-norm)');
    title(strcat(dataname, ' / relative num\_error (inf-norm)'));
    legend(legend_names, 'Location', 'NorthEast');
    hold off;
    grid on;
    set(gca,'fontsize', 12)
  end

  for i = 1:S.n_logs
    for j = 1:length(S.probe.stride)

      if size(S.data.diff{j, i}, 1) ~= size(S.data.diff{j, i}, 2) && (size(S.data.diff{j, i}, 2) == 1)
        %% the next two lines take up lots of memory
        %S.num_error.entrywise{j, i} = S.data.diff{j, i};
        %S.num_error.entrywise_rel{j, i} = S.data.diff{j, i}./S.data.Ai;

        S.fig.f_error_entrywise{j, i} = figure();
        plot(S.data.diff{j, i}./S.data.Ai, '*');
        grid on;
        xlabel('entry index');
        ylabel('relative entrywise num_error (inf-norm)');
        title(strcat('stride=', string(S.probe.stride(j, i)), ', level=', string(S.probe.n_levels(j, i))));
        set(gca,'fontsize', 12)
      end
    end
  end
  %else
    %g(j, i) = figure();
    %surf(S.data.diff{j, i}(1:150, 1:150));
    %grid on;
  %end

end
