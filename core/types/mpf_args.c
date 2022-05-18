#include "mpf.h"

void mpf_args_init
(
  MPF_Args *args
)
{
  args->n_total = 10;
  args->n_inner_solve = 0;
  args->n_outer_solve = 0;
  args->n_probe = 0;
}

void mpf_args_printout
(
  MPF_Args *args
)
{
  printf("total: %d\n", args->n_total);
  printf("outer_solve: %d\n", args->n_outer_solve);
  printf("inner_solve: %d\n", args->n_inner_solve);
  printf("probe: %d\n", args->n_probe);
  printf("n_args+n_args_outer+n_args_inner+n_args_probe+3: %d\n",
    args->n_total+args->n_outer_solve+args->n_inner_solve+args->n_probe+3);
}
