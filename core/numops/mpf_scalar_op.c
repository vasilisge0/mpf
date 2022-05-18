#include "mpf.h"

/*=============================================*/
/* == complex scalar manipulation functions == */
/*=============================================*/

MPF_ComplexDouble mpf_scalar_z_set
(
  const double variable_real
)
{
  MPF_ComplexDouble variable_complex;
  variable_complex.real = variable_real;
  variable_complex.imag = 0.0;
  return variable_complex;
}

MPF_ComplexDouble mpf_scalar_z_init
(
  const double real_value,
  const double imag_value
)
{
  MPF_ComplexDouble gamma;
  gamma.real = real_value;
  gamma.imag = imag_value;
  return gamma;
}

MPF_ComplexDouble mpf_scalar_z_add
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
)
{
  MPF_ComplexDouble gamma;
  gamma.real = alpha.real + beta.real;
  gamma.imag = alpha.imag + beta.imag;
  return gamma;
}

MPF_ComplexDouble mpf_scalar_z_divide
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
)
{
  MPF_ComplexDouble gamma;
  gamma.real = (alpha.real*beta.real + alpha.imag*beta.imag)
             / (beta.real*beta.real + beta.imag*beta.imag);
  gamma.imag = (alpha.imag*beta.real - alpha.real*beta.imag)
             / (beta.real*beta.real + beta.imag*beta.imag);
  return gamma;
}

MPF_ComplexDouble mpf_scalar_z_multiply
(
  const MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
)
{
  MPF_ComplexDouble gamma;
  gamma.real = alpha.real*beta.real - alpha.imag*beta.imag;
  gamma.imag = alpha.real*beta.imag + alpha.imag*beta.real;
  return gamma;
}

MPF_ComplexDouble mpf_scalar_z_normalize
(
  MPF_ComplexDouble alpha,
  const double beta
)
{
  alpha.real = alpha.real / beta;
  alpha.imag = alpha.imag / beta;
  return alpha;
}

MPF_ComplexDouble mpf_scalar_z_subtract
(
  MPF_ComplexDouble alpha,
  const MPF_ComplexDouble beta
)
{
  alpha.real = alpha.real - beta.real;
  alpha.imag = alpha.imag - beta.imag;
  return alpha;
}

MPF_ComplexDouble mpf_scalar_z_invert_sign
(
  MPF_ComplexDouble alpha
)
{
  alpha.real = -alpha.real;
  alpha.imag = -alpha.imag;
  return alpha;
}

MPF_Complex mpf_scalar_c_init
(
  const float real_value,
  const float imag_value
)
{
  MPF_Complex gamma;
  gamma.real = real_value;
  gamma.imag = imag_value;
  return gamma;
}

MPF_Complex mpf_scalar_c_add
(
  const MPF_Complex alpha,
  const MPF_Complex beta
)
{
  MPF_Complex gamma;
  gamma.real = alpha.real + beta.real;
  gamma.imag = alpha.imag + beta.imag;
  return gamma;
}

MPF_Complex mpf_scalar_c_divide
(
  const MPF_Complex alpha,
  const MPF_Complex beta
)
{
  MPF_Complex gamma;
  gamma.real = (alpha.real*beta.real + alpha.imag*beta.imag)
             / (beta.real*beta.real + beta.imag*beta.imag);
  gamma.imag = (alpha.imag*beta.real - alpha.real*beta.imag)
             / (beta.real*beta.real + beta.imag*beta.imag);
  return gamma;
}

MPF_Complex mpf_scalar_c_multiply
(
  const MPF_Complex alpha,
  const MPF_Complex beta
)
{
  MPF_Complex gamma;
  gamma.real = alpha.real*beta.real - alpha.imag*beta.imag;
  gamma.imag = alpha.real*beta.imag + alpha.imag*beta.real;
  return gamma;
}

MPF_Complex mpf_scalar_c_normalize
(
  MPF_Complex alpha,
  const float beta
)
{
  alpha.real = alpha.real / beta;
  alpha.imag = alpha.imag / beta;
  return alpha;
}

MPF_Complex mpf_scalar_c_subtract
(
  MPF_Complex alpha,
  const MPF_Complex beta
)
{
  alpha.real = alpha.real - beta.real;
  alpha.imag = alpha.imag - beta.imag;
  return alpha;
}

MPF_Complex mpf_scalar_c_invert_sign
(
  MPF_Complex alpha
)
{
  alpha.real = -alpha.real;
  alpha.imag = -alpha.imag;
  return alpha;
}

MPF_Int mpf_i_max
(
  const MPF_Int alpha,
  const MPF_Int beta
)
{
  return (alpha > beta) ? alpha : beta;
}

MPF_Int mpf_i_min
(
  const MPF_Int alpha,
  const MPF_Int beta
)
{
  return (alpha > beta) ? beta : alpha;
}

void mpf_diag_fA_average
(
  MPF_Context *context
)
{
  if (context->data_type == MPF_REAL)
  {
    mpf_dscal(context->A.m*context->solver.blk_fA, 1.0/context->probe.iterations,
      (double*)context->diag_fA.data, 1);
  }
  else if (context->data_type == MPF_COMPLEX)
  {
    MPF_ComplexDouble z = mpf_scalar_z_init(1.0/context->probe.iterations, 0.0);
    mpf_zscal(context->A.m*context->solver.blk_fA, &z,
      (MPF_ComplexDouble*)context->diag_fA.data, 1);
  }
}
