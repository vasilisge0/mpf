
void mp_memory_outer_get
(
  MPContext *context
)
{
  MPInt blk = context->blk_solver;
  switch(context->solver_outer_type)
  {
    case MP_CUDA_BATCH:
    {
      if (context->data_type == MP_REAL)
      {
        context->bytes_outer =
          sizeof(double)*(context->m_A*context->blk*2);
        context->bytes_cuda_outer =
          sizeof(double)*(context->m_A*context->blk*2);
        context->bytes_fA_data = sizeof(double)*context->m_A*context->blk_fA;
      }
      else if (context->data_type == MP_COMPLEX)
      {
        context->bytes_outer = sizeof(MPComplexDouble)*
          (context->m_A*context->blk*2);
        context->bytes_cuda_outer = sizeof(cuDoubleComplex)*
          (context->m_A*context->blk*2);
        context->bytes_fA_data = sizeof(MPComplexDouble)*context->m_A
          *context->blk_fA;
      }
      break;
    }
    default:
      break;
  }
}
