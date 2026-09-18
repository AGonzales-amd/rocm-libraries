/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "rocauxiliary_trevc_backtransform.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T>
rocblas_status rocsolver_trevc_backtransform_impl(rocblas_handle handle,
                                                  const rocblas_side side,
                                                  const rocblas_int n,
                                                  const T* T_mat,
                                                  const rocblas_int ldt,
                                                  T* VL,
                                                  const rocblas_int ldvl,
                                                  T* VR,
                                                  const rocblas_int ldvr,
                                                  const rocblas_int mm)
try
{
    ROCSOLVER_ENTER_TOP("trevc_backtransform", "--side", side, "-n", n, "--ldt", ldt, "--ldvl",
                        ldvl, "--ldvr", ldvr, "--mm", mm);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trevc_backtransform_argCheck(handle, side, n, T_mat, ldt, VL,
                                                               ldvl, VR, ldvr, mm);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftT = 0;
    rocblas_stride shiftVL = 0;
    rocblas_stride shiftVR = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideT = 0;
    rocblas_stride strideVL = 0;
    rocblas_stride strideVR = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes
    size_t size_work;
    rocsolver_trevc_backtransform_getMemorySize<false, T>(side, n, mm, batch_count, &size_work);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work);

    // memory workspace allocation
    void* work;
    rocblas_device_malloc mem(handle, size_work);

    if(!mem)
        return rocblas_status_memory_error;

    work = mem[0];

    // execution
    return rocsolver_trevc_backtransform_template<T>(handle, side, n, T_mat, shiftT, ldt, strideT,
                                                     VL, shiftVL, ldvl, strideVL, VR, shiftVR,
                                                     ldvr, strideVR, mm, batch_count, work);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

/*! @{
    \brief TREVC_BACKTRANSFORM computes left and/or right eigenvectors of an upper
    triangular (complex) or upper quasi-triangular (real) matrix T.

    \details
    This is the back-transform step of eigenvector computation. Given eigenvectors
    of T, this routine multiplies them by the orthogonal/unitary matrix Q (stored
    in VL/VR on entry) when computing eigenvectors of the original matrix A = Q T Q^H.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        #rocblas_side.\\n
                Specifies whether to compute left, right, or both sets of eigenvectors.
    @param[in]
    n           rocblas_int. n >= 0.\\n
                The order of the matrix T.
    @param[in]
    T           pointer to type. Array on the GPU of dimension ldt*n.\\n
                The upper triangular (complex) or upper quasi-triangular (real) matrix.
    @param[in]
    ldt         rocblas_int. ldt >= max(1, n).\\n
                Leading dimension of T.
    @param[inout]
    VL          pointer to type. Array on the GPU of dimension ldvl*mm.\\n
                On exit (if side is left or both), contains the computed left eigenvectors.
                Not referenced if side is right only.
    @param[in]
    ldvl        rocblas_int. ldvl >= max(1, n) if side is left or both.\\n
                Leading dimension of VL.
    @param[inout]
    VR          pointer to type. Array on the GPU of dimension ldvr*mm.\\n
                On exit (if side is right or both), contains the computed right eigenvectors.
                Not referenced if side is left only.
    @param[in]
    ldvr        rocblas_int. ldvr >= max(1, n) if side is right or both.\\n
                Leading dimension of VR.
    @param[in]
    mm          rocblas_int. mm >= 0.\\n
                The number of columns in VL and/or VR.
    ********************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_strevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               const float* T,
                                                               const rocblas_int ldt,
                                                               float* VL,
                                                               const rocblas_int ldvl,
                                                               float* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<float>(handle, side, n, T, ldt, VL, ldvl,
                                                                VR, ldvr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dtrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               const double* T,
                                                               const rocblas_int ldt,
                                                               double* VL,
                                                               const rocblas_int ldvl,
                                                               double* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<double>(handle, side, n, T, ldt, VL, ldvl,
                                                                 VR, ldvr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ctrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               const rocblas_float_complex* T,
                                                               const rocblas_int ldt,
                                                               rocblas_float_complex* VL,
                                                               const rocblas_int ldvl,
                                                               rocblas_float_complex* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<rocblas_float_complex>(
        handle, side, n, T, ldt, VL, ldvl, VR, ldvr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ztrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               const rocblas_double_complex* T,
                                                               const rocblas_int ldt,
                                                               rocblas_double_complex* VL,
                                                               const rocblas_int ldvl,
                                                               rocblas_double_complex* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<rocblas_double_complex>(
        handle, side, n, T, ldt, VL, ldvl, VR, ldvr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}
//! @}

} // extern C
