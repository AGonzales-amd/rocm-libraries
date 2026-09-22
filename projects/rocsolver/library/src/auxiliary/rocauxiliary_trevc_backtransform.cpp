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
                                                  T* VL,
                                                  const rocblas_int ldvl,
                                                  T* VR,
                                                  const rocblas_int ldvr,
                                                  const T* QL,
                                                  const rocblas_int ldql,
                                                  const T* QR,
                                                  const rocblas_int ldqr,
                                                  const rocblas_int mm)
try
{
    ROCSOLVER_ENTER_TOP("trevc_backtransform", "--side", side, "-n", n, "--ldvl", ldvl, "--ldvr",
                        ldvr, "--ldql", ldql, "--ldqr", ldqr, "--mm", mm);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_trevc_backtransform_argCheck<T>(handle, side, n, VL, ldvl, VR, ldvr,
                                                               QL, ldql, QR, ldqr, mm);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_stride shiftVL = 0;
    rocblas_stride shiftVR = 0;
    rocblas_stride shiftQL = 0;
    rocblas_stride shiftQR = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideVL = 0;
    rocblas_stride strideVR = 0;
    rocblas_stride strideQL = 0;
    rocblas_stride strideQR = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes
    size_t size_work, size_ips, size_workArr;
    rocsolver_trevc_backtransform_getMemorySize<false, T>(side, n, mm, batch_count, &size_work,
                                                          &size_ips, &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(handle, size_work, size_ips, size_workArr);

    // memory workspace allocation
    rocblas_device_malloc mem(handle, size_work, size_ips, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    T* work = (T*)mem[0];
    rocblas_int* ips = (rocblas_int*)mem[1];
    T** workArr = (T**)mem[2];

    // execution
    return rocsolver_trevc_backtransform_template<T>(handle, side, n, VL, shiftVL, ldvl, strideVL,
                                                     VR, shiftVR, ldvr, strideVR, QL, shiftQL,
                                                     ldql, strideQL, QR, shiftQR, ldqr, strideQR,
                                                     mm, batch_count, work, ips, workArr);
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
    \brief The TREVC_BACKTRANSFORM applies the back-transform step of eigenvector computation
    from a Schur-form factorization.

    \details
    Given left and/or right eigenvector matrices VL and VR (of the Schur form T), and
    the orthogonal/unitary matrices QL and QR, this routine computes the eigenvectors
    of the original matrix A = QL * T * QR^H by updating VL := QL * VL and VR := QR * VR.

    @param[in]
    handle      rocblas_handle.
    @param[in]
    side        #rocblas_side.
                Specifies whether to compute left, right, or both sets of eigenvectors.
    @param[in]
    n           rocblas_int. n >= 0.
                The order of the matrices.
    @param[inout]
    VL          pointer to type. Array on the GPU of dimension ldvl*mm.
                On entry, the left eigenvectors of the Schur form T. On exit, the left
                eigenvectors of the original matrix A. Not referenced if side is right only.
    @param[in]
    ldvl        rocblas_int. ldvl >= max(1, n) if side is left or both.
                Leading dimension of VL.
    @param[inout]
    VR          pointer to type. Array on the GPU of dimension ldvr*mm.
                On entry, the right eigenvectors of the Schur form T. On exit, the right
                eigenvectors of the original matrix A. Not referenced if side is left only.
    @param[in]
    ldvr        rocblas_int. ldvr >= max(1, n) if side is right or both.
                Leading dimension of VR.
    @param[in]
    QL          pointer to type. Array on the GPU of dimension ldql*n.
                The orthogonal/unitary matrix used to form the left back-transform. Not
                referenced if side is right only.
    @param[in]
    ldql        rocblas_int. ldql >= max(1, n) if side is left or both.
                Leading dimension of QL.
    @param[in]
    QR          pointer to type. Array on the GPU of dimension ldqr*n.
                The orthogonal/unitary matrix used to form the right back-transform. Not
                referenced if side is left only.
    @param[in]
    ldqr        rocblas_int. ldqr >= max(1, n) if side is right or both.
                Leading dimension of QR.
    @param[in]
    mm          rocblas_int. 0 <= mm <= n.
                The number of columns in VL, VR.
    ********************************************************************/
ROCSOLVER_EXPORT rocblas_status rocsolver_strevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               float* VL,
                                                               const rocblas_int ldvl,
                                                               float* VR,
                                                               const rocblas_int ldvr,
                                                               const float* QL,
                                                               const rocblas_int ldql,
                                                               const float* QR,
                                                               const rocblas_int ldqr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<float>(handle, side, n, VL, ldvl, VR, ldvr,
                                                                QL, ldql, QR, ldqr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dtrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               double* VL,
                                                               const rocblas_int ldvl,
                                                               double* VR,
                                                               const rocblas_int ldvr,
                                                               const double* QL,
                                                               const rocblas_int ldql,
                                                               const double* QR,
                                                               const rocblas_int ldqr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<double>(handle, side, n, VL, ldvl, VR,
                                                                 ldvr, QL, ldql, QR, ldqr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ctrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               rocblas_float_complex* VL,
                                                               const rocblas_int ldvl,
                                                               rocblas_float_complex* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_float_complex* QL,
                                                               const rocblas_int ldql,
                                                               const rocblas_float_complex* QR,
                                                               const rocblas_int ldqr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<rocblas_float_complex>(
        handle, side, n, VL, ldvl, VR, ldvr, QL, ldql, QR, ldqr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_ztrevc_backtransform(rocblas_handle handle,
                                                               const rocblas_side side,
                                                               const rocblas_int n,
                                                               rocblas_double_complex* VL,
                                                               const rocblas_int ldvl,
                                                               rocblas_double_complex* VR,
                                                               const rocblas_int ldvr,
                                                               const rocblas_double_complex* QL,
                                                               const rocblas_int ldql,
                                                               const rocblas_double_complex* QR,
                                                               const rocblas_int ldqr,
                                                               const rocblas_int mm)
{
#if defined(ROCSOLVER_ENABLE_TREVC3)
    return rocsolver::rocsolver_trevc_backtransform_impl<rocblas_double_complex>(
        handle, side, n, VL, ldvl, VR, ldvr, QL, ldql, QR, ldqr, mm);
#else
    return rocblas_status_not_implemented;
#endif
}
//! @}

} // extern C
