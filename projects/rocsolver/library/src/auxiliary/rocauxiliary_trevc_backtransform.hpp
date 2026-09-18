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

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
void rocsolver_trevc_backtransform_getMemorySize(const rocblas_side side,
                                                 const rocblas_int n,
                                                 const rocblas_int mm,
                                                 const rocblas_int batch_count,
                                                 size_t* size_work)
{
    *size_work = 0;

    if(n == 0 || batch_count == 0)
        return;

    // TODO: determine actual workspace requirements once algorithm is implemented
}

template <typename T, typename U>
rocblas_status rocsolver_trevc_backtransform_argCheck(rocblas_handle handle,
                                                      const rocblas_side side,
                                                      const rocblas_int n,
                                                      U T_mat,
                                                      const rocblas_int ldt,
                                                      T* VL,
                                                      const rocblas_int ldvl,
                                                      T* VR,
                                                      const rocblas_int ldvr,
                                                      const rocblas_int mm,
                                                      const rocblas_int batch_count = 1)
{
    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right && side != rocblas_side_both)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || ldt < std::max(1, n) || mm < 0 || batch_count < 0)
        return rocblas_status_invalid_size;
    if(left && ldvl < std::max(1, n))
        return rocblas_status_invalid_size;
    if(right && ldvr < std::max(1, n))
        return rocblas_status_invalid_size;

    // skip pointer checks if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if(n && !T_mat)
        return rocblas_status_invalid_pointer;
    if(left && mm && !VL)
        return rocblas_status_invalid_pointer;
    if(right && mm && !VR)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status rocsolver_trevc_backtransform_template(rocblas_handle handle,
                                                      const rocblas_side side,
                                                      const rocblas_int n,
                                                      U T_mat,
                                                      const rocblas_stride shiftT,
                                                      const rocblas_int ldt,
                                                      const rocblas_stride strideT,
                                                      T* VL,
                                                      const rocblas_stride shiftVL,
                                                      const rocblas_int ldvl,
                                                      const rocblas_stride strideVL,
                                                      T* VR,
                                                      const rocblas_stride shiftVR,
                                                      const rocblas_int ldvr,
                                                      const rocblas_stride strideVR,
                                                      const rocblas_int mm,
                                                      const rocblas_int batch_count,
                                                      void* work)
{
    ROCSOLVER_ENTER("trevc_backtransform", "side:", side, "n:", n, "shiftT:", shiftT, "ldt:", ldt,
                    "shiftVL:", shiftVL, "ldvl:", ldvl, "shiftVR:", shiftVR, "ldvr:", ldvr, "mm:",
                    mm, "bc:", batch_count);

    if(n == 0 || batch_count == 0)
        return rocblas_status_success;

    // TODO: implement TREVC3 backtransform algorithm
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
