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

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) trevc_bt_scale_complex_kernel(const I n,
                                                                     const I mm,
                                                                     U __restrict__ VV,
                                                                     const rocblas_stride shiftV,
                                                                     const I ldv,
                                                                     const rocblas_stride strideV,
                                                                     T* __restrict__ WW,
                                                                     const rocblas_stride ldw,
                                                                     const rocblas_stride strideW,
                                                                     const I batch_count)
{
    using S = decltype(std::real(T{}));

    const I row_start = threadIdx.x;
    const I col_start = blockIdx.x;
    const I bid_start = blockIdx.z;

    const I row_inc = blockDim.x;
    const I col_inc = gridDim.x;
    const I bid_inc = gridDim.z;

    __shared__ S sval[BS1];

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* __restrict__ V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
        const T* __restrict__ W = load_ptr_batch<T>(WW, bid, 0, strideW);

        for(I col = col_start; col < mm; col += col_inc)
        {
            iamax<BS1>(row_start, n, &W[idx2D(0, col, ldw)], 1, sval);
            __syncthreads();

            const auto scale = (T)1 / sval[0];
            for(I row = row_start; row < n; row += row_inc)
            {
                V[idx2D(row, col, ldv)] = scale * W[idx2D(row, col, ldw)];
            }
        }
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) trevc_bt_get_scale_real_kernel(
                const I n,
                                                                     const I mm,
                                                                     U __restrict__ TTS,
                                                                     const rocblas_stride shiftT,
                                                                     const I ldt,
                                                                     const rocblas_stride strideT,
                                                                     I* __restrict__ ips,
                                                                     const I batch_count)
{
    const I col_start = blockIdx.x * blockDim.x + threadIdx.x;
    const I bid_start = blockIdx.z;

    const I col_inc = blockDim.x * gridDim.x;
    const I bid_inc = gridDim.z;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        const T* __restrict__ TS = load_ptr_batch<T>(TTS, bid, shiftT, strideT);

        for(I col = col_start; col < mm; col += col_inc)
        {
            // 0: no pair, -1: first pair, 1: second pair
            int ip = 0;
            if(col < mm - 1 && TS[idx2D(col + 1, col, ldt)] != (T)0)
                ip = -1;
            else if(col > 0 && TS[idx2D(col, col - 1, ldt)] != (T)0)
                ip = 1;

            ips[bid * mm + col] = ip;
        }
    }
}

template <typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(BS1) trevc_bt_scale_real_kernel(const I n,
                                                                     const I mm,
                                                                     U __restrict__ VV,
                                                                     const rocblas_stride shiftV,
                                                                     const I ldv,
                                                                     const rocblas_stride strideV,
                                                                     T* __restrict__ WW,
                                                                     const rocblas_stride ldw,
                                                                     const rocblas_stride strideW,
                                                                     const I* __restrict__ ips,
                                                                     const I batch_count)
{
    using S = decltype(std::real(T{}));

    const I row_start = threadIdx.x;
    const I col_start = blockIdx.x;
    const I bid_start = blockIdx.z;

    const I row_inc = blockDim.x;
    const I col_inc = gridDim.x;
    const I bid_inc = gridDim.z;

    __shared__ S sval[BS1];

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* __restrict__ V = load_ptr_batch<T>(VV, bid, shiftV, strideV);
        const T* __restrict__ W = load_ptr_batch<T>(WW, bid, 0, strideW);

        for(I col = col_start; col < mm; col += col_inc)
        {
            const auto ip = ips[bid * mm + col];
            if(ip == 0)
            {
                iamax<BS1>(row_start, n, &W[idx2D(0, col, ldw)], 1, sval);
            }
            else if(ip == -1)
            {   
                iamax<BS1>(row_start, n, &W[idx2D(0, col, ldw)], &W[idx2D(0, col + 1, ldw)], 1, sval);
            }
            else
            {
                continue;
            }
            __syncthreads();

            const auto scale = (T)1 / sval[0];
            for(I row = row_start; row < n; row += row_inc)
            {
                V[idx2D(row, col, ldv)] = scale * W[idx2D(row, col, ldw)];

                if(ip == -1)
                    V[idx2D(row, col + 1, ldv)] = scale * W[idx2D(row, col + 1, ldw)];
            }
        }
    }
}

// Host helper to launch trevc_bt_scale_kernel
template <typename T, typename I, typename U1, typename U2, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
void trevc_bt_scale(rocblas_handle handle,
                const I n,
                const I mm,
                U1 __restrict__ TS,
                const rocblas_stride shiftT,
                const I ldt,
                const rocblas_stride strideT,
                U2 __restrict__ V,
                const rocblas_stride shiftV,
                const I ldv,
                const rocblas_stride strideV,
                T* __restrict__ W,
                const rocblas_stride ldw,
                const rocblas_stride strideW,
                I* __restrict__ ips,
                const I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const auto& grid_limits = props->maxGridSize;

    I bx = std::min<I>(mm, grid_limits[0]);
    I bz = std::min<I>(batch_count, grid_limits[2]);
    ROCSOLVER_LAUNCH_KERNEL((trevc_bt_scale_complex_kernel), dim3(bx, 1, bz), dim3(BS1), 0,
                            stream, n, mm, V, shiftV, ldv, strideV, W, ldw, strideW,
                            batch_count);
}

template <typename T, typename I, typename U1, typename U2, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
void trevc_bt_scale(rocblas_handle handle,
                const I n,
                const I mm,
                U1 __restrict__ TS,
                const rocblas_stride shiftT,
                const I ldt,
                const rocblas_stride strideT,
                U2 __restrict__ V,
                const rocblas_stride shiftV,
                const I ldv,
                const rocblas_stride strideV,
                T* __restrict__ W,
                const rocblas_stride ldw,
                const rocblas_stride strideW,
                I* __restrict__ ips,
                const I batch_count)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    const auto& grid_limits = props->maxGridSize;
    {
        I bx = std::min<I>((mm + BS1 - 1) / BS1, grid_limits[0]);
        I bz = std::min<I>(batch_count, grid_limits[2]);
        ROCSOLVER_LAUNCH_KERNEL((trevc_bt_get_scale_real_kernel<T>), dim3(bx, 1, bz), dim3(BS1), 0,
                                stream, n, mm, TS, shiftT, ldt, strideT, ips,
                                batch_count);
    }
    {
        I bx = std::min<I>(mm, grid_limits[0]);
        I bz = std::min<I>(batch_count, grid_limits[2]);
        ROCSOLVER_LAUNCH_KERNEL((trevc_bt_scale_real_kernel), dim3(bx, 1, bz), dim3(BS1), 0,
                                stream, n, mm, V, shiftV, ldv, strideV, W, ldw, strideW, ips,
                                batch_count);
    }
}

template <bool BATCHED, typename T>
void rocsolver_trevc_backtransform_getMemorySize(const rocblas_side side,
                                                 const rocblas_int n,
                                                 const rocblas_int mm,
                                                 const rocblas_int batch_count,
                                                 size_t* size_work,
                                                 size_t* size_ips,
                                                 size_t* size_workArr)
{
    *size_work = 0;
    *size_ips = 0;
    *size_workArr = 0;

    if(n == 0 || mm == 0 || batch_count == 0)
        return;

    *size_work = sizeof(T) * n * mm * batch_count;
    if constexpr(!rocblas_is_complex<T>)
        *size_ips = sizeof(rocblas_int) * mm * batch_count;
    *size_workArr = sizeof(T*) * batch_count;
}

template <typename T, typename U1, typename U2>
rocblas_status rocsolver_trevc_backtransform_argCheck(rocblas_handle handle,
                                                      const rocblas_side side,
                                                      const rocblas_int n,
                                                      const rocblas_int mm,
                                                      U1 TS,
                                                      const rocblas_int ldt,
                                                      U1 QL,
                                                      const rocblas_int ldql,
                                                      U1 QR,
                                                      const rocblas_int ldqr,
                                                      U2 VL,
                                                      const rocblas_int ldvl,
                                                      U2 VR,
                                                      const rocblas_int ldvr,
                                                      const rocblas_int batch_count = 1)
{
    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right && side != rocblas_side_both)
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || mm < 0 || batch_count < 0 || (ldt < n) || (left && ldvl < n) || (right && ldvr < n) || (left && ldql < n) || (right && ldqr < n))
        return rocblas_status_invalid_size;

    // skip pointer checks if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !TS) || (left && mm && !VL) || (right && mm && !VR) || (left && mm && !QL) || (right && mm && !QR))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename U1, typename U2>
rocblas_status rocsolver_trevc_backtransform_template(rocblas_handle handle,
                                                      const rocblas_side side,
                                                      const rocblas_int n,
                                                      const rocblas_int mm,
                                                      U1 TS,
                                                      const rocblas_stride shiftTS,
                                                      const rocblas_int ldt,
                                                      const rocblas_stride strideTS,
                                                      U1 QL,
                                                      const rocblas_stride shiftQL,
                                                      const rocblas_int ldql,
                                                      const rocblas_stride strideQL,
                                                      U1 QR,
                                                      const rocblas_stride shiftQR,
                                                      const rocblas_int ldqr,
                                                      const rocblas_stride strideQR,
                                                      U2 VL,
                                                      const rocblas_stride shiftVL,
                                                      const rocblas_int ldvl,
                                                      const rocblas_stride strideVL,
                                                      U2 VR,
                                                      const rocblas_stride shiftVR,
                                                      const rocblas_int ldvr,
                                                      const rocblas_stride strideVR,
                                                      const rocblas_int batch_count,
                                                      T* work,
                                                      rocblas_int* ips,
                                                      T** workArr)
{
    ROCSOLVER_ENTER("trevc_backtransform", "side:", side, "n:", n, "mm:", mm, "shiftTS:", shiftTS,
                    "ldt:", ldt, "shiftQL:", shiftQL, "ldql:", ldql, "shiftQR:", shiftQR, "ldqr:",
                    ldqr, "shiftVL:", shiftVL, "ldvl:", ldvl, "shiftVR:", shiftVR, "ldvr:", ldvr,
                    "bc:", batch_count);

    if(n == 0 || mm == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // everything must be executed with scalars on the host
    rocblas_pointer_mode_saver saver(handle, rocblas_pointer_mode_host);

    const T zero = (T)0.0;
    const T one = (T)1.0;

    rocblas_stride strideW = n * mm;

    if(left)
    {
        // work = QL * VL
        rocsolver_gemm<T>(handle, rocblas_operation_none, rocblas_operation_none, n, mm,
                            n, &one, QL, shiftQL, ldql, strideQL, cast2constType<T>(VL), shiftVL, ldvl, strideVL,
                            &zero, work, 0, n, strideW, batch_count, workArr);

        // scale VL
        trevc_bt_scale<T>(handle, n, mm, TS, shiftTS, ldt, strideTS, VL, shiftVL, ldvl, strideVL, work, n, strideW, ips, batch_count);
    }

    if(right)
    {
        // work = QR * VR
        rocsolver_gemm<T>(handle, rocblas_operation_none, rocblas_operation_none, n, mm,
                            n, &one, QR, shiftQR, ldqr, strideQR, cast2constType<T>(VR), shiftVR, ldvr, strideVR,
                            &zero, work, 0, n, strideW, batch_count, workArr);

        // scale VR
        trevc_bt_scale<T>(handle, n, mm, TS, shiftTS, ldt, strideTS, VR, shiftVR, ldvr, strideVR, work, n, strideW, ips, batch_count);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
