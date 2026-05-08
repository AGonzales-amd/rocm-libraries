/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/**
 * ------------------------------------------------------
 * Perform LU factorization with partial pivoting for a small m x n matrix.
 * The function executes in a single thread block per matrix.
 * ------------------------------------------------------
 *
 * Layout:
 *   - One thread per row: thread tx owns row tx.
 *   - All n columns of the owned row are held in registers: rA[NB].
 *   - Shared memory holds:
 *       col_sh[MAX_M]   : current column broadcast for pivot search
 *       pivrow_sh[NB]   : pivot row broadcast for rank-1 update
 *       sval[MAX_M]     : absolute values for parallel reduction
 *       sidx[MAX_M]     : row indices for parallel reduction
 *
 * NB     Number of columns (compile-time constant, = n).
 * MAX_M  Thread block size (= m, padded to the launch value).
 **/
template <int NB, int MAX_M, typename T, typename I, typename INFO, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(MAX_M)
    getf2_kernel_small(const I m,
                       U AA,
                       const rocblas_stride shiftA,
                       const I lda,
                       const rocblas_stride strideA,
                       I* ipivA,
                       const rocblas_stride shiftP,
                       const rocblas_stride strideP,
                       INFO* infoA,
                       const I batch_count,
                       const I offset,
                       I* permut_idx,
                       const rocblas_stride stridePI)
{
    using S = decltype(std::real(T{}));

    const I tx = hipThreadIdx_x;
    const I id = hipBlockIdx_z;

    if(id >= batch_count)
        return;

    T* A = load_ptr_batch<T>(AA, id, shiftA, strideA);
    I* ipiv = load_ptr_batch<I>(ipivA, id, shiftP, strideP);
    I* permut = (permut_idx != nullptr ? permut_idx + id * stridePI : nullptr);
    INFO* info = infoA + id;

    // shared memory layout:
    //   col_sh   [MAX_M] T  — current column for pivot search
    //   pivrow_sh[NB]    T  — pivot row for trailing update
    //   sval     [MAX_M] S  — abs values for reduction
    //   sidx     [MAX_M] I  — row indices for reduction
    extern __shared__ double lmem[];
    T* col_sh    = reinterpret_cast<T*>(lmem);
    T* pivrow_sh = col_sh + MAX_M;
    S* sval      = reinterpret_cast<S*>(pivrow_sh + NB);
    I* sidx      = reinterpret_cast<I*>(sval + MAX_M);

    // Each thread loads its row into registers.
    T rA[NB];
#pragma unroll NB
    for(I j = 0; j < NB; ++j)
        rA[j] = (tx < m) ? A[tx + j * lda] : T(0);

    I myrow = tx; // logical row (swapped lazily like getf2_small_kernel)
    I mypiv = tx + 1; // 1-based pivot index for this thread's final row
    INFO myinfo = 0;

    const I dim = (m < NB) ? m : NB; // number of pivot steps

#pragma unroll NB
    for(I k = 0; k < NB; ++k)
    {
        if(k >= dim)
            break;

        // ----------------------------------------------------------------
        // 1. Broadcast column k to shared memory for pivot search.
        // ----------------------------------------------------------------
        col_sh[tx] = rA[k];
        sval[tx]   = (tx >= k && tx < m) ? aabs<S>(rA[k]) : S(-1);
        sidx[tx]   = tx;
        __syncthreads();

        // ----------------------------------------------------------------
        // 2. Parallel reduction to find index of max absolute value in
        //    col_sh[k..m-1].
        // ----------------------------------------------------------------
        for(I stride = MAX_M / 2; stride > 0; stride /= 2)
        {
            if(tx < stride)
            {
                S v1 = sval[tx];
                S v2 = sval[tx + stride];
                I i1 = sidx[tx];
                I i2 = sidx[tx + stride];
                // prefer the larger absolute value; break ties by smaller index
                if(v1 < v2 || (v1 == v2 && i1 > i2))
                {
                    sval[tx] = v2;
                    sidx[tx] = i2;
                }
            }
            __syncthreads();
        }

        I pivot_idx = sidx[0]; // row index of the pivot element
        T pivot_val = col_sh[pivot_idx];

        // ----------------------------------------------------------------
        // 3. Check singularity and compute reciprocal of pivot.
        // ----------------------------------------------------------------
        if(pivot_val != T(0))
            pivot_val = T(1) / pivot_val;
        else if(myinfo == 0)
            myinfo = static_cast<INFO>(k + 1);

        // ----------------------------------------------------------------
        // 4. Lazy row swap (mirrors getf2_small_kernel):
        //    The thread holding pivot_idx swaps identities with thread k.
        // ----------------------------------------------------------------
        if(myrow == static_cast<I>(pivot_idx))
        {
            myrow = k;
            // share the pivot row (columns k+1..NB-1) into shared memory
#pragma unroll NB
            for(I j = k + 1; j < NB; ++j)
                pivrow_sh[j] = rA[j];
        }
        else if(myrow == k)
        {
            myrow = pivot_idx;
            mypiv = pivot_idx + 1;
            if(permut != nullptr && static_cast<I>(pivot_idx) != k)
                swap(permut[k], permut[pivot_idx]);
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // 5. Scale L column and update trailing submatrix in registers.
        // ----------------------------------------------------------------
        if(myrow > k && myrow < m)
        {
            rA[k] *= pivot_val;
#pragma unroll NB
            for(I j = k + 1; j < NB; ++j)
                rA[j] -= rA[k] * pivrow_sh[j];
        }
        __syncthreads();
    }

    // Write results back to global memory.
    if(myrow < dim)
        ipiv[myrow] = mypiv + offset;
    if(myrow == 0 && *info == 0 && myinfo > 0)
        *info = myinfo + offset;
    if(myrow < m)
    {
#pragma unroll NB
        for(I j = 0; j < NB; ++j)
            A[myrow + j * lda] = rA[j];
    }
}

/*************************************************************
    Launcher
*************************************************************/

template <typename T, typename I, typename INFO, typename U>
rocblas_status getf2_run_kernel_small(rocblas_handle handle,
                                      const I m,
                                      const I n,
                                      U A,
                                      const rocblas_stride shiftA,
                                      const I lda,
                                      const rocblas_stride strideA,
                                      I* ipiv,
                                      const rocblas_stride shiftP,
                                      const rocblas_stride strideP,
                                      INFO* info,
                                      const I batch_count,
                                      const I offset,
                                      I* permut_idx,
                                      const rocblas_stride stridePI)
{
    ROCSOLVER_ENTER("getf2_kernel_small", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda,
                    "bc:", batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // Round m up to the next power of two for the reduction, capped at 1024.
    I max_m = 1;
    while(max_m < m)
        max_m *= 2;

    size_t lmemsize = sizeof(T) * (max_m + n) + max_m * (sizeof(decltype(std::real(T{}))) + sizeof(I));

    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);
    if(lmemsize > props->sharedMemPerBlock)
        return rocblas_status_internal_error;

// Macro: launch the kernel with NB and MAX_M baked in at compile time.
#define RUN_GETF2_KERNEL_SMALL(NB, MAX_M)                                                     \
    ROCSOLVER_LAUNCH_KERNEL((getf2_kernel_small<NB, MAX_M, T, I, INFO, U>), grid, block,      \
                            lmemsize, stream, m, A, shiftA, lda, strideA, ipiv, shiftP,       \
                            strideP, info, batch_count, offset, permut_idx, stridePI)

    dim3 grid(1, 1, batch_count);
    dim3 block(max_m, 1, 1);

    // Dispatch over n (compile-time NB) and max_m (compile-time MAX_M).
    // NB runs 1..64 (same range as getf2_run_small).
    // MAX_M is the next power of two >= m, capped at 1024.
    switch(n)
    {
    case 1:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(1, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(1, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(1, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(1, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(1, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(1, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(1, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(1, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(1, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(1, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(1, 1024); break;
        }
        break;
    case 2:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(2, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(2, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(2, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(2, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(2, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(2, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(2, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(2, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(2, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(2, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(2, 1024); break;
        }
        break;
    case 3:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(3, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(3, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(3, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(3, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(3, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(3, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(3, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(3, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(3, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(3, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(3, 1024); break;
        }
        break;
    case 4:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(4, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(4, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(4, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(4, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(4, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(4, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(4, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(4, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(4, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(4, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(4, 1024); break;
        }
        break;
    case 5:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(5, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(5, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(5, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(5, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(5, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(5, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(5, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(5, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(5, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(5, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(5, 1024); break;
        }
        break;
    case 6:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(6, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(6, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(6, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(6, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(6, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(6, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(6, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(6, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(6, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(6, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(6, 1024); break;
        }
        break;
    case 7:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(7, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(7, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(7, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(7, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(7, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(7, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(7, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(7, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(7, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(7, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(7, 1024); break;
        }
        break;
    case 8:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(8, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(8, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(8, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(8, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(8, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(8, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(8, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(8, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(8, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(8, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(8, 1024); break;
        }
        break;
    case 9:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(9, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(9, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(9, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(9, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(9, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(9, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(9, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(9, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(9, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(9, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(9, 1024); break;
        }
        break;
    case 10:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(10, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(10, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(10, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(10, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(10, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(10, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(10, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(10, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(10, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(10, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(10, 1024); break;
        }
        break;
    case 11:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(11, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(11, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(11, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(11, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(11, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(11, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(11, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(11, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(11, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(11, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(11, 1024); break;
        }
        break;
    case 12:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(12, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(12, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(12, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(12, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(12, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(12, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(12, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(12, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(12, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(12, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(12, 1024); break;
        }
        break;
    case 13:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(13, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(13, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(13, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(13, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(13, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(13, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(13, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(13, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(13, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(13, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(13, 1024); break;
        }
        break;
    case 14:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(14, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(14, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(14, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(14, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(14, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(14, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(14, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(14, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(14, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(14, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(14, 1024); break;
        }
        break;
    case 15:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(15, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(15, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(15, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(15, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(15, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(15, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(15, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(15, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(15, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(15, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(15, 1024); break;
        }
        break;
    case 16:
        switch(max_m)
        {
        case 1: RUN_GETF2_KERNEL_SMALL(16, 1); break;
        case 2: RUN_GETF2_KERNEL_SMALL(16, 2); break;
        case 4: RUN_GETF2_KERNEL_SMALL(16, 4); break;
        case 8: RUN_GETF2_KERNEL_SMALL(16, 8); break;
        case 16: RUN_GETF2_KERNEL_SMALL(16, 16); break;
        case 32: RUN_GETF2_KERNEL_SMALL(16, 32); break;
        case 64: RUN_GETF2_KERNEL_SMALL(16, 64); break;
        case 128: RUN_GETF2_KERNEL_SMALL(16, 128); break;
        case 256: RUN_GETF2_KERNEL_SMALL(16, 256); break;
        case 512: RUN_GETF2_KERNEL_SMALL(16, 512); break;
        default: RUN_GETF2_KERNEL_SMALL(16, 1024); break;
        }
        break;
    default: return rocblas_status_internal_error;
    }

#undef RUN_GETF2_KERNEL_SMALL

    return rocblas_status_success;
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GETF2_KERNEL_SMALL(T, I, INFO, U)                                              \
    template rocblas_status getf2_run_kernel_small<T, I, INFO, U>(                                 \
        rocblas_handle handle, const I m, const I n, U A, const rocblas_stride shiftA,             \
        const I lda, const rocblas_stride strideA, I* ipiv, const rocblas_stride shiftP,           \
        const rocblas_stride strideP, INFO* info, const I batch_count, const I offset,             \
        I* permut_idx, const rocblas_stride stridePI)

ROCSOLVER_END_NAMESPACE
