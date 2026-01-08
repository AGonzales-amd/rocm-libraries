/* **************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_utility.hpp"

#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define ROCSOLVER_MFMA_ENABLED 1
#else
#define ROCSOLVER_MFMA_ENABLED 0
#endif // defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)

ROCSOLVER_BEGIN_NAMESPACE

#if ROCSOLVER_MFMA_ENABLED

template <typename T>
struct mfma_16x16x4_base
{
    using RegT = T;
    using AccT = __attribute__((__vector_size__(4 * sizeof(T)))) T;
};

template <typename T>
struct mfma_16x16x4;

// float specialization
template <>
struct mfma_16x16x4<float> : public mfma_16x16x4_base<float>
{
    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
    }
};

// double specialization
template <>
struct mfma_16x16x4<double> : public mfma_16x16x4_base<double>
{
    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
    }
};

// complex specialization
template <typename T>
struct mfma_16x16x4
{
    using RegT = T;
    using AccT = std::array<T, 4>;

    using S = decltype(std::real(T{}));
    using RegS = typename mfma_16x16x4_base<S>::RegT;
    using AccS = typename mfma_16x16x4_base<S>::AccT;

    __device__ inline auto operator()(const RegT& a, const RegT& b, const AccT& c) const
    {
        RegS ar = a.real();
        RegS ai = a.imag();
        RegS br = b.real();
        RegS bi = b.imag();
        AccS cr = {c[0].real(), c[1].real(), c[2].real(), c[3].real()};
        AccS ci = {c[0].imag(), c[1].imag(), c[2].imag(), c[3].imag()};
        AccS zero = {0};

        const auto mfma_S = mfma_16x16x4<S>();

        // real x real
        auto arbr = mfma_S(ar, br, zero);

        // real x imag
        auto arbi = mfma_S(ar, bi, zero);

        // imag x real
        auto aibr = mfma_S(ai, br, zero);

        // imag x imag
        auto aibi = mfma_S(ai, bi, zero);

        // cr += r x r - i x i
        cr += arbr - aibi;
        // ci += r x i + i x r
        ci += arbi + aibr;

        return AccT{rocblas_complex_num<S>(cr[0], ci[0]), rocblas_complex_num<S>(cr[1], ci[1]),
                    rocblas_complex_num<S>(cr[2], ci[2]), rocblas_complex_num<S>(cr[3], ci[3])};
    }
};

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, rocblas_float_complex>, int> = 0>
__device__ inline I get_c_col(I li, I lj, I gpri, I inc_C, I ldc)
{
    return lj;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>, int> = 0>
__device__ inline I get_c_col(I li, I lj, I gpri, I inc_C, I ldc)
{
    return lj;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, float> || std::is_same_v<T, rocblas_float_complex>, int> = 0>
__device__ inline I get_c_row(I li, I lj, I gpri, I inc_C, I ldc)
{
    return gpri + li * 4;
}

template <typename T,
          typename I,
          std::enable_if_t<std::is_same_v<T, double> || std::is_same_v<T, rocblas_double_complex>, int> = 0>
__device__ inline I get_c_row(I li, I lj, I gpri, I inc_C, I ldc)
{
    return gpri * 4 + li;
}

struct warp_gemm
{
    static constexpr auto M = 16;
    static constexpr auto N = 16;
    static constexpr auto K = 4;

    template <typename T>
    using accumulator = typename mfma_16x16x4<T>::AccT;

    struct handle
    {
        __device__ handle(int tid)
            : lid(tid % warpSize)
            , cmajor_i_16x4(lid % M)
            , cmajor_j_16x4(lid / M)
            , cmajor_i_4x16(lid % K)
            , cmajor_j_4x16(lid / K)
            , c2r_src(cmajor_i_16x4 * 4 + cmajor_j_16x4)
            , r2c_src(cmajor_i_4x16 * 16 + cmajor_j_4x16)
        {
        }

        __device__ handle()
            : handle(hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
                     + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y))
        {
        }

        const int lid;

        // addresses to index elements in registers
        const int cmajor_i_16x4;
        const int cmajor_j_16x4;
        const int cmajor_i_4x16;
        const int cmajor_j_4x16;

        // addresses to transpose B from col-major to row-major
        // and transpose C from row-major to col-major
        const int c2r_src;
        const int r2c_src;
    };

    template <typename T, typename I>
    static __device__ T load_a(const handle& h,
                               const rocblas_operation transA,
                               const I m,
                               const I k,
                               const T* A,
                               const I inc,
                               const I lda)
    {
        // load A
        T amk = 0;
        if(transA == rocblas_operation_none)
        {
            // read col major 16x4 A
            if(h.cmajor_i_16x4 < m && h.cmajor_j_16x4 < k)
                amk = A[h.cmajor_j_16x4 * lda + h.cmajor_i_16x4 * inc];
        }
        else
        {
            // read col major 4x16 op(A)
            if(h.cmajor_j_4x16 < m && h.cmajor_i_4x16 < k)
                amk = A[h.cmajor_j_4x16 * lda + h.cmajor_i_4x16 * inc];

            // transpose op(A) to 16x4
            amk = shfl(amk, h.c2r_src);

            if constexpr(rocblas_is_complex<T>)
            {
                if(transA == rocblas_operation_conjugate_transpose)
                    amk = conj(amk);
            }
        }

        return amk;
    }

    template <typename T, typename I>
    static __device__ T load_b(const handle& h,
                               const rocblas_operation transB,
                               const I n,
                               const I k,
                               const T* B,
                               const I inc,
                               const I ldb)
    {
        T bkn = 0;

        // load B
        if(transB == rocblas_operation_none)
        {
            // read col major 4x16 B
            if(h.cmajor_j_4x16 < n && h.cmajor_i_4x16 < k)
                bkn = B[h.cmajor_j_4x16 * ldb + h.cmajor_i_4x16 * inc];

            // transpose B to row major
            bkn = shfl(bkn, h.c2r_src);
        }
        else
        {
            // read col major 16x4 op(B)
            if(h.cmajor_i_16x4 < n && h.cmajor_j_16x4 < k)
                bkn = B[h.cmajor_j_16x4 * ldb + h.cmajor_i_16x4 * inc];

            if constexpr(rocblas_is_complex<T>)
            {
                if(transB == rocblas_operation_conjugate_transpose)
                    bkn = conj(bkn);
            }
        }

        return bkn;
    }

    template <typename T, typename I, typename T4 = typename mfma_16x16x4<T>::AccT>
    static __device__ T4 load_c(const handle& h,
                                const rocblas_operation transC,
                                const I m,
                                const I n,
                                const T* C,
                                const I inc,
                                const I ldc)
    {
        T4 dmn = {0};

        if(transC == rocblas_operation_none)
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                if(c_col < n && c_row < m)
                    dmn[i] = C[idx];

                // transpose C to row major
                dmn[i] = shfl(dmn[i], h.c2r_src);
            }
        }
        else
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                if(c_col < n && c_row < m)
                    dmn[i] = C[idx];
            }
        }

        return dmn;
    }

    template <typename T, typename I, typename T4 = typename mfma_16x16x4<T>::AccT>
    static __device__ void write_c(const handle& h,
                                   const rocblas_operation transC,
                                   const I m,
                                   const I n,
                                   T* C,
                                   const I inc,
                                   const I ldc,
                                   T4& dmn)
    {
        if(transC == rocblas_operation_none)
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                // transpose C to col major
                dmn[i] = shfl(dmn[i], h.r2c_src);

                if(c_col < n && c_row < m)
                    C[idx] = dmn[i];
            }
        }
        else
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                if(c_col < n && c_row < m)
                    C[idx] = dmn[i];
            }
        }
    }

    template <typename T, typename I, typename T4 = typename mfma_16x16x4<T>::AccT>
    static __device__ void write_c(const handle& h,
                                   const rocblas_operation transC,
                                   const I m,
                                   const I n,
                                   T alpha,
                                   T beta,
                                   T* C,
                                   const I inc,
                                   const I ldc,
                                   T4& dmn)
    {
        if(transC == rocblas_operation_none)
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                // transpose C to col major
                dmn[i] = shfl(dmn[i], h.r2c_src);

                if(c_col < n && c_row < m)
                    C[idx] = alpha * dmn[i] + beta * C[idx];
            }
        }
        else
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                if(c_col < n && c_row < m)
                    C[idx] = alpha * dmn[i] + beta * C[idx];
            }
        }
    }

    template <typename T, typename I, typename T4 = typename mfma_16x16x4<T>::AccT>
    static __device__ void write_c_tri(const handle& h,
                                       const rocblas_operation transC,
                                       const rocblas_fill uploC,
                                       const I m,
                                       const I n,
                                       T* C,
                                       const I inc,
                                       const I ldc,
                                       T4& dmn)
    {
        bool const is_upper = (uploC == rocblas_fill_upper);
        if(transC == rocblas_operation_none)
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_4x16, (I)h.cmajor_j_4x16, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                // transpose C to col major
                dmn[i] = shfl(dmn[i], h.r2c_src);

                if(c_col < n && c_row < m)
                {
                    if((is_upper && c_col >= c_row) || (!is_upper && c_col <= c_row))
                        C[idx] = dmn[i];
                }
            }
        }
        else
        {
#pragma unroll
            for(I i = 0; i < K; ++i)
            {
                const I c_col = get_c_col<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I c_row = get_c_row<T>((I)h.cmajor_i_16x4, (I)h.cmajor_j_16x4, i, inc, ldc);
                const I idx = (c_col * ldc) + (c_row * inc);

                if(c_col < n && c_row < m)
                {
                    if((is_upper && c_col >= c_row) || (!is_upper && c_col <= c_row))
                        C[idx] = dmn[i];
                }
            }
        }
    }

    template <typename T, typename... Args>
    static __device__ auto run(Args&&... args)
    {
        return mfma_16x16x4<T>(std::forward<Args>(args)...);
    }
};

/** GEMM device function to compute C = alpha * A * B + beta * C.

    Where C is an m x n matrix, A is an m x p matrix, and B is an
    p x n matrix. This is a wave function, every lane of the wave
    must perform call this function.

    MFMA instruction element and register mapping tool:
    https://github.com/ROCm/amd_matrix_instruction_calculator

    transA      form of op(A).
    transB      form of op(B).
    m           number of rows of matrix C.
                0 < m <= 16
    n           number of columns of matrix C.
                0 < n <= 16
    p           number of columns of matrix op(A) and number of rows of matrix op(B).
                0 < p
    alpha       scalar alpha.
    A           pointer to matrix A.
    inc_A       stride from the start of one row to the next of matrix A.
    lda         leading dimension of A.
    B           pointer to matrix B.
    inc_B       stride from the start of one row to the next of matrix B.
    ldb         leading dimension of B.
    C           pointer to matrix C.
    inc_C       stride from the start of one row to the next of matrix C.
    ldc         leading dimension of C.

**/
// Run with warpSize sized block
template <typename T, typename I>
__device__ void gemm_16x16xp(rocblas_operation transA,
                             rocblas_operation transB,
                             I m,
                             I n,
                             I p,
                             T alpha,
                             const T* A,
                             I inc_A,
                             I lda,
                             const T* B,
                             I inc_B,
                             I ldb,
                             T beta,
                             T* C,
                             I inc_C,
                             I ldc)
{
    const auto handle = warp_gemm::handle();
    auto dmn = warp_gemm::accumulator<T>{0};

    for(I kb = 0; kb < p; kb += warp_gemm::K)
    {
        // read A and B in col-major
        T amk = 0;
        T bkn = 0;

        // load A
        if(transA == rocblas_operation_none)
            amk = warp_gemm::load_a(handle, transA, m, p - kb, A + (kb * lda), inc_A, lda);
        else
            amk = warp_gemm::load_a(handle, transA, m, p - kb, A + (kb * inc_A), inc_A, lda);

        // load B
        if(transB == rocblas_operation_none)
            bkn = warp_gemm::load_b(handle, transB, n, p - kb, B + (kb * inc_B), inc_B, ldb);
        else
            bkn = warp_gemm::load_b(handle, transB, n, p - kb, B + (kb * ldb), inc_B, ldb);

        dmn = warp_gemm::run<T>()(amk, bkn, dmn);
    }

    warp_gemm::write_c(handle, rocblas_operation_none, m, n, alpha, beta, C, inc_C, ldc, dmn);
}

#endif // ROCSOLVER_MFMA_ENABLED

ROCSOLVER_END_NAMESPACE
