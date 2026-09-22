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

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"

template <typename T>
void trevc_backtransform_checkBadArgs(const rocblas_handle handle,
                                      const rocblas_side side,
                                      const rocblas_int n,
                                      T* dVL,
                                      const rocblas_int ldvl,
                                      T* dVR,
                                      const rocblas_int ldvr,
                                      const T* dQL,
                                      const rocblas_int ldql,
                                      const T* dQR,
                                      const rocblas_int ldqr,
                                      const rocblas_int mm)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(nullptr, side, n, dVL, ldvl, dVR, ldvr, dQL, ldql, dQR,
                                      ldqr, mm),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, (rocblas_side)0, n, dVL, ldvl, dVR, ldvr, dQL, ldql,
                                      dQR, ldqr, mm),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_left, n, (T*)nullptr, ldvl, dVR, ldvr,
                                      dQL, ldql, dQR, ldqr, mm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_right, n, dVL, ldvl, (T*)nullptr, ldvr,
                                      dQL, ldql, dQR, ldqr, mm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_left, n, dVL, ldvl, dVR, ldvr,
                                      (T*)nullptr, ldql, dQR, ldqr, mm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_right, n, dVL, ldvl, dVR, ldvr, dQL,
                                      ldql, (T*)nullptr, ldqr, mm),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, 0, (T*)nullptr, ldvl, (T*)nullptr, ldvr,
                                      (T*)nullptr, ldql, (T*)nullptr, ldqr, 0),
        rocblas_status_success);
}

template <typename T>
void testing_trevc_backtransform_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_right;
    rocblas_int n = 5;
    rocblas_int ldvl = 5;
    rocblas_int ldvr = 5;
    rocblas_int ldql = 5;
    rocblas_int ldqr = 5;
    rocblas_int mm = 5;

#ifdef ROCSOLVER_ENABLE_TREVC3
    // memory allocations
    device_strided_batch_vector<T> dVL(ldvl * mm, 1, ldvl * mm, 1);
    device_strided_batch_vector<T> dVR(ldvr * mm, 1, ldvr * mm, 1);
    device_strided_batch_vector<T> dQL(ldql * mm, 1, ldql * mm, 1);
    device_strided_batch_vector<T> dQR(ldqr * mm, 1, ldqr * mm, 1);
    CHECK_HIP_ERROR(dVL.memcheck());
    CHECK_HIP_ERROR(dVR.memcheck());
    CHECK_HIP_ERROR(dQL.memcheck());
    CHECK_HIP_ERROR(dQR.memcheck());

    // check bad arguments
    trevc_backtransform_checkBadArgs(handle, side, n, dVL.data(), ldvl, dVR.data(), ldvr,
                                     dQL.data(), ldql, dQR.data(), ldqr, mm);
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trevc_backtransform_initData(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  Td& dQL,
                                  const rocblas_int ldql,
                                  Td& dQR,
                                  const rocblas_int ldqr,
                                  const rocblas_int mm,
                                  Th& hVL,
                                  Th& hVR,
                                  Th& hQL,
                                  Th& hQR,
                                  host_strided_batch_vector<T>& hTRef)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));

        bool left = (side == rocblas_side_left || side == rocblas_side_both);
        bool right = (side == rocblas_side_right || side == rocblas_side_both);

        // Build an upper (quasi-)triangular matrix T resembling a Schur form from HSEQR.
        // For real types, a handful of adjacent column pairs get a 2x2 block with
        // off-diagonal entries to emulate conjugate-pair eigenvalues.
        rocblas_int ldt_tmp = n;
    host_strided_batch_vector<T> hT(n * n, 1, n * n, 1);
        rocblas_init<T>(hT, true);

        for(rocblas_int j = 0; j < n; ++j)
        {
            // zero out strictly lower triangular part
            for(rocblas_int i = j + 1; i < n; ++i)
                hT[0][i + j * ldt_tmp] = T(0);
            // scale diagonal: self-product keeps real part positive and scales by n
            // to ensure eigenvalues are well-separated from zero
            hT[0][j + j * ldt_tmp] += 400;
        }

        if constexpr(!rocblas_is_complex<T>)
        {
            // Mark every third pair of adjacent columns as a 2x2 conjugate-pair block
            // for(rocblas_int j = 0; j + 1 < n; j += 3)
            rocblas_int j = n/2;
            if(j < n - 1)
            {
                hT[0][(j + 1) + j * ldt_tmp] = hT[0][j + j * ldt_tmp];
                hT[0][(j + 1) + (j + 1) * ldt_tmp] = -hT[0][j + (j + 1) * ldt_tmp];
            }
        }

        hTRef.copy_from(hT);

        // QL/QR represent the orthogonal Schur-vector matrix Q from HSEQR
        if(left)
            rocblas_init<T>(hQL, true);
        if(right)
            rocblas_init<T>(hQR, true);

        // Compute VL/VR as eigenvectors of T via cpu_trevc3
        if(left || right)
        {
            rocblas_int m_out = 0;
            rocblas_int lwork_tv = 3 * n;
            rocblas_int lrwork_tv = n;
            std::vector<T> work_tv(lwork_tv);
            std::vector<S> rwork_tv(lrwork_tv);
            cpu_trevc3<T, S>(side, 'A', n, hT[0], ldt_tmp,
                             left ? hVL[0] : nullptr, ldvl,
                             right ? hVR[0] : nullptr, ldvr,
                             mm, &m_out,
                             work_tv.data(), lwork_tv,
                             rwork_tv.data(), lrwork_tv);
        }
    }

    if(GPU)
    {
        if(side == rocblas_side_left || side == rocblas_side_both)
        {
            CHECK_HIP_ERROR(dVL.transfer_from(hVL));
            CHECK_HIP_ERROR(dQL.transfer_from(hQL));
        }
        if(side == rocblas_side_right || side == rocblas_side_both)
        {
            CHECK_HIP_ERROR(dVR.transfer_from(hVR));
            CHECK_HIP_ERROR(dQR.transfer_from(hQR));
        }
    }
}

template <typename T, typename Td, typename Th>
void trevc_backtransform_getError(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  Td& dQL,
                                  const rocblas_int ldql,
                                  Td& dQR,
                                  const rocblas_int ldqr,
                                  const rocblas_int mm,
                                  Th& hVL,
                                  Th& hVLRes,
                                  Th& hVR,
                                  Th& hVRRes,
                                  Th& hQL,
                                  Th& hQR,
                                  double* max_err)
{
    using S = decltype(std::real(T{}));

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    rocblas_int m_out = 0;
    rocblas_int lwork_tv = 3 * n;
    rocblas_int lrwork_tv = n;
    std::vector<T> work_tv(lwork_tv);
    std::vector<S> rwork_tv(lrwork_tv);

    host_strided_batch_vector<T> hT(n * n, 1, n * n, 1);

    // input data initialization
    trevc_backtransform_initData<true, true, T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL, ldql,
                                                dQR, ldqr, mm, hVL, hVR, hQL, hQR, hT);

    // GPU execution
    CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, dVL, ldvl,
                                                      dVR, ldvr, dQL, ldql,
                                                      dQR, ldqr, mm));
    if(left)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(right)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));

    std::cout << "hQL" << hQL << std::endl;
    std::cout << "hVL" << hVL << std::endl;

    cpu_trevc3<T, S>(side, 'B', n, hT[0], n,
                    left ? hQL[0] : nullptr, ldql,
                    right ? hQR[0] : nullptr, ldqr,
                    mm, &m_out,
                    work_tv.data(), lwork_tv,
                    rwork_tv.data(), lrwork_tv);

    double err;
    *max_err = 0;
    if(left)
    {
        err = norm_error('F', n, mm, ldvl, hQL[0], hVLRes[0]);
        *max_err = err > *max_err ? err : *max_err;

    std::cout << "hQL" << hQL << std::endl;
    std::cout << "hVLRes" << hVLRes << std::endl;
    }
    if(right)
    {
        err = norm_error('F', n, mm, ldvr, hQR[0], hVRRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Td, typename Th>
void trevc_backtransform_getPerfData(const rocblas_handle handle,
                                     const rocblas_side side,
                                     const rocblas_int n,
                                     Td& dVL,
                                     const rocblas_int ldvl,
                                     Td& dVR,
                                     const rocblas_int ldvr,
                                     Td& dQL,
                                     const rocblas_int ldql,
                                     Td& dQR,
                                     const rocblas_int ldqr,
                                     const rocblas_int mm,
                                     Th& hVL,
                                     Th& hVR,
                                     Th& hQL,
                                     Th& hQR,
                                     double* gpu_time_used,
                                     double* cpu_time_used,
                                     const rocblas_int hot_calls,
                                     const int profile,
                                     const bool profile_kernels,
                                     const bool perf)
{
    host_strided_batch_vector<T> hT(n * n, 1, n * n, 1);

    if(!perf)
    {
        trevc_backtransform_initData<true, false, T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL,
                                                     ldql, dQR, ldqr, mm, hVL, hVR, hQL, hQR, hT);

        // cpu-lapack performance (not implemented -- measure zero)
        *cpu_time_used = get_time_us_no_sync();
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trevc_backtransform_initData<true, false, T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL, ldql,
                                                 dQR, ldqr, mm, hVL, hVR, hQL, hQR, hT);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trevc_backtransform_initData<false, true, T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL,
                                                     ldql, dQR, ldqr, mm, hVL, hVR, hQL, hQR, hT);

        CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, dVL.data(), ldvl,
                                                          dVR.data(), ldvr, dQL.data(), ldql,
                                                          dQR.data(), ldqr, mm));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    rocsolver_timer timer;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        trevc_backtransform_initData<false, true, T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL,
                                                     ldql, dQR, ldqr, mm, hVL, hVR, hQL, hQR, hT);

        timer.start(stream);
        rocsolver_trevc_backtransform(handle, side, n, dVL.data(), ldvl, dVR.data(), ldvr,
                                      dQL.data(), ldql, dQR.data(), ldqr, mm);
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

template <typename T>
void testing_trevc_backtransform(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    char sideC = argus.get<char>("side");
    rocblas_side side = char2rocblas_side(sideC);
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n);
    rocblas_int ldql = argus.get<rocblas_int>("ldql", n);
    rocblas_int ldqr = argus.get<rocblas_int>("ldqr", n);
    rocblas_int mm = argus.get<rocblas_int>("mm", n);

    rocblas_int hot_calls = argus.iters;

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // determine sizes
    size_t size_VL = left ? (size_t)ldvl * n : 0;
    size_t size_VR = right ? (size_t)ldvr * n : 0;
    size_t size_QL = left ? (size_t)ldql * n : 0;
    size_t size_QR = right ? (size_t)ldqr * n : 0;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_VLRes = (argus.unit_check || argus.norm_check) ? size_VL : 0;
    size_t size_VRRes = (argus.unit_check || argus.norm_check) ? size_VR : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_TREVC3
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldvl, (T*)nullptr, ldvr,
                                          (T*)nullptr, ldql, (T*)nullptr, ldqr, mm),
            rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || mm < 0 || (left && ldvl < std::max(1, n))
                         || (right && ldvr < std::max(1, n)) || (left && ldql < std::max(1, n))
                         || (right && ldqr < std::max(1, n)));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldvl, (T*)nullptr, ldvr,
                                          (T*)nullptr, ldql, (T*)nullptr, ldqr, mm),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldvl,
                                                        (T*)nullptr, ldvr, (T*)nullptr, ldql,
                                                        (T*)nullptr, ldqr, mm));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1, 1);
    host_strided_batch_vector<T> hVLRes(size_VLRes > 0 ? size_VLRes : 1, 1,
                                        size_VLRes > 0 ? size_VLRes : 1, 1);
    host_strided_batch_vector<T> hVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1, 1);
    host_strided_batch_vector<T> hVRRes(size_VRRes > 0 ? size_VRRes : 1, 1,
                                        size_VRRes > 0 ? size_VRRes : 1, 1);
    host_strided_batch_vector<T> hQL(size_QL > 0 ? size_QL : 1, 1, size_QL > 0 ? size_QL : 1, 1);
    host_strided_batch_vector<T> hQR(size_QR > 0 ? size_QR : 1, 1, size_QR > 0 ? size_QR : 1, 1);
    device_strided_batch_vector<T> dVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1,
                                       1);
    device_strided_batch_vector<T> dVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1,
                                       1);
    device_strided_batch_vector<T> dQL(size_QL > 0 ? size_QL : 1, 1, size_QL > 0 ? size_QL : 1,
                                       1);
    device_strided_batch_vector<T> dQR(size_QR > 0 ? size_QR : 1, 1, size_QR > 0 ? size_QR : 1,
                                       1);
    if(size_VL)
        CHECK_HIP_ERROR(dVL.memcheck());
    if(size_VR)
        CHECK_HIP_ERROR(dVR.memcheck());
    if(size_QL)
        CHECK_HIP_ERROR(dQL.memcheck());
    if(size_QR)
        CHECK_HIP_ERROR(dQR.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, dVL.data(), ldvl, dVR.data(), ldvr,
                                          dQL.data(), ldql, dQR.data(), ldqr, mm),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        trevc_backtransform_getError<T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL, ldql, dQR,
                                        ldqr, mm, hVL, hVLRes, hVR, hVRRes, hQL, hQR, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        trevc_backtransform_getPerfData<T>(handle, side, n, dVL, ldvl, dVR, ldvr, dQL, ldql, dQR,
                                           ldqr, mm, hVL, hVR, hQL, hQR, &gpu_time_used,
                                           &cpu_time_used, hot_calls, argus.profile,
                                           argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using 8 * n * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 8 * n);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("side", "n", "ldvl", "ldvr", "ldql", "ldqr", "mm");
            rocsolver_bench_output(side, n, ldvl, ldvr, ldql, ldqr, mm);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_TREVC_BACKTRANSFORM(...) \
    extern template void testing_trevc_backtransform<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_TREVC_BACKTRANSFORM, FOREACH_SCALAR_TYPE, APPLY_STAMP)
