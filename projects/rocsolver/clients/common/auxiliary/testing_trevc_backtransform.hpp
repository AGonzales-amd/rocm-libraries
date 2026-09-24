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
                                      const rocblas_int mm,
                                      const T* dTS,
                                      const rocblas_int ldt,
                                      const T* dQL,
                                      const rocblas_int ldql,
                                      const T* dQR,
                                      const rocblas_int ldqr,
                                      T* dVL,
                                      const rocblas_int ldvl,
                                      T* dVR,
                                      const rocblas_int ldvr)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(nullptr, side, n, mm, dTS, ldt, dQL, ldql, dQR, ldqr, dVL,
                                      ldvl, dVR, ldvr),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, (rocblas_side)0, n, mm, dTS, ldt, dQL, ldql, dQR,
                                      ldqr, dVL, ldvl, dVR, ldvr),
        rocblas_status_invalid_value);

    // sizes
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, n, mm, dTS, n - 1, dQL, ldql, dQR, ldqr, dVL,
                                      ldvl, dVR, ldvr),
        rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, n, mm, (T*)nullptr, ldt, dQL, ldql, dQR, ldqr,
                                      dVL, ldvl, dVR, ldvr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_left, n, mm, dTS, ldt, (T*)nullptr,
                                      ldql, dQR, ldqr, dVL, ldvl, dVR, ldvr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_right, n, mm, dTS, ldt, dQL, ldql,
                                      (T*)nullptr, ldqr, dVL, ldvl, dVR, ldvr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_left, n, mm, dTS, ldt, dQL, ldql, dQR,
                                      ldqr, (T*)nullptr, ldvl, dVR, ldvr),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_right, n, mm, dTS, ldt, dQL, ldql, dQR,
                                      ldqr, dVL, ldvl, (T*)nullptr, ldvr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, 0, 0, (T*)nullptr, ldt, (T*)nullptr, ldql,
                                      (T*)nullptr, ldqr, (T*)nullptr, ldvl, (T*)nullptr, ldvr),
        rocblas_status_success);
}

template <typename T>
void testing_trevc_backtransform_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_right;
    rocblas_int n = 5;
    rocblas_int mm = 5;
    rocblas_int ldt = 5;
    rocblas_int ldql = 5;
    rocblas_int ldqr = 5;
    rocblas_int ldvl = 5;
    rocblas_int ldvr = 5;

#ifdef ROCSOLVER_ENABLE_TREVC3
    // memory allocations
    device_strided_batch_vector<T> dTS(ldt * n, 1, ldt * n, 1);
    device_strided_batch_vector<T> dQL(ldql * n, 1, ldql * n, 1);
    device_strided_batch_vector<T> dQR(ldqr * n, 1, ldqr * n, 1);
    device_strided_batch_vector<T> dVL(ldvl * mm, 1, ldvl * mm, 1);
    device_strided_batch_vector<T> dVR(ldvr * mm, 1, ldvr * mm, 1);
    CHECK_HIP_ERROR(dTS.memcheck());
    CHECK_HIP_ERROR(dQL.memcheck());
    CHECK_HIP_ERROR(dQR.memcheck());
    CHECK_HIP_ERROR(dVL.memcheck());
    CHECK_HIP_ERROR(dVR.memcheck());

    // check bad arguments
    trevc_backtransform_checkBadArgs(handle, side, n, mm, dTS.data(), ldt, dQL.data(), ldql,
                                     dQR.data(), ldqr, dVL.data(), ldvl, dVR.data(), ldvr);
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trevc_backtransform_initData(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  const rocblas_int mm,
                                  Td& dTS,
                                  const rocblas_int ldt,
                                  Td& dQL,
                                  const rocblas_int ldql,
                                  Td& dQR,
                                  const rocblas_int ldqr,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  Th& hTS,
                                  Th& hQL,
                                  Th& hQR,
                                  Th& hVL,
                                  Th& hVR)
{
    if(CPU)
    {
        using S = decltype(std::real(T{}));

        bool left = (side == rocblas_side_left || side == rocblas_side_both);
        bool right = (side == rocblas_side_right || side == rocblas_side_both);

        // Build an upper (quasi-)triangular matrix T resembling a Schur form from HSEQR.
        // For real types, a handful of adjacent column pairs get a 2x2 block with
        // off-diagonal entries to emulate conjugate-pair eigenvalues.
        rocblas_init<T>(hTS, true);

        for(rocblas_int j = 0; j < n; ++j)
        {
            // zero out strictly lower triangular part
            for(rocblas_int i = j + 1; i < n; ++i)
                hTS[0][i + j * ldt] = T(0);
            // scale diagonal to ensure eigenvalues are well-separated from zero
            hTS[0][j + j * ldt] += 400;
        }

        if constexpr(!rocblas_is_complex<T>)
        {
            // Mark every third pair of adjacent columns as a 2x2 conjugate-pair block
            // for(rocblas_int j = 0; j + 1 < n; j += 3)
            rocblas_int j = n / 2;
            if(j < n - 1)
            {
                hTS[0][(j + 1) + j * ldt] = hTS[0][j + j * ldt];
                hTS[0][(j + 1) + (j + 1) * ldt] = -hTS[0][j + (j + 1) * ldt];
            }
        }

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
            cpu_trevc3<T, S>(side, 'A', n, hTS[0], ldt,
                             left ? hVL[0] : nullptr, ldvl,
                             right ? hVR[0] : nullptr, ldvr,
                             mm, &m_out,
                             work_tv.data(), lwork_tv,
                             rwork_tv.data(), lrwork_tv);
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dTS.transfer_from(hTS));
        if(side == rocblas_side_left || side == rocblas_side_both)
        {
            CHECK_HIP_ERROR(dQL.transfer_from(hQL));
            CHECK_HIP_ERROR(dVL.transfer_from(hVL));
        }
        if(side == rocblas_side_right || side == rocblas_side_both)
        {
            CHECK_HIP_ERROR(dQR.transfer_from(hQR));
            CHECK_HIP_ERROR(dVR.transfer_from(hVR));
        }
    }
}

template <typename T, typename Td, typename Th>
void trevc_backtransform_getError(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  const rocblas_int mm,
                                  Td& dTS,
                                  const rocblas_int ldt,
                                  Td& dQL,
                                  const rocblas_int ldql,
                                  Td& dQR,
                                  const rocblas_int ldqr,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  Th& hTS,
                                  Th& hQL,
                                  Th& hQR,
                                  Th& hVL,
                                  Th& hVLRes,
                                  Th& hVR,
                                  Th& hVRRes,
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

    // input data initialization
    trevc_backtransform_initData<true, true, T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR,
                                                ldqr, dVL, ldvl, dVR, ldvr, hTS, hQL, hQR, hVL,
                                                hVR);

    // GPU execution
    CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, mm, dTS.data(), ldt,
                                                      dQL.data(), ldql, dQR.data(), ldqr,
                                                      dVL.data(), ldvl, dVR.data(), ldvr));
    if(left)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(right)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));

    cpu_trevc3<T, S>(side, 'B', n, hTS[0], ldt,
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
                                     const rocblas_int mm,
                                     Td& dTS,
                                     const rocblas_int ldt,
                                     Td& dQL,
                                     const rocblas_int ldql,
                                     Td& dQR,
                                     const rocblas_int ldqr,
                                     Td& dVL,
                                     const rocblas_int ldvl,
                                     Td& dVR,
                                     const rocblas_int ldvr,
                                     Th& hTS,
                                     Th& hQL,
                                     Th& hQR,
                                     Th& hVL,
                                     Th& hVR,
                                     double* gpu_time_used,
                                     double* cpu_time_used,
                                     const rocblas_int hot_calls,
                                     const int profile,
                                     const bool profile_kernels,
                                     const bool perf)
{
    if(!perf)
    {
        trevc_backtransform_initData<true, false, T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR,
                                                     ldqr, dVL, ldvl, dVR, ldvr, hTS, hQL, hQR,
                                                     hVL, hVR);

        // cpu-lapack performance (not implemented -- measure zero)
        *cpu_time_used = get_time_us_no_sync();
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trevc_backtransform_initData<true, false, T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR,
                                                 ldqr, dVL, ldvl, dVR, ldvr, hTS, hQL, hQR, hVL,
                                                 hVR);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trevc_backtransform_initData<false, true, T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR,
                                                     ldqr, dVL, ldvl, dVR, ldvr, hTS, hQL, hQR,
                                                     hVL, hVR);

        CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, mm, dTS.data(), ldt,
                                                          dQL.data(), ldql, dQR.data(), ldqr,
                                                          dVL.data(), ldvl, dVR.data(), ldvr));
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
        trevc_backtransform_initData<false, true, T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR,
                                                     ldqr, dVL, ldvl, dVR, ldvr, hTS, hQL, hQR,
                                                     hVL, hVR);

        timer.start(stream);
        rocsolver_trevc_backtransform(handle, side, n, mm, dTS.data(), ldt, dQL.data(), ldql,
                                      dQR.data(), ldqr, dVL.data(), ldvl, dVR.data(), ldvr);
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
    rocblas_int mm = argus.get<rocblas_int>("mm", n);
    rocblas_int ldt = argus.get<rocblas_int>("ldt", n);
    rocblas_int ldql = argus.get<rocblas_int>("ldql", n);
    rocblas_int ldqr = argus.get<rocblas_int>("ldqr", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n);

    rocblas_int hot_calls = argus.iters;

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // determine sizes
    size_t size_TS = (size_t)ldt * n;
    size_t size_VL = left ? (size_t)ldvl * mm : 0;
    size_t size_VR = right ? (size_t)ldvr * mm : 0;
    size_t size_QL = left ? (size_t)ldql * n : 0;
    size_t size_QR = right ? (size_t)ldqr * n : 0;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_VLRes = (argus.unit_check || argus.norm_check) ? size_VL : 0;
    size_t size_VRRes = (argus.unit_check || argus.norm_check) ? size_VR : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_TREVC3
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, mm, (T*)nullptr, ldt, (T*)nullptr, ldql,
                                          (T*)nullptr, ldqr, (T*)nullptr, ldvl, (T*)nullptr, ldvr),
            rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || mm < 0 || (ldt < n) || (left && ldvl < n) || (right && ldvr < n) || (left && ldql < n) || (right && ldqr < n));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, mm, (T*)nullptr, ldt, (T*)nullptr, ldql,
                                          (T*)nullptr, ldqr, (T*)nullptr, ldvl, (T*)nullptr, ldvr),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_trevc_backtransform(handle, side, n, mm, (T*)nullptr, ldt,
                                                        (T*)nullptr, ldql, (T*)nullptr, ldqr,
                                                        (T*)nullptr, ldvl, (T*)nullptr, ldvr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hTS(size_TS, 1, size_TS, 1);
    host_strided_batch_vector<T> hQL(size_QL > 0 ? size_QL : 1, 1, size_QL > 0 ? size_QL : 1, 1);
    host_strided_batch_vector<T> hQR(size_QR > 0 ? size_QR : 1, 1, size_QR > 0 ? size_QR : 1, 1);
    host_strided_batch_vector<T> hVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1, 1);
    host_strided_batch_vector<T> hVLRes(size_VLRes > 0 ? size_VLRes : 1, 1,
                                        size_VLRes > 0 ? size_VLRes : 1, 1);
    host_strided_batch_vector<T> hVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1, 1);
    host_strided_batch_vector<T> hVRRes(size_VRRes > 0 ? size_VRRes : 1, 1,
                                        size_VRRes > 0 ? size_VRRes : 1, 1);
    device_strided_batch_vector<T> dTS(size_TS, 1, size_TS, 1);
    device_strided_batch_vector<T> dQL(size_QL > 0 ? size_QL : 1, 1, size_QL > 0 ? size_QL : 1,
                                       1);
    device_strided_batch_vector<T> dQR(size_QR > 0 ? size_QR : 1, 1, size_QR > 0 ? size_QR : 1,
                                       1);
    device_strided_batch_vector<T> dVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1,
                                       1);
    device_strided_batch_vector<T> dVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1,
                                       1);
    CHECK_HIP_ERROR(dTS.memcheck());
    if(size_QL)
        CHECK_HIP_ERROR(dQL.memcheck());
    if(size_QR)
        CHECK_HIP_ERROR(dQR.memcheck());
    if(size_VL)
        CHECK_HIP_ERROR(dVL.memcheck());
    if(size_VR)
        CHECK_HIP_ERROR(dVR.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, mm, dTS.data(), ldt, dQL.data(), ldql,
                                          dQR.data(), ldqr, dVL.data(), ldvl, dVR.data(), ldvr),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        trevc_backtransform_getError<T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR, ldqr, dVL,
                                        ldvl, dVR, ldvr, hTS, hQL, hQR, hVL, hVLRes, hVR, hVRRes,
                                        &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        trevc_backtransform_getPerfData<T>(handle, side, n, mm, dTS, ldt, dQL, ldql, dQR, ldqr,
                                           dVL, ldvl, dVR, ldvr, hTS, hQL, hQR, hVL, hVR,
                                           &gpu_time_used, &cpu_time_used, hot_calls, argus.profile,
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
            rocsolver_bench_output("side", "n", "mm", "ldt", "ldql", "ldqr", "ldvl", "ldvr");
            rocsolver_bench_output(side, n, mm, ldt, ldql, ldqr, ldvl, ldvr);
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
