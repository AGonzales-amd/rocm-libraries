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
                                      const T* dT,
                                      const rocblas_int ldt,
                                      T* dVL,
                                      const rocblas_int ldvl,
                                      T* dVR,
                                      const rocblas_int ldvr,
                                      const rocblas_int mm)
{
    // handle
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(nullptr, side, n, dT, ldt, dVL, ldvl, dVR, ldvr, mm),
        rocblas_status_invalid_handle);

    // values
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, (rocblas_side)0, n, dT, ldt, dVL, ldvl, dVR, ldvr,
                                      mm),
        rocblas_status_invalid_value);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldt, dVL, ldvl, dVR, ldvr, mm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_left, n, dT, ldt, (T*)nullptr, ldvl,
                                      dVR, ldvr, mm),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, rocblas_side_right, n, dT, ldt, dVL, ldvl, (T*)nullptr,
                                      ldvr, mm),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_trevc_backtransform(handle, side, 0, (T*)nullptr, ldt, (T*)nullptr, ldvl,
                                      (T*)nullptr, ldvr, 0),
        rocblas_status_success);
}

template <typename T>
void testing_trevc_backtransform_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_side side = rocblas_side_right;
    rocblas_int n = 5;
    rocblas_int ldt = 5;
    rocblas_int ldvl = 5;
    rocblas_int ldvr = 5;
    rocblas_int mm = 5;

#ifdef ROCSOLVER_ENABLE_TREVC3
    // memory allocations
    device_strided_batch_vector<T> dT(ldt * n, 1, ldt * n, 1);
    device_strided_batch_vector<T> dVL(ldvl * mm, 1, ldvl * mm, 1);
    device_strided_batch_vector<T> dVR(ldvr * mm, 1, ldvr * mm, 1);
    CHECK_HIP_ERROR(dT.memcheck());
    CHECK_HIP_ERROR(dVL.memcheck());
    CHECK_HIP_ERROR(dVR.memcheck());

    // check bad arguments
    trevc_backtransform_checkBadArgs(handle, side, n, dT.data(), ldt, dVL.data(), ldvl, dVR.data(),
                                     ldvr, mm);
#endif
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void trevc_backtransform_initData(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  Td& dT_mat,
                                  const rocblas_int ldt,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  const rocblas_int mm,
                                  Th& hT_mat,
                                  Th& hVL,
                                  Th& hVR)
{
    if(CPU)
    {
        bool left = (side == rocblas_side_left || side == rocblas_side_both);
        bool right = (side == rocblas_side_right || side == rocblas_side_both);

        // fill T with a random upper triangular matrix
        rocblas_init<T>(hT_mat, true);
        // zero out the strictly lower triangular part
        for(rocblas_int j = 0; j < n; ++j)
            for(rocblas_int i = j + 1; i < n; ++i)
                hT_mat[0][i + j * ldt] = 0;

        // fill VL/VR with identity (suitable for back-transform or all modes)
        if(left)
        {
            rocblas_init<T>(hVL, true);
        }
        if(right)
        {
            rocblas_init<T>(hVR, true);
        }
    }

    if(GPU)
    {
        CHECK_HIP_ERROR(dT_mat.transfer_from(hT_mat));
        if(side == rocblas_side_left || side == rocblas_side_both)
            CHECK_HIP_ERROR(dVL.transfer_from(hVL));
        if(side == rocblas_side_right || side == rocblas_side_both)
            CHECK_HIP_ERROR(dVR.transfer_from(hVR));
    }
}

template <typename T, typename Td, typename Th>
void trevc_backtransform_getError(const rocblas_handle handle,
                                  const rocblas_side side,
                                  const rocblas_int n,
                                  Td& dT_mat,
                                  const rocblas_int ldt,
                                  Td& dVL,
                                  const rocblas_int ldvl,
                                  Td& dVR,
                                  const rocblas_int ldvr,
                                  const rocblas_int mm,
                                  Th& hT_mat,
                                  Th& hVL,
                                  Th& hVLRes,
                                  Th& hVR,
                                  Th& hVRRes,
                                  double* max_err)
{
    using S = decltype(std::real(T{}));

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // input data initialization
    trevc_backtransform_initData<true, true, T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR, ldvr,
                                                mm, hT_mat, hVL, hVR);

    // GPU execution
    CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, dT_mat.data(), ldt,
                                                      dVL.data(), ldvl, dVR.data(), ldvr, mm));
    if(left)
        CHECK_HIP_ERROR(hVLRes.transfer_from(dVL));
    if(right)
        CHECK_HIP_ERROR(hVRRes.transfer_from(dVR));

    // CPU reference
    rocblas_int m_out = mm;
    rocblas_int lwork = 3 * n;
    rocblas_int lrwork = n;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    cpu_trevc3<T, S>(side, n, hT_mat[0], ldt, hVL[0], ldvl, hVR[0], ldvr, mm, &m_out, work.data(),
                     lwork, rwork.data(), lrwork);

    // error is max Frobenius norm of diff for VL and/or VR
    double err;
    *max_err = 0;
    if(left)
    {
        err = norm_error('F', n, mm, ldvl, hVL[0], hVLRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
    if(right)
    {
        err = norm_error('F', n, mm, ldvr, hVR[0], hVRRes[0]);
        *max_err = err > *max_err ? err : *max_err;
    }
}

template <typename T, typename Td, typename Th>
void trevc_backtransform_getPerfData(const rocblas_handle handle,
                                     const rocblas_side side,
                                     const rocblas_int n,
                                     Td& dT_mat,
                                     const rocblas_int ldt,
                                     Td& dVL,
                                     const rocblas_int ldvl,
                                     Td& dVR,
                                     const rocblas_int ldvr,
                                     const rocblas_int mm,
                                     Th& hT_mat,
                                     Th& hVL,
                                     Th& hVR,
                                     double* gpu_time_used,
                                     double* cpu_time_used,
                                     const rocblas_int hot_calls,
                                     const int profile,
                                     const bool profile_kernels,
                                     const bool perf)
{
    using S = decltype(std::real(T{}));

    rocblas_int lwork = 3 * n;
    rocblas_int lrwork = n;
    std::vector<T> work(lwork);
    std::vector<S> rwork(lrwork);
    rocblas_int m_out = mm;

    if(!perf)
    {
        trevc_backtransform_initData<true, false, T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR,
                                                     ldvr, mm, hT_mat, hVL, hVR);

        // cpu-lapack performance
        *cpu_time_used = get_time_us_no_sync();
        cpu_trevc3<T, S>(side, n, hT_mat[0], ldt, hVL[0], ldvl, hVR[0], ldvr, mm, &m_out,
                         work.data(), lwork, rwork.data(), lrwork);
        *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
    }

    trevc_backtransform_initData<true, false, T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR,
                                                 ldvr, mm, hT_mat, hVL, hVR);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        trevc_backtransform_initData<false, true, T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR,
                                                     ldvr, mm, hT_mat, hVL, hVR);

        CHECK_ROCBLAS_ERROR(rocsolver_trevc_backtransform(handle, side, n, dT_mat.data(), ldt,
                                                          dVL.data(), ldvl, dVR.data(), ldvr, mm));
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
        trevc_backtransform_initData<false, true, T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR,
                                                     ldvr, mm, hT_mat, hVL, hVR);

        timer.start(stream);
        rocsolver_trevc_backtransform(handle, side, n, dT_mat.data(), ldt, dVL.data(), ldvl,
                                      dVR.data(), ldvr, mm);
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
    rocblas_int ldt = argus.get<rocblas_int>("ldt", n);
    rocblas_int ldvl = argus.get<rocblas_int>("ldvl", n);
    rocblas_int ldvr = argus.get<rocblas_int>("ldvr", n);
    rocblas_int mm = argus.get<rocblas_int>("mm", n);

    rocblas_int hot_calls = argus.iters;

    bool left = (side == rocblas_side_left || side == rocblas_side_both);
    bool right = (side == rocblas_side_right || side == rocblas_side_both);

    // determine sizes
    size_t size_T = ldt * n;
    size_t size_VL = left ? (size_t)ldvl * mm : 0;
    size_t size_VR = right ? (size_t)ldvr * mm : 0;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_VLRes = (argus.unit_check || argus.norm_check) ? size_VL : 0;
    size_t size_VRRes = (argus.unit_check || argus.norm_check) ? size_VR : 0;

// check feature flag
#ifndef ROCSOLVER_ENABLE_TREVC3
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldt, (T*)nullptr, ldvl,
                                          (T*)nullptr, ldvr, mm),
            rocblas_status_not_implemented);

        if(argus.timing)
            rocsolver_bench_inform(inform_not_implemented);

        return;
    }
#endif

    // check invalid sizes
    bool invalid_size = (n < 0 || ldt < std::max(1, n) || mm < 0
                         || (left && ldvl < std::max(1, n)) || (right && ldvr < std::max(1, n)));
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldt, (T*)nullptr, ldvl,
                                          (T*)nullptr, ldvr, mm),
            rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_trevc_backtransform(handle, side, n, (T*)nullptr, ldt,
                                                        (T*)nullptr, ldvl, (T*)nullptr, ldvr, mm));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    host_strided_batch_vector<T> hT_mat(size_T, 1, size_T, 1);
    host_strided_batch_vector<T> hVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1, 1);
    host_strided_batch_vector<T> hVLRes(size_VLRes > 0 ? size_VLRes : 1, 1,
                                        size_VLRes > 0 ? size_VLRes : 1, 1);
    host_strided_batch_vector<T> hVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1, 1);
    host_strided_batch_vector<T> hVRRes(size_VRRes > 0 ? size_VRRes : 1, 1,
                                        size_VRRes > 0 ? size_VRRes : 1, 1);
    device_strided_batch_vector<T> dT_mat(size_T, 1, size_T, 1);
    device_strided_batch_vector<T> dVL(size_VL > 0 ? size_VL : 1, 1, size_VL > 0 ? size_VL : 1,
                                       1);
    device_strided_batch_vector<T> dVR(size_VR > 0 ? size_VR : 1, 1, size_VR > 0 ? size_VR : 1,
                                       1);
    if(size_T)
        CHECK_HIP_ERROR(dT_mat.memcheck());
    if(size_VL)
        CHECK_HIP_ERROR(dVL.memcheck());
    if(size_VR)
        CHECK_HIP_ERROR(dVR.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_trevc_backtransform(handle, side, n, dT_mat.data(), ldt, dVL.data(), ldvl,
                                          dVR.data(), ldvr, mm),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        trevc_backtransform_getError<T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR, ldvr, mm,
                                        hT_mat, hVL, hVLRes, hVR, hVRRes, &max_error);

    // collect performance data
    if(argus.timing && hot_calls > 0)
        trevc_backtransform_getPerfData<T>(handle, side, n, dT_mat, ldt, dVL, ldvl, dVR, ldvr, mm,
                                           hT_mat, hVL, hVR, &gpu_time_used, &cpu_time_used,
                                           hot_calls, argus.profile, argus.profile_kernels,
                                           argus.perf);

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
            rocsolver_bench_output("side", "n", "ldt", "ldvl", "ldvr", "mm");
            rocsolver_bench_output(side, n, ldt, ldvl, ldvr, mm);
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
