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

#include "clients_utility.hpp"
#include "common/containers/d_vector.hpp"
#include <amd_smi/amdsmi.h>
#include <chrono>
#include <iostream>
#include <thread>

#define AMDSMI_CHECK(...)                                                       \
    do                                                                          \
    {                                                                           \
        auto _status = (__VA_ARGS__);                                           \
        if(_status != AMDSMI_STATUS_SUCCESS)                                    \
        {                                                                       \
            const char* _errstr;                                                \
            (void)amdsmi_status_code_to_string(_status, &_errstr);              \
            fmt::print(stderr, "error: {} ({}) at {}:{}\n", _errstr,            \
                       static_cast<std::int32_t>(_status), __FILE__, __LINE__); \
            rocblas_abort();                                                    \
        }                                                                       \
    } while(0)

// Warms the GPU, using complex enough dummy kernel code
/// that the compiler can't optimize it away.
__global__ static void warmup_kernel(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        float x = 0.5f + idx * 0.0001f;

        for(int i = 0; i < 10000; ++i)
        {
            x += sinf(x) * cosf(x);
            x *= 1.0000001f;
            x = sqrtf(x + 1.0f);
            x = logf(x + 1.0f);
        }

        data[idx] = x;
    }
}

class rocsolver_temp
{
public:
    // Delete all copy/move constructors and assignment operators.
    rocsolver_temp(const rocsolver_temp&) = delete;
    rocsolver_temp& operator=(const rocsolver_temp&) = delete;
    rocsolver_temp(rocsolver_temp&&) = delete;
    rocsolver_temp& operator=(rocsolver_temp&&) = delete;

    /// Singleton accessor.
    static rocsolver_temp& instance()
    {
        static rocsolver_temp instance;
        return instance;
    }

    void set_target_temp(const rocblas_local_handle& handle,
                         uint16_t min_gpu_temp,
                         uint16_t max_gpu_temp,
                         double max_duration) const
    {
        auto start = std::chrono::steady_clock::now();

        // warm up device
        while(true)
        {
            uint16_t gpu_temp = get_temp(handle);
            if(gpu_temp >= min_gpu_temp)
                break;

            dim3 blocks(m_num_items / m_threads_per_block);
            dim3 threads(m_threads_per_block);

            warmup_kernel<<<blocks, threads, 0, handle.get_stream()>>>(m_device_storage, m_num_items);

            auto status = hipStreamSynchronize(handle.get_stream());
            if(status != hipSuccess)
            {
                fmt::print(stderr, "error: {} ({}) at {}:{}\n", hipGetErrorString(status),
                           static_cast<std::int32_t>(status), __FILE__, __LINE__);
                rocblas_abort();
            }

            auto duration = std::chrono::steady_clock::now() - start;
            if(duration >= std::chrono::duration<double>(max_duration))
            {
                fmt::print(stderr, "error: failed to reach target temp after {} at {}:{}\n",
                           max_duration, __FILE__, __LINE__);
                rocblas_abort();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // cool down device
        auto cool_down_start_time = std::chrono::steady_clock::now();
        while(true)
        {
            uint16_t gpu_temp = get_temp(handle);
            if(gpu_temp <= max_gpu_temp)
                break;

            auto duration = std::chrono::steady_clock::now() - start;
            if(duration >= std::chrono::duration<double>(max_duration))
            {
                fmt::print(stderr, "error: failed to reach target temp after {} at {}:{}\n",
                           max_duration, __FILE__, __LINE__);
                rocblas_abort();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

private:
    rocsolver_temp()
    {
        int grid_size;
        auto status = hipOccupancyMaxPotentialBlockSize(
            &grid_size, // Minimum grid size for full occupancy.
            &m_threads_per_block, // Block size for full occupancy.
            warmup_kernel,
            0, // Dynamic shared memory.
            0 // Block size limit (0 = no limit).
        );
        if(status != hipSuccess)
        {
            fmt::print(stderr, "error: {} ({}) at {}:{}\n", hipGetErrorString(status),
                       static_cast<std::int32_t>(status), __FILE__, __LINE__);
            rocblas_abort();
        }

        m_num_items = grid_size * m_threads_per_block;

        m_device_storage = d_vector<float, 0, float>(m_num_items).device_vector_setup();
    }

    ~rocsolver_temp()
    {
        d_vector<float, 0, float>(m_num_items).device_vector_teardown(m_device_storage);
    }

    /// Returns the current temperature.
    uint16_t get_temp(const rocblas_local_handle& handle) const
    {
        const auto props = handle.get_dev_props();

        AMDSMI_CHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

        // Build the AMD SMI BDF struct from HIP device properties.
        amdsmi_bdf_t addr{
            .function_number = 0, // HIP doesn't expose PCI function ID.
            .device_number = static_cast<uint8_t>(props.pciDeviceID),
            .bus_number = static_cast<uint8_t>(props.pciBusID),
            .domain_number = static_cast<uint16_t>(props.pciDomainID),
        };

        amdsmi_processor_handle amdsmi_device;
        AMDSMI_CHECK(amdsmi_get_processor_handle_from_bdf(addr, &amdsmi_device));

        constexpr amdsmi_temperature_type_t types[] = {
            AMDSMI_TEMPERATURE_TYPE_EDGE,
            AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
        };

        int64_t t;
        for(auto type : types)
        {
            if(amdsmi_get_temp_metric(amdsmi_device, type, AMDSMI_TEMP_CURRENT, &t)
               == AMDSMI_STATUS_SUCCESS)
            {
                return t;
            }
        }

        fmt::print(stderr, "error: no amdsmi temp metric at {}:{}\n", __FILE__, __LINE__);
        rocblas_abort();

        return -1;
    }

private:
    int m_threads_per_block;
    int m_num_items;

    float* m_device_storage;
}; // class rocsolver_temp
