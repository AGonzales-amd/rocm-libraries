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

#include "common/auxiliary/testing_trevc_backtransform.hpp"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;
using namespace std;

// each size_range entry is {side ('L'=0,'R'=1,'B'=2), n, ldt, ldvl, ldvr, mm}
// case when n = 0 will also execute the bad arguments test
// (null handle, null pointers and invalid values)

// for checkin_lapack tests
const vector<vector<int>> trevc_backtransform_size_range = {
    // quick return (n = 0)
    {1, 0, 1, 1, 1, 0},
    // invalid
    {1, -1, 5, 1, 5, 5},  // n < 0
    {1, 5, 4, 1, 5, 5},   // ldt < n
    {0, 5, 5, 4, 1, 5},   // ldvl < n (side = left)
    {1, 5, 5, 1, 4, 5},   // ldvr < n (side = right)
    // normal (valid) samples — side: 0=left, 1=right, 2=both
    {0, 5, 5, 5, 1, 5},
    {1, 5, 5, 1, 5, 5},
    {2, 5, 5, 5, 5, 5},
    {1, 10, 10, 1, 10, 10},
    {2, 20, 20, 20, 20, 20},
    {0, 50, 50, 50, 1, 50},
    {1, 50, 50, 1, 50, 50},
    {2, 50, 50, 50, 50, 50},
};

// for daily_lapack tests
const vector<vector<int>> trevc_backtransform_large_size_range = {
    {1, 200, 200, 1, 200, 200},
    {2, 512, 512, 512, 512, 512},
    {2, 1000, 1024, 1024, 1024, 1000},
};

Arguments trevc_backtransform_setup_arguments(vector<int> sz)
{
    Arguments arg;

    char side_c = (sz[0] == 0) ? 'L' : (sz[0] == 1) ? 'R' : 'B';
    arg.set<char>("side", side_c);
    arg.set<rocblas_int>("n", sz[1]);
    arg.set<rocblas_int>("ldt", sz[2]);
    arg.set<rocblas_int>("ldvl", sz[3]);
    arg.set<rocblas_int>("ldvr", sz[4]);
    arg.set<rocblas_int>("mm", sz[5]);

    arg.timing = 0;

    return arg;
}

class TREVC_BACKTRANSFORM : public ::TestWithParam<vector<int>>
{
protected:
    void TearDown() override
    {
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    template <typename T>
    void run_tests()
    {
        Arguments arg = trevc_backtransform_setup_arguments(GetParam());

        if(arg.peek<rocblas_int>("n") == 0)
            testing_trevc_backtransform_bad_arg<T>();

        testing_trevc_backtransform<T>(arg);
    }
};

// non-batch tests

TEST_P(TREVC_BACKTRANSFORM, __float)
{
    run_tests<float>();
}

TEST_P(TREVC_BACKTRANSFORM, __double)
{
    run_tests<double>();
}

TEST_P(TREVC_BACKTRANSFORM, __float_complex)
{
    run_tests<rocblas_float_complex>();
}

TEST_P(TREVC_BACKTRANSFORM, __double_complex)
{
    run_tests<rocblas_double_complex>();
}

INSTANTIATE_TEST_SUITE_P(daily_lapack,
                         TREVC_BACKTRANSFORM,
                         ValuesIn(trevc_backtransform_large_size_range));

INSTANTIATE_TEST_SUITE_P(checkin_lapack,
                         TREVC_BACKTRANSFORM,
                         ValuesIn(trevc_backtransform_size_range));
