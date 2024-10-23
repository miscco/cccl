//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/std/__algorithm>
#include <cuda/std/cmath>
#include <cuda/std/ranges>

#include "nvbench_helper.cuh"

template <class T>
struct square_t
{
  __host__ __device__ void operator()(T& x) const noexcept
  {
    x = x * x + cuda::std::sin(static_cast<double>(x));
  }
};

template <typename T>
static void basic(nvbench::state& state, nvbench::type_list<T>)
{
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements"));

  std::vector<T> in(elements, T{1});

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  state.exec(nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cuda::std::ranges::for_each(cuda::std::execution::unseq_host, in.begin(), in.end(), square_t<T>{});
  });
}

NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(fundamental_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements", nvbench::range(14, 18, 4));
