// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// InclusiveSum shaped like https://github.com/NVIDIA/cccl/issues/11840.
// The stored symbols are an integer type. A zero marks a segment reset every 1024 elements.
// The fancy path scans a transform_iterator of uint32 reset flags into a transform_output_iterator that keeps the
// low 8 bits. The pointer path materializes those flags as uint8 and scans raw pointers.

#include <cub/device/device_scan.cuh>

#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

constexpr int reset_period = 1024;

template <typename T>
struct make_symbol_t
{
  __host__ __device__ T operator()(std::int64_t index) const
  {
    return index % reset_period == 0 ? T{} : T{1};
  }
};

template <typename T>
struct zero_to_reset_flag_t
{
  __host__ __device__ std::uint32_t operator()(T symbol) const
  {
    return symbol == T{} ? 1u : 0u;
  }
};

struct mod_to_uint8_t
{
  __host__ __device__ std::uint8_t operator()(std::uint32_t value) const
  {
    return static_cast<std::uint8_t>(value % 256u);
  }
};

template <typename T>
void reset_segment_scan(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto elements = state.get_int64("Elements{io}");
  const auto source   = state.get_string("Source");
  const bool fancy    = source == "transform";

  thrust::device_vector<T> symbols(static_cast<std::size_t>(elements));
  auto indices = cuda::counting_iterator<std::int64_t>(0);
  thrust::transform(indices, indices + elements, symbols.begin(), make_symbol_t<T>{});

  thrust::device_vector<std::uint8_t> flags(static_cast<std::size_t>(elements));
  thrust::device_vector<std::uint8_t> segments(static_cast<std::size_t>(elements));
  if (!fancy)
  {
    thrust::transform(symbols.begin(), symbols.end(), flags.begin(), zero_to_reset_flag_t<T>{});
  }

  auto* d_symbols  = thrust::raw_pointer_cast(symbols.data());
  auto* d_flags    = thrust::raw_pointer_cast(flags.data());
  auto* d_segments = thrust::raw_pointer_cast(segments.data());

  state.add_element_count(elements);
  if (fancy)
  {
    state.add_global_memory_reads<T>(elements, "Size");
    state.add_global_memory_writes<std::uint8_t>(elements);
  }
  else
  {
    state.add_global_memory_reads<std::uint8_t>(elements, "Size");
    state.add_global_memory_writes<std::uint8_t>(elements);
  }

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    if (fancy)
    {
      auto input  = cuda::transform_iterator(d_symbols, zero_to_reset_flag_t<T>{});
      auto output = cuda::transform_output_iterator(d_segments, mod_to_uint8_t{});
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveSum, "Scan failed", input, output, static_cast<std::int64_t>(elements), env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceScan::InclusiveSum, "Scan failed", d_flags, d_segments, static_cast<std::int64_t>(elements), env);
    }
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

using value_types = nvbench::type_list<std::int8_t, std::int16_t, std::int32_t, std::int64_t, int128_t>;

NVBENCH_BENCH_TYPES(reset_segment_scan, NVBENCH_TYPE_AXES(value_types))
  .set_name("reset_segment")
  .set_type_axes_names({"T{ct}"})
  .add_string_axis("Source", {"pointer", "transform"})
  .add_int64_power_of_two_axis("Elements{io}", {20, 24, 28, 31});
