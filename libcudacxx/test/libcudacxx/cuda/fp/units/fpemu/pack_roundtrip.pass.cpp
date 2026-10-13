// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Unpack/pack must preserve zero, denormals, normal values and infinities
// without shifts exceeding the mantissa width.

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpemu>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

namespace cudax = cuda::experimental;

TEST_FUNC void check(cuda::std::uint64_t bits)
{
  auto unpacked = cudax::__internal_fp64emu_unpack(bits);
  assert(cudax::__internal_fp64emu_pack<cudax::__fpemu_rounding::rn>(unpacked) == bits);
  assert(cudax::__internal_fp64emu_pack<cudax::__fpemu_rounding::rz>(unpacked) == bits);
  assert(cudax::__internal_fp64emu_pack<cudax::__fpemu_rounding::ru>(unpacked) == bits);
  assert(cudax::__internal_fp64emu_pack<cudax::__fpemu_rounding::rd>(unpacked) == bits);
}

int main(int, char**)
{
  check(0);
  check(0x8000000000000000ULL);
  check(1);
  check(0x8000000000000001ULL);
  check(0x000fffffffffffffULL);
  check(0x800fffffffffffffULL);
  check(0x0010000000000000ULL);
  check(0x8010000000000000ULL);
  check(0x3ff0000000000000ULL);
  check(0x7fefffffffffffffULL);
  check(0x7ff0000000000000ULL);
  check(0xfff0000000000000ULL);
  return 0;
}
