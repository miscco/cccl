//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Check that std::rotl and std::rotr are marked [[nodiscard]]

#include <cuda/std/bit>

__host__ __device__ void func() {
  cuda::std::rotl(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cuda::std::rotr(0u, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}

int main(int, char **)
{
  return 0;
}