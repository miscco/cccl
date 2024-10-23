//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_SIMD_H
#define __CCCL_SIMD_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

// We want to ensure that all warning emmiting from this header are supressed
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Enable SIMD for compilers that support OpenMP 4.0
#if defined(_OPENMP) && _OPENMP >= 201307
#  if defined(_CCCL_COMPILER_MSVC)

#    define _CCCL_PRAGMA_SIMD                     __pragma(_CCCL_TOSTRING(omp simd))
#    define _CCCL_PRAGMA_DECLARE_SIMD             __pragma(_CCCL_TOSTRING(omp declare simd))
#    define _CCCL_PRAGMA_SIMD_REDUCTION(PRM)      __pragma(_CCCL_TOSTRING(omp simd reduction(PRM)))
#    define _CCCL_PRAGMA_SIMD_SCAN(PRM)           __pragma(_CCCL_TOSTRING(omp simd reduction(inscan, PRM)))
#    define _CCCL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) __pragma(_CCCL_TOSTRING(omp scan inclusive(PRM)))
#    define _CCCL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) __pragma(_CCCL_TOSTRING(omp scan exclusive(PRM)))

// Declaration of reduction functor, where
// NAME - the name of the functor
// OP - type of the callable object with the reduction operation
// omp_in - refers to the local partial result
// omp_out - refers to the final value of the combiner operator
// omp_priv - refers to the private copy of the initial value
// omp_orig - refers to the original variable to be reduced
#    define _CCCL_PRAGMA_DECLARE_REDUCTION(NAME, OP) \
      __pragma(_CCCL_TOSTRING(omp declare reduction(NAME:OP : omp_out(omp_in)) initializer(omp_priv = omp_orig)))

#  else // ^^^ MSVC ^^^ / vvv !MSVC vvv

#    define _CCCL_PRAGMA_SIMD                     _Pragma(_CCCL_TOSTRING(omp simd))
#    define _CCCL_PRAGMA_DECLARE_SIMD             _Pragma(_CCCL_TOSTRING(omp declare simd))
#    define _CCCL_PRAGMA_SIMD_REDUCTION(PRM)      _Pragma(_CCCL_TOSTRING(omp simd reduction(PRM)))
#    define _CCCL_PRAGMA_SIMD_SCAN(PRM)           _Pragma(_CCCL_TOSTRING(omp simd reduction(inscan, PRM)))
#    define _CCCL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM) _Pragma(_CCCL_TOSTRING(omp scan inclusive(PRM)))
#    define _CCCL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM) _Pragma(_CCCL_TOSTRING(omp scan exclusive(PRM)))

// Declaration of reduction functor, where
// NAME - the name of the functor
// OP - type of the callable object with the reduction operation
// omp_in - refers to the local partial result
// omp_out - refers to the final value of the combiner operator
// omp_priv - refers to the private copy of the initial value
// omp_orig - refers to the original variable to be reduced
#    define _CCCL_PRAGMA_DECLARE_REDUCTION(NAME, OP) \
      _Pragma(_CCCL_TOSTRING(omp declare reduction(NAME:OP : omp_out(omp_in)) initializer(omp_priv = omp_orig)))

#  endif // !MSVC
#else // ^^^ _OPENMP >= 201307 ^^^ / vvv !_OPENMP vvv

#  define _CCCL_PRAGMA_SIMD
#  define _CCCL_PRAGMA_DECLARE_SIMD
#  define _CCCL_PRAGMA_SIMD_REDUCTION(PRM)
#  define _CCCL_PRAGMA_SIMD_SCAN(PRM)
#  define _CCCL_PRAGMA_SIMD_INCLUSIVE_SCAN(PRM)
#  define _CCCL_PRAGMA_SIMD_EXCLUSIVE_SCAN(PRM)
#  define _CCCL_PRAGMA_DECLARE_REDUCTION(NAME, OP)

#endif // !_OPENMP

#endif // __CCCL_SIMD_H
