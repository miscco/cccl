#!/usr/bin/env python3
"""Install unofficial Circle compiler detection into a CMake module tree.

CMake discovers compiler IDs only from ``${CMAKE_ROOT}/Modules``. Circle is not
an upstream compiler ID, and it defines ``__clang__``, so the Circle test has
to run before Clang. This script copies the Circle modules next to CMake's
own and patches the two files that hard-code the ID list.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"{path.name}: pattern not found ({label})")
    path.write_text(text.replace(old, new, 1))


def install(cmake_root: Path, module_src: Path) -> None:
    compiler_dir = cmake_root / "Modules" / "Compiler"
    if not compiler_dir.is_dir():
        raise SystemExit(f"Not a CMake module tree: {compiler_dir}")

    for src in sorted(module_src.glob("Compiler/Circle*.cmake")):
        shutil.copy2(src, compiler_dir / src.name)

    id_detection = cmake_root / "Modules" / "CMakeCompilerIdDetection.cmake"
    replace_once(
        id_detection,
        """    if("x${lang}" STREQUAL "xCUDA")
      set(ordered_compilers NVIDIA Clang)
    endif()""",
        """    if("x${lang}" STREQUAL "xCUDA")
      # unofficial Circle compiler detection
      set(ordered_compilers Circle NVIDIA Clang)
    endif()""",
        "CUDA ordered_compilers",
    )
    replace_once(
        id_detection,
        """    list(APPEND ordered_compilers
      Clang
      LCC
      GNU""",
        """    list(APPEND ordered_compilers
      # unofficial Circle compiler detection (Circle defines __clang__)
      Circle
      Clang
      LCC
      GNU""",
        "C/CXX ordered_compilers",
    )

    determine = cmake_root / "Modules" / "CMakeDetermineCUDACompiler.cmake"
    replace_once(
        determine,
        """    list(APPEND CMAKE_CUDA_COMPILER_ID_VENDORS NVIDIA Clang)
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_NVIDIA "nvcc: [^\\n]+ Cuda compiler driver")
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_Clang "(clang version)")""",
        """    list(APPEND CMAKE_CUDA_COMPILER_ID_VENDORS Circle NVIDIA Clang)
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_Circle "Circle build")
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_NVIDIA "nvcc: [^\\n]+ Cuda compiler driver")
    set(CMAKE_CUDA_COMPILER_ID_VENDOR_REGEX_Clang "(clang version)")""",
        "CUDA vendor list",
    )
    replace_once(
        determine,
        """  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(clang_test_flags "--cuda-path=\\"${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}\\"")""",
        """  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Circle")
    # unofficial Circle compiler detection
    set(circle_test_flags "--cuda-path=\\"${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}\\" -DNULL=0")
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(clang_test_flags "--cuda-path=\\"${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}\\"")""",
        "Circle ID test flags",
    )
    replace_once(
        determine,
        """  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    list(APPEND CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "${nvcc_test_flags}")
  endif()""",
        """  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "Circle")
    # Circle takes -sm_XX. CTK 13 dropped everything below sm_75.
    list(APPEND CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "${circle_test_flags} -sm_75")
  elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    list(APPEND CMAKE_CUDA_COMPILER_ID_TEST_FLAGS_FIRST "${nvcc_test_flags}")
  endif()""",
        "Circle ID architecture flags",
    )
    replace_once(
        determine,
        """if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  string(REGEX MATCHALL "-target-cpu sm_([0-9]+)" _clang_target_cpus "${CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT}")""",
        """if(CMAKE_CUDA_COMPILER_ID MATCHES "^(Clang|Circle)$")
  string(REGEX MATCHALL "-target-cpu sm_([0-9]+)" _clang_target_cpus "${CMAKE_CUDA_COMPILER_PRODUCED_OUTPUT}")""",
        "Circle toolkit include discovery",
    )
    replace_once(
        determine,
        """  unset(_CUDA_INCLUDE_DIRS)
  unset(_CUDA_LIBRARY_DIR)
  unset(_CUDA_TARGET_DIR)
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")""",
        """  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Circle" AND NOT CMAKE_CUDA_ARCHITECTURES_DEFAULT)
    # Circle's -v output does not include Clang's `-target-cpu sm_N` lines.
    set(CMAKE_CUDA_ARCHITECTURES_DEFAULT "75")
  endif()

  unset(_CUDA_INCLUDE_DIRS)
  unset(_CUDA_LIBRARY_DIR)
  unset(_CUDA_TARGET_DIR)
elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")""",
        "Circle default architecture",
    )

    toolkit = cmake_root / "Modules" / "Internal" / "CMakeCUDAFindToolkit.cmake"
    replace_once(
        toolkit,
        """  if(CMAKE_${lang}_COMPILER_ID STREQUAL "Clang")
    execute_process(COMMAND ${_CUDA_NVCC_EXECUTABLE} "--version"
      OUTPUT_VARIABLE CMAKE_${lang}_COMPILER_ID_OUTPUT
      RESULT_VARIABLE _result_nvcc_version)
  endif()""",
        """  if(CMAKE_${lang}_COMPILER_ID MATCHES "^(Clang|Circle)$")
    # Circle's --version does not include the toolkit V<major.minor.patch> string.
    execute_process(COMMAND ${_CUDA_NVCC_EXECUTABLE} "--version"
      OUTPUT_VARIABLE CMAKE_${lang}_COMPILER_ID_OUTPUT
      RESULT_VARIABLE _result_nvcc_version)
  endif()""",
        "Circle toolkit version",
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} CMAKE_ROOT MODULE_SRC")
    install(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
