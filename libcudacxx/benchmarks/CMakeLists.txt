include(${CMAKE_SOURCE_DIR}/benchmarks/cmake/CCCLBenchmarkRegistry.cmake)

if(NOT CCCL_ENABLE_CUB)
  message(FATAL_ERROR "libcudacxx benchmarks depend on CUB: set CCCL_ENABLE_CUB.")
endif()

if(NOT CCCL_ENABLE_BENCHMARKS)
  message(FATAL_ERROR "libcudacxx benchmarks depend on CUB benchmarks: set CCCL_ENABLE_BENCHMARKS.")
endif()

set(benches_root "${CMAKE_CURRENT_LIST_DIR}")

function(get_recursive_subdirs subdirs)
  set(dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/bench/*"
  )

  foreach(test_dir IN LISTS contents)
    if(IS_DIRECTORY "${test_dir}")
      list(APPEND dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs} "${dirs}" PARENT_SCOPE)
endfunction()

function(add_bench target_name bench_name bench_src)
  set(bench_target ${bench_name})
  set(${target_name} ${bench_target} PARENT_SCOPE)

  message("LIBUDACXX_BENCHMARKS: added bench_target ${bench_target}")
  add_executable(${bench_target} "${bench_src}")

  set_target_properties(${bench_target}
    PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
      CUDA_STANDARD 17
      CXX_STANDARD 17)
  target_link_libraries(${bench_target} PRIVATE nvbench_helper nvbench::main)
endfunction()

function(add_bench_dir bench_dir)
  file(GLOB bench_srcs CONFIGURE_DEPENDS "${bench_dir}/*.cu")
  file(RELATIVE_PATH bench_prefix "${benches_root}" "${bench_dir}")
  file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
  string(REPLACE "/" "." bench_prefix "${bench_prefix}")

  foreach(bench_src IN LISTS bench_srcs)
    set(real_bench_src "${bench_src}")

    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    string(PREPEND bench_name "libcudacxx.${bench_prefix}.")
    register_cccl_benchmark("${bench_name}" "")

    string(APPEND bench_name ".base")
    add_bench(base_bench_target ${bench_name} "${real_bench_src}")

    if(OPENMP_FOUND)
      target_compile_options(${base_bench_target} PRIVATE "-Xcompiler=-fopenmp")
      target_link_options(${base_bench_target} PRIVATE "-lgomp")
    endif()
  endforeach()
endfunction()

get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  add_bench_dir("${subdir}")
endforeach()
