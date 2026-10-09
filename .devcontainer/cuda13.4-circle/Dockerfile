# Circle on the default CUDA 13.4 devcontainer.
#
# That image already installs NVCC. Circle enables CUDA only when
# --cuda-path points at that toolkit:
# https://github.com/seanbaxter/circle/blob/master/cuda/README.md
#
# Build:
#   docker build -t cccl/devcontainer:cuda13.4-circle .devcontainer/cuda13.4-circle
#
# Circle is not an upstream CMake compiler ID. install-cmake-support.py copies
# unofficial detection modules into this image's CMake and teaches the ID
# scan to recognize Circle before Clang (Circle defines __clang__).

FROM rapidsai/devcontainers:26.12-cpp-gcc15-cuda13.4

USER root

RUN curl -fsSL -o /tmp/circle.tgz https://www.circle-lang.org/download/circle_linux_x86_64.tgz \
    && tar -xzf /tmp/circle.tgz -C /usr/local/bin \
    && chmod 755 /usr/local/bin/circle \
    && rm /tmp/circle.tgz \
    && circle -dumpversion

COPY cmake /opt/cccl-circle/cmake
COPY install-cmake-support.py /opt/cccl-circle/install-cmake-support.py

# Patch the CMake that ships in the base image. CMAKE_ROOT is not the
# /usr/local layout of the official Kitware tarball.
RUN cmake_root="$(cmake --system-information 2>/dev/null | sed -n 's/.*CMAKE_ROOT \"\([^\"]*\)\".*/\1/p' | head -1)" \
    && test -n "${cmake_root}" \
    && test -x /usr/local/cuda/bin/nvcc \
    && python3 /opt/cccl-circle/install-cmake-support.py "${cmake_root}" /opt/cccl-circle/cmake \
    && chmod a+r \
        "${cmake_root}"/Modules/Compiler/Circle*.cmake \
        "${cmake_root}"/Modules/CMakeCompilerIdDetection.cmake \
        "${cmake_root}"/Modules/CMakeDetermineCUDACompiler.cmake \
        "${cmake_root}"/Modules/Internal/CMakeCUDAFindToolkit.cmake

# --cuda-path is the NVCC toolkit. -sm_XX is required; CTK 13 starts at sm_75.
# -DNULL=0: with GCC 15, Circle parses CTK 13.4's cuda_runtime_api.h before
# NULL is declared. The define stays in CUDAFLAGS so host C++ compiles are unchanged.
# sccache does not recognize Circle and injects -P, which Circle rejects.
# /usr/bin/env is a no-op launcher. The base image's profile uses ${VAR:-sccache},
# so a non-empty value here is what keeps sccache from replacing it.
ENV CC=circle \
    CXX=circle \
    CUDAHOSTCXX=circle \
    CUDACXX=circle \
    CUDA_PATH=/usr/local/cuda \
    CUDAARCHS=75 \
    CUDAFLAGS="--cuda-path=/usr/local/cuda -sm_75 -DNULL=0" \
    CMAKE_C_COMPILER_LAUNCHER=/usr/bin/env \
    CMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/env \
    CMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/env

CMD ["/bin/bash"]
