#!/bin/bash

set -eo pipefail

# Script defaults
VERBOSE=${VERBOSE:-}
HOST_COMPILER=${CXX:-g++} # $CXX if set, otherwise `g++`
CXX_STANDARD=17
CUDA_COMPILER=${CUDACXX:-nvcc} # $CUDACXX if set, otherwise `nvcc`
CUDA_ARCHS= # Empty, use presets by default.
GLOBAL_CMAKE_OPTIONS=()

# Check if the correct number of arguments has been provided
function usage {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -v/--verbose: enable shell echo for debugging"
    echo "  -cuda: CUDA compiler (Defaults to \$CUDACXX if set, otherwise nvcc)"
    echo "  -cxx: Host compiler (Defaults to \$CXX if set, otherwise g++)"
    echo "  -arch: Target CUDA arches, e.g. \"60-real;70;80-virtual\" (Defaults to value in presets file)"
    echo
    echo "Examples:"
    echo "  $ PARALLEL_LEVEL=8 $0"
    echo "  $ PARALLEL_LEVEL=8 $0 -cxx g++-9"
    echo "  $ $0 -cxx clang++-8"
    echo "  $ $0 -cxx g++-8 -std 14 -arch 80-real -v -cuda /usr/local/bin/nvcc"
    echo "  $ $0 -cmake-options \"-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-Wfatal-errors\""
    exit 1
}

# Parse options

# Copy the args into a temporary array, since we will modify them and
# the parent script may still need them.
args=("$@")
while [ "${#args[@]}" -ne 0 ]; do
    case "${args[0]}" in
    -v | --verbose) VERBOSE=1; args=("${args[@]:1}");;
    -cxx)  HOST_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -cuda) CUDA_COMPILER="${args[1]}"; args=("${args[@]:2}");;
    -arch) CUDA_ARCHS="${args[1]}";    args=("${args[@]:2}");;
    -h | -help | --help) usage ;;
    *) echo "Unrecognized option: ${args[0]}"; usage ;;
    esac
done

# Convert to full paths:
HOST_COMPILER=$(which ${HOST_COMPILER})
CUDA_COMPILER=$(which ${CUDA_COMPILER})

if [ $VERBOSE ]; then
    set -x
fi

# Begin processing unsets after option parsing
set -u

cd ../

git clone --recursive https://github.com/cupy/cupy.git
cd cupy;
git submodule update --init --recursive
rm -rf third_party/cccl
ln -s ../cccl third_party/cccl

export CUPY_NVCC_GENERATE_CODE="${CUDA_ARCHS}"
CXX=${HOST_COMPILER} CUDA_PATH="$(whereis ${CUDA_COMPILER})" pip install -v -e .
