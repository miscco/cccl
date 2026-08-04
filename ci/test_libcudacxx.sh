#!/usr/bin/env bash

cd "$(dirname "${BASH_SOURCE[0]}")"
# shellcheck source=ci/build_common.sh
source "./build_common.sh"

print_environment_details

"./build_libcudacxx.sh" "$@" -cmake-options "-DLIBCUDACXX_KEEP_TEST_ARTIFACTS=ON"

test_preset "libcudacxx (CTest)" "libcudacxx-ctest"

sccache -z > /dev/null || :
test_preset "libcudacxx (lit)" "libcudacxx-lit"
sccache --show-adv-stats || :

print_time_summary
