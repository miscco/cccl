# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Unofficial compiler detection for the Circle C++ compiler
# (https://www.circle-lang.org/). Circle is not part of upstream CMake.
# It defines __clang__ and __GNUC__, so this test must run before those IDs.

set(_compiler_id_pp_test "defined(__CIRCLE_LANG__)")

# `circle -dumpversion` is <__circle_major__>.<__circle_minor__>.<__circle_build__>
# (for example 1.0.234).
set(_compiler_id_version_compute "
# define @PREFIX@COMPILER_VERSION_MAJOR @MACRO_DEC@(__circle_major__)
# define @PREFIX@COMPILER_VERSION_MINOR @MACRO_DEC@(__circle_minor__)
# define @PREFIX@COMPILER_VERSION_PATCH @MACRO_DEC@(__circle_build__)")
