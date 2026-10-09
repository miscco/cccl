# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/Circle)
__compiler_circle(C)

set(CMAKE_C_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c)

set(CMAKE_C99_STANDARD_COMPILE_OPTION "-std=c99")
set(CMAKE_C99_EXTENSION_COMPILE_OPTION "-std=gnu99")
set(CMAKE_C11_STANDARD_COMPILE_OPTION "-std=c11")
set(CMAKE_C11_EXTENSION_COMPILE_OPTION "-std=gnu11")
set(CMAKE_C17_STANDARD_COMPILE_OPTION "-std=c17")
set(CMAKE_C17_EXTENSION_COMPILE_OPTION "-std=gnu17")
set(CMAKE_C_STANDARD_LATEST 17)
