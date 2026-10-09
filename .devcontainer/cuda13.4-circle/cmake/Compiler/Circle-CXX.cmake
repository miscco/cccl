# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include(Compiler/Circle)
__compiler_circle(CXX)

set(CMAKE_CXX_COMPILE_OPTIONS_EXPLICIT_LANGUAGE -x c++)
set(CMAKE_CXX_COMPILE_OPTIONS_VISIBILITY_INLINES_HIDDEN "-fvisibility-inlines-hidden")

# Circle implements C++11 and later. -std=c++98 is rejected.
set(CMAKE_CXX11_STANDARD_COMPILE_OPTION "-std=c++11")
set(CMAKE_CXX11_EXTENSION_COMPILE_OPTION "-std=gnu++11")
set(CMAKE_CXX14_STANDARD_COMPILE_OPTION "-std=c++14")
set(CMAKE_CXX14_EXTENSION_COMPILE_OPTION "-std=gnu++14")
set(CMAKE_CXX17_STANDARD_COMPILE_OPTION "-std=c++17")
set(CMAKE_CXX17_EXTENSION_COMPILE_OPTION "-std=gnu++17")
set(CMAKE_CXX20_STANDARD_COMPILE_OPTION "-std=c++20")
set(CMAKE_CXX20_EXTENSION_COMPILE_OPTION "-std=gnu++20")
set(CMAKE_CXX23_STANDARD_COMPILE_OPTION "-std=c++23")
set(CMAKE_CXX23_EXTENSION_COMPILE_OPTION "-std=gnu++23")
set(CMAKE_CXX_STANDARD_LATEST 23)
