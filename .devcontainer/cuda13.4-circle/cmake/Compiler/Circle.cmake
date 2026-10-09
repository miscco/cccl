# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# Unofficial Circle compiler information. Circle's version is 1.0.<build>,
# which is below the version gates in Compiler/GNU.cmake, so the GNU-like
# flags Circle does accept are set again here.

if(__COMPILER_CIRCLE)
  return()
endif()
set(__COMPILER_CIRCLE 1)

include(Compiler/GNU)

macro(__compiler_circle lang)
  __compiler_gnu(${lang})
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
  set(CMAKE_DEPFILE_FLAGS_${lang} "-MD -MT <DEP_TARGET> -MF <DEP_FILE>")
  set(CMAKE_${lang}_DEPFILE_FORMAT gcc)
  set(CMAKE_${lang}_DEPENDS_USE_COMPILER TRUE)
endmacro()
