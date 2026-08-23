# FindNCCL.cmake
# Finds the NCCL library and creates an IMPORTED target NCCL::NCCL.
#
# Searched hints:
#   NCCL_ROOT, ENV{NCCL_ROOT}, CUDA_TOOLKIT_ROOT_DIR
#
# Result variables:
#   NCCL_FOUND, NCCL_INCLUDE_DIRS, NCCL_LIBRARIES, NCCL_VERSION
#
# Imported targets:
#   NCCL::NCCL

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "NCCL header directory")
set(NCCL_LIB_DIR     $ENV{NCCL_LIB_DIR}     CACHE PATH "NCCL library directory")

list(APPEND _nccl_search_roots
  ${NCCL_ROOT}
  $ENV{NCCL_ROOT}
  $ENV{NCCL_ROOT_DIR}
  ${CUDAToolkit_ROOT}
  ${CUDA_TOOLKIT_ROOT_DIR}
)

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR} ${_nccl_search_roots}
  PATH_SUFFIXES include
)

find_library(NCCL_LIBRARIES
  NAMES nccl
  HINTS ${NCCL_LIB_DIR} ${_nccl_search_roots}
  PATH_SUFFIXES lib lib64
)

# Extract version from nccl.h
if(NCCL_INCLUDE_DIRS AND EXISTS "${NCCL_INCLUDE_DIRS}/nccl.h")
  file(READ "${NCCL_INCLUDE_DIRS}/nccl.h" _nccl_header)

  string(REGEX MATCH "#define NCCL_MAJOR[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_nccl_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define NCCL_MINOR[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_nccl_minor "${CMAKE_MATCH_1}")
  string(REGEX MATCH "#define NCCL_PATCH[ \t]+([0-9]+)" _ "${_nccl_header}")
  set(_nccl_patch "${CMAKE_MATCH_1}")

  if(_nccl_major)
    set(NCCL_VERSION "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_INCLUDE_DIRS NCCL_LIBRARIES
  VERSION_VAR   NCCL_VERSION
)

if(NCCL_FOUND AND NOT TARGET NCCL::NCCL)
  add_library(NCCL::NCCL SHARED IMPORTED)
  set_target_properties(NCCL::NCCL PROPERTIES
    IMPORTED_LOCATION             "${NCCL_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIRS}"
  )
  message(STATUS "NCCL ${NCCL_VERSION}: include=${NCCL_INCLUDE_DIRS}, lib=${NCCL_LIBRARIES}")
endif()

mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIB_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
