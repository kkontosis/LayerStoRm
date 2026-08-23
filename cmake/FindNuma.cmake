# FindNuma.cmake
# Finds libnuma and creates an IMPORTED target Numa::Numa.
#
# Result variables:
#   Numa_FOUND, Numa_INCLUDE_DIRS, Numa_LIBRARIES
#
# Imported targets:
#   Numa::Numa

find_path(Numa_INCLUDE_DIRS
  NAMES numa.h
  PATH_SUFFIXES include
)

find_library(Numa_LIBRARIES
  NAMES numa
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Numa
  REQUIRED_VARS Numa_INCLUDE_DIRS Numa_LIBRARIES
)

if(Numa_FOUND AND NOT TARGET Numa::Numa)
  add_library(Numa::Numa SHARED IMPORTED)
  set_target_properties(Numa::Numa PROPERTIES
    IMPORTED_LOCATION             "${Numa_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${Numa_INCLUDE_DIRS}"
  )
endif()

mark_as_advanced(Numa_INCLUDE_DIRS Numa_LIBRARIES)
