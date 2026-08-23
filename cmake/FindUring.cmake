# FindUring.cmake
# Finds liburing and creates an IMPORTED target Uring::Uring.
#
# Result variables:
#   Uring_FOUND, Uring_INCLUDE_DIRS, Uring_LIBRARIES
#
# Imported targets:
#   Uring::Uring

find_path(Uring_INCLUDE_DIRS
  NAMES liburing.h
  PATH_SUFFIXES include
)

find_library(Uring_LIBRARIES
  NAMES uring
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Uring
  REQUIRED_VARS Uring_INCLUDE_DIRS Uring_LIBRARIES
)

if(Uring_FOUND AND NOT TARGET Uring::Uring)
  add_library(Uring::Uring SHARED IMPORTED)
  set_target_properties(Uring::Uring PROPERTIES
    IMPORTED_LOCATION             "${Uring_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${Uring_INCLUDE_DIRS}"
  )
endif()

mark_as_advanced(Uring_INCLUDE_DIRS Uring_LIBRARIES)
