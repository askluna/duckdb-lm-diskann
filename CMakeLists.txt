cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For generating compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add project-wide CXX flags
add_compile_options(-Wno-deprecated-literal-operator)

# Set extension name here
set(TARGET_NAME lm_diskann)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# option(USE_SIMSIMD
#        "Use SIMSIMD library to sacrifice portability for vectorized search" OFF)

# Find the Highway package provided by vcpkg
# This makes Highway's targets available.
find_package(hwy CONFIG REQUIRED)
# find_package(simsimd CONFIG REQUIRED)

# For SimSIMD (header-only via vcpkg)
find_path(SIMSIMD_INCLUDE_DIR "simsimd/simsimd.h"
    HINTS ${VCPKG_INSTALL_INFO_DIR}/../include NO_DEFAULT_PATH) # More robust find
if(NOT SIMSIMD_INCLUDE_DIR)
    message(FATAL_ERROR "SimSIMD include directory not found. Check vcpkg installation.")
endif()
message(STATUS "Found SimSIMD include directory: ${SIMSIMD_INCLUDE_DIR}")

# Add SimSIMD include directory globally for all sources in this project
# Or, more targetedly to specific extension targets if preferred.
# For simplicity with DuckDB extensions, adding it to include_directories often works.
include_directories(${SIMSIMD_INCLUDE_DIR})

include_directories(src/include)

set(EXTENSION_SOURCES src/lm_diskann_extension.cpp)
add_subdirectory(src/lm_diskann)

# Build the static extension
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

# Link hwy::hwy to the static extension
target_link_libraries(${EXTENSION_NAME} hwy::hwy)

# Build the loadable extension, passing hwy::hwy as a library
build_loadable_extension(${TARGET_NAME} "hwy::hwy" ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")


message(STATUS "Current CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")