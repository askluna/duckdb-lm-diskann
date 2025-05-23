cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For generating compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add project-wide CXX flags
if (NOT MSVC)
    add_compile_options(-Wno-deprecated-literal-operator)
endif()

# Set extension name here
set(TARGET_NAME lm_diskann)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# -- vcpkg packages --
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
find_package(hwy CONFIG REQUIRED)
find_package(cista CONFIG REQUIRED)

# -- project sources --
include_directories(src/include)
set(EXTENSION_SOURCES src/lm_diskann_extension.cpp)
add_subdirectory(src/lm_diskann)


build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

target_include_directories(${EXTENSION_NAME}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}"                   # Project root for HWY_TARGET_INCLUDE
        "${CMAKE_CURRENT_LIST_DIR}/src/include"       # If your extension needs headers from here
        "${SIMSIMD_INCLUDE_DIR}"                      # For SimSIMD headers
)

# -- link libraries --
target_link_libraries(${EXTENSION_NAME} hwy::hwy)
target_link_libraries(${EXTENSION_NAME} cista::cista)

# -- build the loadable extension --
build_loadable_extension(${TARGET_NAME} "hwy::hwy cista::cista" ${EXTENSION_SOURCES})


install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")


message(STATUS "Current CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")