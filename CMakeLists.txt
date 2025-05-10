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

# Mark DuckDB's core include directory as a SYSTEM include path for your extension.
# This should be applied regardless of safety check level.
target_include_directories(${EXTENSION_NAME} SYSTEM
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/duckdb/src/include"
        # You might need to add other specific duckdb/third_party include paths here
        # if they also generate errors, e.g.:
        # "${CMAKE_CURRENT_LIST_DIR}/duckdb/third_party/fmt/include"
)

# Conditionally apply strict compiler flags to the static extension target
if(DEFINED ENV{LOCAL_SAFETY_CHECKS} AND "$ENV{LOCAL_SAFETY_CHECKS}" STREQUAL "1")
    message(STATUS "LOCAL_SAFETY_CHECKS is ON (set to '1'). Applying strict compiler flags and sanitizers to ${EXTENSION_NAME}.")
    target_compile_options(${EXTENSION_NAME} PRIVATE
        -g3                     # Enhanced debug information
        -fno-omit-frame-pointer # For more reliable stack traces with sanitizers

        -Wall                   # Enable many common, important warnings
        -Wextra                 # Enable additional useful warnings
        -Wpedantic              # Issue all warnings demanded by strict ISO C++

        -Werror                 # Treat all warnings (for this target's code) as errors

        # Specific warnings you want to enable for your code
        -Wshadow                # Warn when a local variable shadows another
        -Wconversion            # Warn on implicit conversions that may alter a value
        -Wsign-conversion       # Warn for implicit sign conversions
        -Wold-style-cast        # Warn for C-style casts
        -Wdouble-promotion      # Warn when float is promoted to double
        # -Wunused-parameter      # Warn on unused function parameters
        # -Wunused-variable       # Warn on unused local variables
        -Wno-unused-parameter   # Explicitly disable unused parameter warnings
        -Wno-unused-variable    # Explicitly disable unused variable warnings
        -Wimplicit-int-float-conversion # Warn on potentially lossy int to float conversions

        # -- Add Sanitizers Here --
        -fsanitize=address,undefined # Add ,leak if appropriate for your platform/needs
        -fno-sanitize-recover=all    # Crash immediately on sanitizer errors

        # Add any other specific warnings you find useful here
    )
endif()

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