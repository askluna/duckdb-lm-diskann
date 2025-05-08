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


include_directories(src/include)

set(EXTENSION_SOURCES src/lm_diskann_extension.cpp)
add_subdirectory(src/lm_diskann)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")


message(STATUS "Current CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")