PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=lm_diskann
EXT_CONFIG=${PROJ_DIR}extension_config.cmake



# --- Add your custom CMake flags here ---
EXTRA_CMAKE_FLAGS := \
    -DCMAKE_BUILD_TYPE:STRING=Debug \
    -DCMAKE_C_COMPILER:FILEPATH=/opt/homebrew/opt/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER:FILEPATH=/opt/homebrew/opt/llvm/bin/clang++ \
    -DCMAKE_OSX_SYSROOT:PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.4.sdk \
    -DCMAKE_CXX_FLAGS:STRING="-Wno-deprecated-literal-operator  -pedantic-errors" \
    -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${PROJ_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -G Ninja


# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile