# Automatically find all .cpp files in the specified subdirectories
# These paths will be relative to CMAKE_CURRENT_SOURCE_DIR (e.g., "common/file.cpp")
file(GLOB LM_DISKANN_COMMON_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "common/*.cpp")
file(GLOB LM_DISKANN_CORE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "core/*.cpp")
file(GLOB LM_DISKANN_DB_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "db/*.cpp")
file(GLOB LM_DISKANN_STORE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "store/*.cpp")
file(GLOB LM_DISKANN_QUANTIZATION_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "quantization/*.cpp")
file(GLOB LM_DISKANN_SCHEDULER_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "scheduler/*.cpp")

set(LM_DISKANN_ABSOLUTE_SOURCES "")

# Make paths absolute by prepending CMAKE_CURRENT_SOURCE_DIR (path to src/lm_diskann)
foreach(FILE ${LM_DISKANN_COMMON_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()
foreach(FILE ${LM_DISKANN_CORE_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()
foreach(FILE ${LM_DISKANN_DB_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()
foreach(FILE ${LM_DISKANN_STORE_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()
foreach(FILE ${LM_DISKANN_QUANTIZATION_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()
foreach(FILE ${LM_DISKANN_SCHEDULER_SRCS})
    list(APPEND LM_DISKANN_ABSOLUTE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${FILE}")
endforeach()

# Append these absolute source paths to the EXTENSION_SOURCES variable from the parent scope
# This ensures that the main CMakeLists.txt, when it calls build_static_extension
# and build_loadable_extension, has all the necessary source files from this module.
set(EXTENSION_SOURCES
    ${EXTENSION_SOURCES}          # Keep existing sources from the parent scope
    ${LM_DISKANN_ABSOLUTE_SOURCES} # Add the absolute paths to sources found in this directory
    PARENT_SCOPE
)