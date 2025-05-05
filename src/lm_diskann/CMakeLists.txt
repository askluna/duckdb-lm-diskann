
set(EXTENSION_SOURCES
        ${EXTENSION_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/distance.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/node.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/search.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/state.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/lm_diskann_index.cpp
        PARENT_SCOPE
)