# List source files for the lm_diskann library target
set(LM_DISKANN_SOURCES
    common/ann_utils.cpp
    core/index_config.cpp
    core/distance.cpp
    core/GraphManager.cpp
    core/Searcher.cpp
    core/ternary_quantization.cpp
    db/LmDiskannIndex.cpp
    db/LmDiskannScanState.cpp
    store/FixedSizeAllocator.cpp
    store/StorageManager.cpp
    # Add other .cpp files as needed
)

set(EXTENSION_SOURCES
        ${EXTENSION_SOURCES}
        PARENT_SCOPE
)