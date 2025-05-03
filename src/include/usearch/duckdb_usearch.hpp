#pragma once

// Include the main Usearch stubs
#include "index.hpp"
#include "index_dense.hpp"

// No need to include index_plugins.hpp as its functionality (metrics, executors, etc.)
// should be stubbed out within index.hpp and index_dense.hpp or isn't called by HNSW.
// No need for the preprocessor defines like USEARCH_USE_SIMSIMD in the stub. 