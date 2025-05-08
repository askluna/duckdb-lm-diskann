/**
 * @file Searcher.hpp
 * @brief Defines the interface and implementation for searching the LM-DiskANN graph.
 */
#pragma once

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "duckdb/common/atomic.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/execution/index/fixed_size_allocator.hpp" // For FixedSizeAllocator
#include "index_config.hpp"                                // For LmDiskannConfig, NodeLayoutOffsets

#include <queue> // For std::priority_queue
#include <vector>

// Forward declarations for interfaces used
namespace diskann {
namespace core {
class IGraphManager;
class IStorageManager;
} // namespace core
} // namespace diskann

// Forward declaration for scan state (assuming it stays in duckdb namespace for now)
// TODO: Decouple LmDiskannScanState if it holds DuckDB specific types directly.
namespace diskann {
namespace db {
struct LmDiskannScanState;
} // namespace db
} // namespace diskann

namespace diskann {
namespace core {

/**
 * @brief Performs the actual search algorithm on the graph.
 * @details This is a free function implementing the beam search logic.
 * @param scan_state Holds query information, search parameters (L), and results.
 * @param config The index configuration.
 * @param node_layout Layout offsets for accessing node data.
 * @param graph_manager Interface to access graph structure (neighbors) and node vectors.
 * @param storage_manager Interface to access raw node block data.
 * @param final_pass If true, calculate exact distances for top candidates; otherwise, use approximate distances.
 */
void PerformSearch(diskann::db::LmDiskannScanState &scan_state, const LmDiskannConfig &config,
                   const NodeLayoutOffsets &node_layout, IGraphManager &graph_manager, IStorageManager &storage_manager,
                   bool final_pass);

// Potentially move NodeCandidate and SearchResult structs here if they are
// core concepts

} // namespace core
} // namespace diskann
