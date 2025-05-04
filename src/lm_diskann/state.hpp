#pragma once

#include "duckdb.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unordered_set.hpp"

#include <queue>
#include <vector>
#include <utility> // For std::pair

namespace duckdb {

// --- Scan State ---
// Represents the state maintained during an index scan operation (e.g., a k-NN query).
// Inherits from DuckDB's base IndexScanState.
struct LMDiskannScanState : public IndexScanState {
    Vector query_vector_handle; // Handle to the query vector for lifetime management
    const float* query_vector_ptr; // Raw pointer to query data (assumed float)
    idx_t k;             // Number of neighbors requested

    // Candidate priority queue (min-heap based on distance)
    // Stores pairs of (distance, row_t)
    std::priority_queue<std::pair<float, row_t>,
                        std::vector<std::pair<float, row_t>>,
                        std::greater<std::pair<float, row_t>>> candidates;

    // Visited set (using row_t as key)
    duckdb::unordered_set<row_t> visited;

    // Top-k results found so far (for re-ranking if needed)
    // Store pairs of (exact_distance, row_t)
    std::vector<std::pair<float, row_t>> top_candidates;

    // Search parameters used for this scan
    uint32_t l_search;

    // Constructor
    LMDiskannScanState(const Vector &query, idx_t k_value, uint32_t l_search_value)
        : query_vector_handle(query), k(k_value), l_search(l_search_value) {
            // Assuming query is always FLOAT for now
            if (query.GetType().id() != LogicalTypeId::ARRAY ||
                ArrayType::GetChildType(query.GetType()).id() != LogicalTypeId::FLOAT) {
                throw BinderException("LMDiskannScanState: Query vector must be ARRAY<FLOAT>.");
            }
            // Ensure the query vector is flattened for direct access
            query_vector_handle.Flatten(1); // Assuming query is a single vector
            query_vector_ptr = FlatVector::GetData<float>(query_vector_handle);
        }

    ~LMDiskannScanState() override = default; // Default destructor
};

} // namespace duckdb

