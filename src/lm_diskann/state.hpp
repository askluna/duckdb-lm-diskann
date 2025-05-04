#pragma once

#include "duckdb.hpp"
#include "duckdb/storage/table/scan_state.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"

#include <queue>
#include <vector>
#include <utility> // For std::pair

namespace duckdb {

// --- Scan State ---
// Represents the state maintained during an index scan operation (e.g., a k-NN query).
// Inherits from DuckDB's base IndexScanState.
struct LMDiskannScanState : public IndexScanState {
    Vector query_vector; // Keep handle for lifetime
    const float* query_vector_ptr = nullptr; // Raw pointer to query data (assumed float)
    idx_t k;             // Number of neighbors requested
    idx_t l_search;      // Search list size (beam width)

    // Results collected during the scan
    std::vector<row_t> result_rowids; // Row IDs of potential candidates
    std::vector<float> result_scores; // Corresponding negated similarity scores

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

    // Constructor
    LMDiskannScanState(const Vector &query, idx_t k_p, idx_t l_search_p)
        : query_vector(query.GetType()), // Initialize with type first
          k(k_p),
          l_search(l_search_p)
    {
        if (query.GetType().id() != LogicalTypeId::ARRAY || ArrayType::GetChildType(query.GetType()).id() != LogicalTypeId::FLOAT) {
            throw BinderException("LM_DISKANN query vector must be a FLOAT ARRAY");
        }
        query_vector.Reference(query); // Now reference the data
        query_vector.Flatten(1); // Assuming single vector
        query_vector_ptr = FlatVector::GetData<float>(query_vector);
    }

    ~LMDiskannScanState() override = default; // Default destructor
};

} // namespace duckdb

