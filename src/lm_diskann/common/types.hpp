#pragma once

// Include relevant DuckDB type headers
#include "duckdb/common/types.hpp"                  // For idx_t, row_t
#include "duckdb/execution/index/index_pointer.hpp" // For IndexPointer

namespace diskann {
namespace common {

// Using DuckDB's native types for core index operations where appropriate.
// This simplifies integration with the LmDiskannIndex layer.

// duckdb::idx_t is typically uint64_t
using idx_t = ::duckdb::idx_t;

// duckdb::row_t is typically int64_t
using row_t = ::duckdb::row_t;

// duckdb::IndexPointer is a specific struct used by DuckDB for block pointers.
using IndexPointer = ::duckdb::IndexPointer;

// You can add other common custom types here if needed, e.g.:
// struct Candidate {
//   float distance;
//   row_t id;
//   bool operator>(const Candidate& other) const { return distance >
//   other.distance; }
// };

} // namespace common
} // namespace diskann