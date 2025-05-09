#pragma once

// Include relevant DuckDB type headers
#include "duckdb/common/common.hpp"
#include "duckdb/common/constants.hpp" // For DConstants, block_id_t definition
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/common/random_engine.hpp" // For RandomEngine
#include "duckdb/common/types.hpp"         // For idx_t, row_t, data_ptr_t, const_data_ptr_t
#include "duckdb/common/unordered_map.hpp"
#include "duckdb/execution/index/index_pointer.hpp" // For IndexPointer
#include "duckdb/storage/block.hpp"
#include "duckdb/storage/storage_info.hpp"

namespace diskann {
namespace common {

// --- Core DuckDB Types Aliased for LM-DiskANN --- //

// duckdb::idx_t is typically uint64_t
using idx_t = ::duckdb::idx_t;

// duckdb::row_t is typically int64_t
using row_t = ::duckdb::row_t;

// duckdb::IndexPointer is a specific struct used by DuckDB for block pointers.
using IndexPointer = ::duckdb::IndexPointer;

// duckdb::block_id_t is typically int32_t
using block_id_t = ::duckdb::block_id_t;

// Constant for an invalid block ID, aliasing DuckDB's constant.
const block_id_t INVALID_BLOCK_ID = ::duckdb::DConstants::INVALID_INDEX;

// duckdb::data_ptr_t and const_data_ptr_t are aliases for unsigned char*
using data_ptr_t = ::duckdb::data_ptr_t;
using const_data_ptr_t = ::duckdb::const_data_ptr_t;

// --- Utility and Exception Types --- //

template <typename T>
using NumericLimits = ::duckdb::NumericLimits<T>;

using RandomEngine = ::duckdb::RandomEngine;

using Printer = ::duckdb::Printer;

// Common exception type to be used within the core LM-DiskANN logic.
using NotImplementedException = ::duckdb::NotImplementedException;
using IOException = ::duckdb::IOException;             // Added alias
using InternalException = ::duckdb::InternalException; // Added alias
// Add other common exception aliases if needed, e.g.:
// using IOException = ::duckdb::IOException;
// using InternalException = ::duckdb::InternalException;

// --- LM-DiskANN Specific Enums --- //

// You can add other common custom types here if needed, e.g.:
// struct Candidate {
//   float distance;
//   row_t id;
//   bool operator>(const Candidate& other) const { return distance >
//   other.distance; }
// };

} // namespace common
} // namespace diskann