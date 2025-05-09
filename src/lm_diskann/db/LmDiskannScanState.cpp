#include "LmDiskannScanState.hpp"

#include "duckdb/common/types/vector_buffer.hpp"                 // For VectorBuffer and other Vector ops
#include "duckdb/common/vector_operations/vector_operations.hpp" // For VectorOperations

namespace diskann {
namespace db {

LmDiskannScanState::LmDiskannScanState(const ::duckdb::Vector &query_vec, // The input query vector
                                       common::idx_t k_param, common::idx_t l_search_param)
    : query_vector_storage(query_vec.GetType()), k(k_param),
      l_search(l_search_param) { // Initialize query_vector_storage

	// Store a copy of the query vector's data.
	// query_vector_storage is already initialized with the correct type.
	if (query_vec.GetVectorType() == ::duckdb::VectorType::FLAT_VECTOR) {
		::duckdb::VectorOperations::Copy(query_vec, query_vector_storage, query_vec.Size(), 0, 0);
		query_vector_ptr = ::duckdb::FlatVector::GetData<float>(query_vector_storage);
	} else {
		// If not flat, create a temporary flat vector, copy to it, then copy to storage.
		::duckdb::Vector flat_query_vec(query_vec.GetType());
		// flat_query_vec is already initialized by its constructor with type
		::duckdb::VectorOperations::Copy(query_vec, flat_query_vec, query_vec.Size(), 0, 0);

		// query_vector_storage is already initialized, just copy data into it.
		::duckdb::VectorOperations::Copy(flat_query_vec, query_vector_storage, flat_query_vec.Size(), 0, 0);
		query_vector_ptr = ::duckdb::FlatVector::GetData<float>(query_vector_storage);
	}
	// result_row_ids is default initialized (empty vector)
}

// The destructor is defaulted in the header, so no definition needed here unless custom logic is added.

} // namespace db
} // namespace diskann