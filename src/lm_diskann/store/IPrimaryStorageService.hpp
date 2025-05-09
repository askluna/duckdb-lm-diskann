#pragma once

#include "../common/duckdb_types.hpp"

#include <string> // For vector names if needed
#include <vector>

namespace diskann {
namespace store {

/**
 * @brief Interface for retrieving vector data from the primary relational store (e.g., a DuckDB table).
 *
 * This service abstracts the mechanism for fetching vector data corresponding to a given row_id,
 * which is necessary during index build or for operations requiring the original vector data.
 */
class IPrimaryStorageService {
	public:
	virtual ~IPrimaryStorageService() = default;

	/**
	 * @brief Fetches the vector data for a specific row_id.
	 *
	 * @param row_id The row identifier of the vector to fetch.
	 * @param vector_dim The expected dimensionality of the vector (for validation).
	 * @param vector_out Output vector (std::vector<float>) to store the fetched data.
	 *                   The vector will be resized appropriately.
	 * @return True if the vector was found and fetched successfully, false otherwise.
	 * @throw common::IOException on underlying storage errors.
	 * @throw common::InternalException if dimensionality mismatch or other inconsistencies.
	 */
	virtual bool GetVector(common::row_t row_id, common::idx_t vector_dim, std::vector<float> &vector_out) = 0;

	/**
	 * @brief Fetches multiple vectors given a list of row_ids.
	 *
	 * @param row_ids A list of row identifiers for the vectors to fetch.
	 * @param vector_dim The expected dimensionality of the vectors.
	 * @param vectors_out Output list of vectors. Each element will be populated.
	 *                    The outer vector will be resized to match row_ids.size().
	 *                    Inner vectors will be resized to vector_dim.
	 * @return True if all requested vectors were found and fetched, false if any failed.
	 *         (Alternatively, could return the number of vectors successfully fetched or throw on partial failure).
	 * @throw common::IOException on underlying storage errors.
	 * @throw common::InternalException if dimensionality mismatch or other inconsistencies.
	 */
	virtual bool GetVectors(const std::vector<common::row_t> &row_ids, common::idx_t vector_dim,
	                        std::vector<std::vector<float>> &vectors_out) = 0;

	// Potentially add methods for initializing the service with table/column names if needed,
	// or for iterating over all vectors during an initial bulk load.
	// virtual void Initialize(const std::string& table_name, const std::string& vector_column_name) = 0;
	// virtual bool GetNextVectorBatch(std::vector<common::row_t>& row_ids_out, std::vector<std::vector<float>>&
	// vectors_out, common::idx_t batch_size) = 0;
};

} // namespace store
} // namespace diskann