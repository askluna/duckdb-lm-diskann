/*******************************************************************************
 * @file LmDiskannIndex.cpp
 * @brief Implementation of the LmDiskannIndex class for DuckDB.
 * @details This file contains the implementation details for managing,
 *searching, inserting into, and deleting from an LM-DiskANN index within
 *DuckDB. It interacts with DuckDB's storage and execution systems.
 ******************************************************************************/
#include "LmDiskannIndex.hpp"

// Include refactored component headers
#include "../core/GraphManager.hpp"
#include "../core/Searcher.hpp"       // For PerformSearch
#include "../core/StorageManager.hpp" // For Load/PersistMetadata, GetEntryPointRowId etc.
#include "../core/distance.hpp"       // For distance/conversion functions
#include "../core/index_config.hpp"
#include "LmDiskannScanState.hpp"

// Include necessary DuckDB headers used in this file
#include "../core/Coordinator.hpp"                           // Include Coordinator
#include "duckdb/catalog/catalog_entry/duck_table_entry.hpp" // Required for table
#include "duckdb/common/constants.hpp"                       // For NumericLimits
#include "duckdb/common/file_system.hpp"                     // Required for FileSystem
#include "duckdb/common/helper.hpp"                          // For AlignValue
#include "duckdb/common/limits.hpp"                          // For NumericLimits
#include "duckdb/common/printer.hpp"
#include "duckdb/common/random_engine.hpp" // For GetSystemRandom
#include "duckdb/common/serializer/binary_deserializer.hpp"
#include "duckdb/common/serializer/binary_serializer.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp" // For Flatten, Slice
#include "duckdb/execution/index/fixed_size_allocator.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context.hpp"                  // Needed for Vacuum?
#include "duckdb/main/client_context_state.hpp"            // Required for GetCurrentClientContext
#include "duckdb/main/database.hpp"                        // Required for GetDatabase
#include "duckdb/main/database_manager.hpp"                // Required for DatabaseManager
#include "duckdb/parser/parsed_data/create_index_info.hpp" // For ArrayType info
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/planner/expression.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/index_storage_info.hpp"
#include "duckdb/storage/metadata/metadata_reader.hpp"
#include "duckdb/storage/metadata/metadata_writer.hpp"
#include "duckdb/storage/storage_manager.hpp" // Required for GraphManager
#include "duckdb/storage/table_io_manager.hpp"

#include <algorithm> // For std::sort, std::min, std::max
#include <cstring>   // For memcpy, memset
#include <map>       // For in-memory map placeholder
#include <random>    // For default_random_engine, uniform_int_distribution (used in GetRandomNodeID placeholder)
#include <set>       // For intermediate pruning steps (if RobustPrune uses it)
#include <vector>

// Required for LmDiskannIndex::ParseOptions
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"

// Forward declare concrete types if headers are not ready (temporary)
namespace diskann {
namespace store {
class DiskannShadowStorageService; // Assuming this is the concrete class
}
} // namespace diskann

namespace diskann {
namespace db {

// Implementation of ParseOptions moved here
core::LmDiskannConfig LmDiskannIndex::ParseOptions(const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options) {
	core::LmDiskannConfig config; // Starts with default values from core::LmDiskannConfigDefaults

	for (const auto &entry : options) {
		const ::duckdb::string &key_upper = ::duckdb::StringUtil::Upper(entry.first);
		const ::duckdb::Value &val = entry.second;

		if (key_upper == core::LmDiskannOptionKeys::METRIC) { // Use core::LmDiskannOptionKeys
			::duckdb::string metric_str = ::duckdb::StringUtil::Upper(val.ToString());
			if (metric_str == "L2") {
				config.metric_type = common::LmDiskannMetricType::L2;
			} else if (metric_str == "COSINE") {
				config.metric_type = common::LmDiskannMetricType::COSINE;
			} else if (metric_str == "IP") {
				config.metric_type = common::LmDiskannMetricType::IP;
			} else {
				throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
				                          ::duckdb::StringUtil::Format("Unsupported METRIC type '%s' for "
				                                                       "LM_DISKANN index. Supported types: "
				                                                       "L2, COSINE, IP",
				                                                       metric_str));
			}
		} else if (key_upper == core::LmDiskannOptionKeys::R) {
			config.r = val.GetValue<uint32_t>();
		} else if (key_upper == core::LmDiskannOptionKeys::L_INSERT) {
			config.l_insert = val.GetValue<uint32_t>();
		} else if (key_upper == core::LmDiskannOptionKeys::ALPHA) {
			config.alpha = val.GetValue<float>();
		} else if (key_upper == core::LmDiskannOptionKeys::L_SEARCH) {
			config.l_search = val.GetValue<uint32_t>();
		} else {
			throw ::duckdb::Exception(::duckdb::ExceptionType::INVALID_INPUT,
			                          ::duckdb::StringUtil::Format("Unknown option '%s' for LM_DISKANN index. Allowed "
			                                                       "options: METRIC, R, L_INSERT, ALPHA, L_SEARCH",
			                                                       entry.first));
		}
	}
	return config;
}

// --- LmDiskannIndex Constructor --- //

LmDiskannIndex::LmDiskannIndex(const ::duckdb::string &name, ::duckdb::IndexConstraintType index_constraint_type,
                               /**
                                * @brief Constructor for LmDiskannIndex.
                                * @param name Index name.
                                * @param index_constraint_type Type of constraint (e.g., UNIQUE).
                                * @param column_ids Physical column IDs covered by the index.
                                */
                               const ::duckdb::vector<::duckdb::column_t> &column_ids,
                               ::duckdb::TableIOManager &table_io_manager,
                               const ::duckdb::vector<::duckdb::unique_ptr<::duckdb::Expression>> &unbound_expressions,
                               ::duckdb::AttachedDatabase &db,
                               const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options,
                               const ::duckdb::IndexStorageInfo &storage_info, idx_t estimated_cardinality)
    : BoundIndex(name, LmDiskannIndex::TYPE_NAME, index_constraint_type, column_ids, table_io_manager,
                 unbound_expressions, db),
      db_state_(db, table_io_manager, unbound_expressions[0]->return_type),
      format_version_(core::LMDISKANN_CURRENT_FORMAT_VERSION)
// is_dirty_ is now managed by Coordinator
{
	// 1. Parse WITH clause options into the config struct
	core::LmDiskannConfig local_config = LmDiskannIndex::ParseOptions(options);

	// 2. Derive dimensions and node_vector_type from the indexed column type
	if (db_state_.indexed_column_type.id() != ::duckdb::LogicalTypeId::ARRAY ||
	    ::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type).id() == ::duckdb::LogicalTypeId::INVALID) {
		throw ::duckdb::BinderException("LM_DISKANN index can only be created on ARRAY types "
		                                "(e.g., FLOAT[N]).");
	}
	local_config.dimensions = ::duckdb::ArrayType::GetSize(db_state_.indexed_column_type);
	if (local_config.dimensions == 0) {
		throw ::duckdb::BinderException("LM_DISKANN index array dimensions cannot be zero.");
	}
	auto array_child_type = ::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type);
	if (array_child_type.id() == ::duckdb::LogicalTypeId::FLOAT) {
		local_config.node_vector_type = common::LmDiskannVectorType::FLOAT32;
	} else if (array_child_type.id() == ::duckdb::LogicalTypeId::TINYINT) {
		local_config.node_vector_type = common::LmDiskannVectorType::INT8;
	} else {
		throw ::duckdb::BinderException("LM_DISKANN index ARRAY child type must be FLOAT or TINYINT, found: " +
		                                array_child_type.ToString());
	}

	// 3. Validate all configuration parameters (including derived ones)
	ValidateParameters(local_config);

	// 4. Calculate node layout based on the fully populated config
	// Store these temporarily if Coordinator's config doesn't directly hold them
	// or if managers need them separately during construction before
	// Coordinator.
	core::NodeLayoutOffsets local_node_layout = CalculateLayoutInternal(local_config);
	idx_t local_block_size_bytes =
	    ::duckdb::AlignValue<idx_t, ::duckdb::Storage::SECTOR_SIZE>(local_node_layout.total_node_size);

	// Determine and create the index-specific directory path
	auto &fs = ::duckdb::FileSystem::Get(db);
	::duckdb::string db_lmd_root_path_str = db.GetName() + ".lmd_idx";
	this->index_data_path_ = fs.JoinPath(db_lmd_root_path_str, this->name);
	local_config.path = this->index_data_path_; // Ensure config has the path

	// Create Coordinator and its dependencies
	// These are placeholder creations. Actual managers will need proper
	// construction. The config passed to Coordinator should be the one it owns.
	// For now, Coordinator takes IndexConfig by const ref, so local_config is
	// fine. If Coordinator takes IndexConfig by unique_ptr, we'd move it.

	auto &buffer_manager = ::duckdb::BufferManager::GetBufferManager(db_state_.db);

	// Instantiate concrete managers - these are placeholders for actual
	// instantiation Their constructors would take necessary params like
	// buffer_manager, config, paths etc.
	std::unique_ptr<core::IStorageManager> storage_manager_ptr =
	    std::make_unique<core::StorageManager>(buffer_manager, local_config, local_node_layout);

	// ISearcher needs IStorageManager*
	std::unique_ptr<core::ISearcher> searcher_ptr = std::make_unique<core::Searcher>(storage_manager_ptr.get());

	// IGraphManager needs LmDiskannConfig, NodeLayoutOffsets, block_size, IStorageManager*, ISearcher*
	std::unique_ptr<core::IGraphManager> graph_manager_ptr = std::make_unique<core::GraphManager>(
	    local_config, local_node_layout, local_block_size_bytes, storage_manager_ptr.get(), searcher_ptr.get());

	// Shadow storage service - placeholder
	// It would need ClientContext or similar for DuckDB interaction.
	// auto current_context = ::duckdb::ClientContext::TryGetCurrent(db);
	std::unique_ptr<store::IShadowStorageService> shadow_service_ptr = nullptr;
	// std::make_unique<store::DiskannShadowStorageService>(/* potřebné parametry,
	// např. ClientContext */);

	coordinator_ = std::make_unique<core::Coordinator>(std::move(storage_manager_ptr), std::move(graph_manager_ptr),
	                                                   std::move(searcher_ptr), std::move(shadow_service_ptr),
	                                                   local_config // Pass the config
	);

	// Remove LmDiskannIndex's direct ownership of these if Coordinator now
	// manages them this->config_ = local_config; // Coordinator has a copy
	// this->node_layout_ = local_node_layout; // Coordinator's config/managers
	// handle this this->block_size_bytes_ = local_block_size_bytes; //
	// Coordinator's config/managers handle this this->node_manager_ = ... //
	// Coordinator has graph_manager_ this->graph_operations_ = ... // Logic
	// moves to GraphManager/Searcher

	if (!storage_info.IsValid()) {
		// This is a new index, create directories if they don't exist
		try {
			// Create the root directory for all LM-DiskANN indexes for this database
			// if it doesn't exist
			if (!fs.DirectoryExists(db_lmd_root_path_str)) {
				fs.CreateDirectory(db_lmd_root_path_str);
			}

			// Now create the specific directory for this index
			if (!fs.DirectoryExists(this->index_data_path_)) {
				fs.CreateDirectory(this->index_data_path_);
			} else {
				// Directory for this specific new index already exists, which is an
				// error.
				throw ::duckdb::CatalogException(::duckdb::StringUtil::Format("Cannot create LM-DiskANN index: directory '%s' "
				                                                              "already exists for new index '%s'. "
				                                                              "Please ensure the path is clear or drop "
				                                                              "potentially orphaned index artifacts.",
				                                                              this->index_data_path_, this->name));
			}
		} catch (const ::duckdb::PermissionException &e) {
			throw ::duckdb::PermissionException(
			    ::duckdb::StringUtil::Format("Failed to create directory structure for LM-DiskANN index '%s' "
			                                 "(path: '%s') due to insufficient permissions: %s",
			                                 this->name, this->index_data_path_, e.what()));
		} catch (const ::duckdb::IOException &e) {
			throw ::duckdb::IOException(::duckdb::StringUtil::Format("Failed to create directory structure for "
			                                                         "LM-DiskANN index '%s' (path: '%s'): %s",
			                                                         this->name, this->index_data_path_, e.what()));
		} catch (const std::exception &e) { // Catch-all for other potential issues during FS ops
			throw ::duckdb::IOException(
			    ::duckdb::StringUtil::Format("An unexpected error occurred while creating directory structure for "
			                                 "LM-DiskANN index '%s' (path: '%s'): %s",
			                                 this->name, this->index_data_path_, e.what()));
		}

		// Initialize a brand new index VIA COORDINATOR
		// InitializeNewIndex(estimated_cardinality); // OLD CALL
		coordinator_->InitializeIndex(estimated_cardinality);
	} else {
		// Load an existing index from storage VIA COORDINATOR
		// The config_.path should ideally be loaded from metadata within
		// LoadFromStorage For now, we assume it's correctly derived or will be set
		// by LoadFromStorage. A basic check here ensures the directory we expect
		// (or loaded) exists.
		if (this->index_data_path_.empty()) {
			// If LoadFromStorage was supposed to set this and didn't, or if it's
			// unexpectedly empty. This path could also be reconstructed here if not
			// persisted but derived. For now, if it's empty on load, it's an issue or
			// needs to be derived like above. Let's assume for loading,
			// LoadFromStorage would set it or it's derived same way. If it's derived
			// same way, it would be set before this 'else' branch.
		}
		if (!this->index_data_path_.empty() && !fs.DirectoryExists(this->index_data_path_)) {
			throw ::duckdb::IOException(
			    ::duckdb::StringUtil::Format("LM-DiskANN index directory '%s' not found for existing index '%s'. "
			                                 "The index files may be missing or corrupted.",
			                                 this->index_data_path_, this->name));
		}
		// LoadFromStorage(storage_info); // OLD CALL
		coordinator_->LoadIndex(this->index_data_path_);
	}

	// --- Logging --- //
	::duckdb::Printer::Print(::duckdb::StringUtil::Format(
	    "LM_DISKANN Index '%s': Metric=%s, Node Type=%s, Dim=%lld, R=%d, "
	    "L_insert=%d, Alpha=%.2f, L_search=%d, BlockSize=%lld, EdgeType=TERNARY",
	    name, LmDiskannMetricTypeToString(local_config.metric_type),
	    LmDiskannVectorTypeToString(local_config.node_vector_type), local_config.dimensions, local_config.r,
	    local_config.l_insert, local_config.alpha, local_config.l_search, local_block_size_bytes));
}

LmDiskannIndex::~LmDiskannIndex() = default;

// --- New Public Wrapper Methods ---
void LmDiskannIndex::PublicMarkDirty(bool dirty_state) {
	// This logic is now primarily in Coordinator::LoadIndex or its managers.
	// LmDiskannIndex might still need to translate IndexStorageInfo if
	// Coordinator or its StorageManager cannot directly use it.

	// Example of what used to be here:
	// core::LmDiskannMetadata metadata = LmDiskannStorageHelper::LoadMetadata(
	//     *this, storage_info.root_block, format_version_);
	// db_state_.metadata_ptr = storage_info.root_block;

	// // Validate loaded config against current config (dimensions, metric, etc.)
	// // This is important for index compatibility.
	// // ...

	// // Set graph state from loaded metadata
	// this->graph_entry_point_rowid_ = metadata.entry_point_rowid;
	// if (metadata.entry_point_rowid != -1 && metadata.entry_point_block_id !=
	// core::NULL_BLOCK_ID) {
	//     this->graph_entry_point_ptr_.Set(metadata.entry_point_block_id,
	//     metadata.entry_point_offset);
	// } else {
	//     this->graph_entry_point_ptr_.Invalidate();
	// }
	// this->delete_queue_head_ptr_ = metadata.delete_queue_head_ptr;

	// // Load the RowID to NodePointer map (persisted by GraphManager's
	// allocator) node_manager_->GetAllocator().Initialize(storage_info); // This
	// would load map
	// // Or if GraphManager has a specific load method:
	// // node_manager_->LoadPersistentState(storage_info);

	// // A freshly loaded index is not dirty
	// // is_dirty_ = false; // Coordinator handles this

	// // For now, this method in LmDiskannIndex can be a no-op or ensure
	// // Coordinator did its job.
	if (coordinator_) {
		// coordinator_->LoadIndex(this->index_data_path_); // This is now called
		// in constructor
		// this->graph_entry_point_rowid_ = coordinator_->GetGraphEntryPointRowId(); // Member removed
		// this->graph_entry_point_ptr_ = coordinator_->GetGraphEntryPointPtr();
	}
	std::cout << "LmDiskannIndex::PublicMarkDirty called (now delegates to "
	             "Coordinator primarily)"
	          << std::endl;
}

float LmDiskannIndex::PublicCalculateApproxDistance(const float *query_ptr,
                                                    ::duckdb::const_data_ptr_t compressed_neighbor_ptr) {
	return this->CalculateApproxDistance(query_ptr, compressed_neighbor_ptr);
}

void LmDiskannIndex::PublicCompressVectorForEdge(const float *input_vector,
                                                 ::duckdb::data_ptr_t output_compressed_vector) {
	this->CompressVectorForEdge(input_vector, output_compressed_vector);
}

void LmDiskannIndex::PublicConvertNodeVectorToFloat(::duckdb::const_data_ptr_t raw_node_vector,
                                                    float *float_vector_out) {
	this->ConvertNodeVectorToFloat(raw_node_vector, float_vector_out);
}

// --- BoundIndex Method Implementations (Updated) --- //

/**
 * @brief Appends a chunk of data to the index.
 * @details Called during initial table loading or bulk insertions.
 *          Iterates through the input chunk and calls Insert for each row.
 * @param lock Lock protecting the index during modification.
 * @param input DataChunk containing the vectors to append.
 * @param row_ids Corresponding row identifiers for the vectors in the input
 * chunk.
 * @return ErrorData indicating success or failure.
 */
::duckdb::ErrorData LmDiskannIndex::Append(::duckdb::IndexLock &lock, ::duckdb::DataChunk &input,
                                           ::duckdb::Vector &row_ids) {
	if (input.size() == 0) {
		return ::duckdb::ErrorData();
	}
	row_ids.Flatten(input.size());

	::duckdb::DataChunk input_chunk;
	input_chunk.InitializeEmpty({db_state_.indexed_column_type});
	::duckdb::Vector row_id_vector(::duckdb::LogicalType::ROW_TYPE);

	for (idx_t i = 0; i < input.size(); ++i) {
		input_chunk.Reset();
		input_chunk.data[0].Slice(input.data[0], i, i + 1);
		input_chunk.SetCardinality(1);

		row_id_vector.Slice(row_ids, i, i + 1);
		row_id_vector.Flatten(1);

		auto err = Insert(lock, input_chunk, row_id_vector);
		if (err.HasError()) {
			return err;
		}
	}
	// is_dirty_ flag is managed by the coordinator during Insert calls
	return ::duckdb::ErrorData();
}

/**
 * @brief Finalizes the dropping of the index.
 * @details Resets the allocator and clears internal pointers.
 * @param index_lock Lock protecting the index state.
 */
void LmDiskannIndex::CommitDrop(::duckdb::IndexLock &index_lock) {
	if (coordinator_) {
		coordinator_->HandleCommitDrop(); // Delegate drop logic
	}
	// Clear any remaining DuckDB-specific state if needed
	db_state_.metadata_ptr.Clear();
	// coordinator_ unique_ptr will handle deletion of components
}

/**
 * @brief Deletes entries from the index based on row identifiers.
 * @details Removes the node from the in-memory map, frees its block, adds it to
 * the delete queue, and potentially clears the entry point if it was deleted.
 * @param lock Lock protecting the index during modification.
 * @param entries DataChunk containing the vectors (ignored, deletion is by
 * row_id).
 * @param row_identifiers Vector of row identifiers to delete.
 */
void LmDiskannIndex::Delete(::duckdb::IndexLock &lock, ::duckdb::DataChunk &entries,
                            ::duckdb::Vector &row_identifiers) {
	row_identifiers.Flatten(entries.size());
	auto row_ids_data = ::duckdb::FlatVector::GetData<::duckdb::row_t>(row_identifiers);

	if (!coordinator_) {
		// Or handle this case more gracefully depending on application logic
		throw ::duckdb::InternalException("Coordinator is not initialized in LmDiskannIndex::Delete");
	}

	for (idx_t i = 0; i < entries.size(); ++i) {
		::duckdb::row_t row_id = row_ids_data[i];
		try {
			// Delegate the entire deletion logic to the coordinator
			coordinator_->Delete(row_id);
			// is_dirty flag is now managed by the coordinator internally

		} catch (const ::duckdb::NotImplementedException &e) {
			// Re-throw specific exceptions if needed
			throw;
		} catch (const std::exception &e) {
			// Log or handle errors reported by the coordinator
			::duckdb::Printer::Print(::duckdb::StringUtil::Format(
			    "Warning: Coordinator failed to delete node for row_id %lld: %s", row_id, e.what()));
			// Depending on requirements, might need to stop or continue processing
			// other deletions For now, just print a warning and continue.
		}
	}
	// No need to set is_dirty_ here, coordinator handles it.
}

/**
 * @brief Inserts a single vector into the index.
 * @details Allocates a new node, converts the input vector, finds neighbors
 * using search, connects the new node using RobustPrune, and updates neighbors
 * reciprocally.
 * @param lock Lock protecting the index during modification.
 * @param data DataChunk containing the single vector to insert.
 * @param row_ids Vector containing the single row identifier for the vector.
 * @return ErrorData indicating success or failure.
 */
::duckdb::ErrorData LmDiskannIndex::Insert(::duckdb::IndexLock &lock, ::duckdb::DataChunk &data,
                                           ::duckdb::Vector &row_ids) {
	if (data.size() == 0) {
		return ::duckdb::ErrorData();
	}
	D_ASSERT(data.size() == 1);
	D_ASSERT(data.ColumnCount() == 1);
	D_ASSERT(row_ids.GetVectorType() == VectorType::FLAT_VECTOR);

	if (!coordinator_) {
		return ::duckdb::ErrorData("Coordinator is not initialized in LmDiskannIndex::Insert");
	}

	auto &input_vector_handle = data.data[0];
	input_vector_handle.Flatten(1);
	auto row_id = ::duckdb::FlatVector::GetData<::duckdb::row_t>(row_ids)[0];
	::duckdb::const_data_ptr_t input_vector_raw_ptr = ::duckdb::FlatVector::GetData(input_vector_handle);

	// Determine vector type and dimensions from Coordinator's config
	const auto &config = coordinator_->GetConfig();
	idx_t dimensions = config.dimensions;
	auto node_vector_type = config.node_vector_type;

	// Prepare float vector (Coordinator::Insert expects float*)
	::duckdb::vector<float> input_vector_float_storage(dimensions); // Use config dimensions
	const float *input_vector_float_ptr = nullptr;

	try {
		if (node_vector_type == common::LmDiskannVectorType::FLOAT32) {
			// Check if input is actually float (should match config derived type)
			if (::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type).id() != ::duckdb::LogicalTypeId::FLOAT) {
				return ::duckdb::ErrorData("Type mismatch: Config expects FLOAT32 but input is not FLOAT.");
			}
			input_vector_float_ptr = reinterpret_cast<const float *>(input_vector_raw_ptr);
		} else if (node_vector_type == common::LmDiskannVectorType::INT8) {
			// Check if input is actually int8 (should match config derived type)
			if (::duckdb::ArrayType::GetChildType(db_state_.indexed_column_type).id() != ::duckdb::LogicalTypeId::TINYINT) {
				return ::duckdb::ErrorData("Type mismatch: Config expects INT8 but input is not TINYINT.");
			}
			// Use the existing conversion helper (which should ideally move to a
			// common place or be part of Coordinator/GraphManager)
			ConvertNodeVectorToFloat(input_vector_raw_ptr, input_vector_float_storage.data());
			input_vector_float_ptr = input_vector_float_storage.data();
		} else {
			return ::duckdb::ErrorData("Unsupported node vector type derived from config.");
		}

		if (!input_vector_float_ptr) {
			return ::duckdb::ErrorData("Internal error: Failed to obtain float pointer for input vector "
			                           "during insert preparation.");
		}

		// Delegate the core insertion logic to the coordinator
		coordinator_->Insert(input_vector_float_ptr, dimensions, row_id);
		// is_dirty_ flag is managed by coordinator

		return ::duckdb::ErrorData(); // Success

	} catch (const std::exception &e) {
		// Catch errors from coordinator or conversion
		return ::duckdb::ErrorData(
		    ::duckdb::StringUtil::Format("Failed during Insert for node %lld: %s", row_id, e.what()));
	}
}

/**
 * @brief Retrieves storage information about the index.
 * @details Used by DuckDB for checkpointing. Fills an IndexStorageInfo struct
 *          with allocator information. Metadata pointer persistence is handled
 * internally.
 * @param get_buffers Flag indicating whether buffer handles are needed (ignored
 * here).
 * @return IndexStorageInfo containing allocator details.
 */
::duckdb::IndexStorageInfo LmDiskannIndex::GetStorageInfo(bool get_buffers) {
	if (!coordinator_) {
		throw ::duckdb::InternalException("Coordinator not initialized in GetStorageInfo");
		// Or return an empty/invalid info struct?
		// ::duckdb::IndexStorageInfo empty_info;
		// empty_info.name = name;
		// return empty_info;
	}
	// Delegate to coordinator
	auto info = coordinator_->GetIndexStorageInfo();
	// Ensure the index name from LmDiskannIndex is set
	info.name = this->name;
	return info;
}

/**
 * @brief Estimates the in-memory size of the index.
 * @return Estimated size in bytes (allocator + map overhead).
 */
idx_t LmDiskannIndex::GetInMemorySize() {
	if (coordinator_) {
		return coordinator_->GetInMemorySize();
	}
	return 0; // Or throw an exception if coordinator_ should always be valid
}

/**
 * @brief Merges another index into this one.
 * @warning Not implemented for LM-DiskANN.
 * @param state Index lock.
 * @param other_index The index to merge into this one.
 * @return Always returns false (not implemented).
 */
bool LmDiskannIndex::MergeIndexes(::duckdb::IndexLock &state, BoundIndex &other_index) {
	throw ::duckdb::NotImplementedException("LmDiskannIndex::MergeIndexes not implemented");
	return false;
}

/**
 * @brief Performs vacuuming operations on the index.
 * @details Currently a placeholder; intended to process the deletion queue.
 * @param state Index lock.
 */
void LmDiskannIndex::Vacuum(::duckdb::IndexLock &state) {
	if (coordinator_) {
		coordinator_->PerformVacuum(); // Delegate vacuum logic
	}
	::duckdb::Printer::Print("LmDiskannIndex::Vacuum called (delegated to Coordinator).");
}

/**
 * @brief Verifies index integrity (placeholder) and returns a string
 * representation.
 * @param state Index lock.
 * @param only_verify If true, only perform verification without generating
 * string.
 * @return A string describing the index state.
 */
::duckdb::string LmDiskannIndex::VerifyAndToString(::duckdb::IndexLock &state, const bool only_verify) {
	::duckdb::string result = "LmDiskannIndex [Not Verified]";
	if (!coordinator_) {
		result += " - Coordinator not initialized!";
		return result;
	}
	const auto &current_config = coordinator_->GetConfig();
	result += ::duckdb::StringUtil::Format("\n - Config: Metric=%s, Type=%s, Dim=%lld, R=%d, L_insert=%d, "
	                                       "Alpha=%.2f, L_search=%d",
	                                       LmDiskannMetricTypeToString(current_config.metric_type),
	                                       LmDiskannVectorTypeToString(current_config.node_vector_type),
	                                       current_config.dimensions, current_config.r, current_config.l_insert,
	                                       current_config.alpha, current_config.l_search);
	if (coordinator_->GetGraphManager()) {
		// Removed GetAllocator().GetSegmentCount() due to IGraphManager not having GetAllocator()
		// result += ::duckdb::StringUtil::Format("\n - Allocator Blocks Used: %lld",
		//                                        coordinator_->GetGraphManager()->GetAllocator().GetSegmentCount());
		result += ::duckdb::StringUtil::Format("\n - Node Count (from GraphManager): %lld",
		                                       coordinator_->GetGraphManager()->GetNodeCount());
	} else {
		result += ::duckdb::StringUtil::Format("\n - GraphManager not available for counts.");
	}
	result += ::duckdb::StringUtil::Format("\n - Entry Point RowID (from Coordinator): %lld",
	                                       static_cast<long long>(coordinator_->GetGraphEntryPointRowId()));
	result += ::duckdb::StringUtil::Format("\n - Metadata Ptr: [BufferID=%lld, Offset=%lld, Meta=%d]",
	                                       db_state_.metadata_ptr.GetBufferId(), db_state_.metadata_ptr.GetOffset(),
	                                       db_state_.metadata_ptr.GetMetadata());
	return result;
}

/**
 * @brief Verifies allocator allocations (placeholder).
 * @param state Index lock.
 */
void LmDiskannIndex::VerifyAllocations(::duckdb::IndexLock &state) {
	// TODO: Check if allocator has verification method
	// if (db_state_.allocator) { db_state_.allocator->Verify(); }
}

/**
 * @brief Generates a constraint violation message.
 * @warning Not supported/applicable for LM-DiskANN similarity index.
 * @param verify_type Type of verification.
 * @param failed_index Index of the failed row.
 * @param input Input chunk causing violation.
 * @return Static error message.
 */
::duckdb::string LmDiskannIndex::GetConstraintViolationMessage(::duckdb::VerifyExistenceType verify_type,
                                                               idx_t failed_index, ::duckdb::DataChunk &input) {
	return "Constraint violation in LM_DISKANN index (Not supported)";
}

// --- Scan Method Implementations --- //

/**
 * @brief Initializes the state for an index scan (k-NN search).
 * @param context The client context.
 * @param query_vector The query vector (must be ARRAY<FLOAT>).
 * @param k The number of nearest neighbors to find.
 * @return A unique pointer to the initialized scan state.
 * @throws BinderException if query vector type/dimension is incorrect or k is
 * 0.
 */
::duckdb::unique_ptr<::duckdb::IndexScanState>
LmDiskannIndex::InitializeScan(::duckdb::ClientContext &context, const ::duckdb::Vector &query_vector, idx_t k) {
	if (query_vector.GetType().id() != ::duckdb::LogicalTypeId::ARRAY ||
	    ::duckdb::ArrayType::GetChildType(query_vector.GetType()).id() != ::duckdb::LogicalTypeId::FLOAT) {
		throw ::duckdb::BinderException("LM_DISKANN query vector must be ARRAY<FLOAT>.");
	}

	if (!coordinator_) {
		throw ::duckdb::InternalException("Coordinator not initialized in InitializeScan");
	}
	const auto &config = coordinator_->GetConfig(); // Get config via coordinator

	idx_t query_dims = ::duckdb::ArrayType::GetSize(query_vector.GetType());
	if (query_dims != config.dimensions) {
		throw ::duckdb::BinderException("Query vector dimension (%d) does not match index dimension (%d).", query_dims,
		                                config.dimensions);
	}
	if (k == 0) {
		throw ::duckdb::BinderException("Cannot perform index scan with k=0");
	}

	// Create the scan state. L_search comes from config.
	auto scan_state = ::duckdb::make_uniq<LmDiskannScanState>(query_vector, k, config.l_search);

	// PerformSearch is no longer called here.
	// Scan() will call Coordinator::Search().

	return std::move(scan_state);
}

/**
 * @brief Performs one step of the index scan (beam search).
 * @details Calls PerformSearch, extracts results from the scan state's priority
 * queue, sorts them, and fills the result vector with RowIDs.
 * @param state The current LmDiskannScanState.
 * @param result The output vector to store resulting RowIDs.
 * @return The number of results produced in this step (up to vector size or k).
 */
idx_t LmDiskannIndex::Scan(::duckdb::IndexScanState &state, ::duckdb::Vector &result) {
	auto &scan_state = state.Cast<LmDiskannScanState>();
	idx_t output_count = 0;
	auto result_data = ::duckdb::FlatVector::GetData<::duckdb::row_t>(result);

	if (!coordinator_) {
		throw ::duckdb::InternalException("Coordinator not initialized in Scan");
	}

	// Ensure the query vector pointer is valid (might be handled in constructor
	// of state)
	if (!scan_state.query_vector_ptr) {
		throw ::duckdb::InternalException("Query vector pointer not set in scan state");
	}

	try {
		// Delegate search to coordinator. Results are populated in
		// scan_state.result_row_ids.
		coordinator_->Search(scan_state.query_vector_ptr, scan_state.k, scan_state.result_row_ids, scan_state.l_search);

		// Copy results from the state vector to the DuckDB output vector
		for (const auto &row_id : scan_state.result_row_ids) {
			if (output_count < STANDARD_VECTOR_SIZE) { // Check against DuckDB vector size
				result_data[output_count++] = row_id;
			} else {
				break; // Stop if output vector is full
			}
		}
		// It's possible scan_state.result_row_ids has more than
		// STANDARD_VECTOR_SIZE results if k > STANDARD_VECTOR_SIZE. DuckDB handles
		// multiple Scan calls. We should remove the copied results from
		// scan_state.result_row_ids so the next Scan call provides the next batch.
		if (output_count > 0 && output_count <= scan_state.result_row_ids.size()) {
			scan_state.result_row_ids.erase(scan_state.result_row_ids.begin(),
			                                scan_state.result_row_ids.begin() + output_count);
		} else {
			// If no results were output, ensure the state vector is clear to prevent
			// infinite loops
			scan_state.result_row_ids.clear();
		}

	} catch (const std::exception &e) {
		// Handle errors from coordinator search
		throw ::duckdb::IOException(::duckdb::StringUtil::Format("Error during LM_DISKANN scan: %s", e.what()));
	}

	return output_count;
}

// --- Helper Method Implementations (Private to LmDiskannIndex) --- //

/**
 * @brief Initializes metadata and state for a brand new index.
 * @param estimated_cardinality Estimated number of rows (unused currently).
 */
void LmDiskannIndex::InitializeNewIndex(idx_t estimated_cardinality) {
	// This logic is now primarily in Coordinator::InitializeIndex
	// or delegated to its managers.
	// LmDiskannIndex might still need to do some high-level DuckDB setup
	// if not handled by Coordinator's dependencies.

	// Example of what used to be here:
	// core::LmDiskannMetadata metadata(config_.dimensions, config_.metric_type,
	//                                  config_.node_vector_type,
	//                                  format_version_);
	// metadata.entry_point_rowid = -1; // No entry point yet
	// metadata.count = 0;
	// metadata.max_node_id_allocated = 0; // Start from 0
	// metadata.delete_queue_head_ptr.block_id = core::NULL_BLOCK_ID;

	// // Initialize the allocator for a new index
	// node_manager_->GetAllocator().Initialize();

	// // Persist initial empty metadata
	// LmDiskannStorageHelper::PersistMetadataAndFreePages(
	//     *this, metadata, db_state_.metadata_ptr, nullptr);

	// // Set initial graph entry point (null/invalid for an empty graph)
	// this->graph_entry_point_rowid_ = metadata.entry_point_rowid;
	// this->graph_entry_point_ptr_.Invalidate();
	// this->delete_queue_head_ptr_.Invalidate();

	// // A new index is initially "dirty" as it's empty and needs to be saved if
	// // anything is added. Or, if metadata is persisted, it's clean.
	// // Let Coordinator handle this.
	// // is_dirty_ = false; // If metadata persistence counts as clean

	// // For now, this method in LmDiskannIndex can be a no-op or ensure
	// // Coordinator did its job.
	if (coordinator_) {
		// coordinator_->InitializeIndex(estimated_cardinality); // This is now
		// called in constructor Entry point might be accessible from coordinator
		// if needed here this->graph_entry_point_rowid_ =
		// coordinator_->GetGraphEntryPointRowId(); // REMOVE: Member doesn't exist
		// this->graph_entry_point_ptr_ = coordinator_->GetGraphEntryPointPtr(); //
		// REMOVE: Member doesn't exist
	}
	std::cout << "LmDiskannIndex::InitializeNewIndex called (now delegates to "
	             "Coordinator primarily)"
	          << std::endl;
}

/**
 * @brief Loads index state and configuration from existing storage.
 * @param storage_info Storage information provided by DuckDB during load.
 */
void LmDiskannIndex::LoadFromStorage(const ::duckdb::IndexStorageInfo &storage_info) {
	// This logic is now primarily in Coordinator::LoadIndex or its managers.
	// LmDiskannIndex might still need to translate IndexStorageInfo if
	// Coordinator or its StorageManager cannot directly use it.

	// Example of what used to be here:
	// core::LmDiskannMetadata metadata = LmDiskannStorageHelper::LoadMetadata(
	//     *this, storage_info.root_block, format_version_);
	// db_state_.metadata_ptr = storage_info.root_block;

	// // Validate loaded config against current config (dimensions, metric, etc.)
	// // This is important for index compatibility.
	// // ...

	// // Set graph state from loaded metadata
	// this->graph_entry_point_rowid_ = metadata.entry_point_rowid;
	// if (metadata.entry_point_rowid != -1 && metadata.entry_point_block_id !=
	// core::NULL_BLOCK_ID) {
	//     this->graph_entry_point_ptr_.Set(metadata.entry_point_block_id,
	//     metadata.entry_point_offset);
	// } else {
	//     this->graph_entry_point_ptr_.Invalidate();
	// }
	// this->delete_queue_head_ptr_ = metadata.delete_queue_head_ptr;

	// // Load the RowID to NodePointer map (persisted by GraphManager's
	// allocator) node_manager_->GetAllocator().Initialize(storage_info); // This
	// would load map
	// // Or if GraphManager has a specific load method:
	// // node_manager_->LoadPersistentState(storage_info);

	// // A freshly loaded index is not dirty
	// // is_dirty_ = false; // Coordinator handles this

	// // For now, this method in LmDiskannIndex can be a no-op or ensure
	// // Coordinator did its job.
	if (coordinator_) {
		// coordinator_->LoadIndex(this->index_data_path_); // This is now called
		// in constructor
		// this->graph_entry_point_rowid_ =
		// coordinator_->GetGraphEntryPointRowId(); // REMOVE: Member doesn't exist
		// this->graph_entry_point_ptr_ = coordinator_->GetGraphEntryPointPtr(); //
		// REMOVE: Member doesn't exist
	}
	std::cout << "LmDiskannIndex::LoadFromStorage called (now delegates to "
	             "Coordinator primarily)"
	          << std::endl;
}

// --- Distance Helper Wrappers --- //

/**
 * @brief Calculates approximate distance between a query vector and a
 * compressed neighbor vector.
 * @param query_ptr Pointer to the float query vector.
 * @param compressed_neighbor_ptr Pointer to the compressed (Ternary) neighbor
 * vector.
 * @return Approximate distance.
 */
float LmDiskannIndex::CalculateApproxDistance(const float *query_ptr,
                                              ::duckdb::const_data_ptr_t compressed_neighbor_ptr) {
	if (!coordinator_)
		throw ::duckdb::InternalException("Coordinator not initialized for CalculateApproxDistance");
	return core::CalculateApproxDistance(query_ptr, compressed_neighbor_ptr, coordinator_->GetConfig());
}

/**
 * @brief Compresses a float vector into the Ternary format for edge storage.
 * @param input_vector Pointer to the input float vector.
 * @param output_compressed_vector Pointer to the output buffer for the
 * compressed vector.
 */
void LmDiskannIndex::CompressVectorForEdge(const float *input_vector, ::duckdb::data_ptr_t output_compressed_vector) {
	if (!coordinator_)
		throw ::duckdb::InternalException("Coordinator not initialized for CompressVectorForEdge");
	if (!core::CompressVectorForEdge(input_vector, output_compressed_vector, coordinator_->GetConfig())) {
		throw ::duckdb::InternalException("Failed to compress vector into Ternary format.");
	}
}

/**
 * @brief Calculates exact distance between a query vector and a raw node
 * vector.
 * @tparam T_QUERY Type of the query vector elements (usually float).
 * @tparam T_NODE Type of the node vector elements (float or int8_t).
 * @param query_ptr Pointer to the query vector.
 * @param node_vector_ptr Pointer to the raw node vector data.
 * @return Exact distance.
 */
template <typename T_QUERY, typename T_NODE>
float LmDiskannIndex::CalculateExactDistance(const T_QUERY *query_ptr, ::duckdb::const_data_ptr_t node_vector_ptr) {
	if (!coordinator_)
		throw ::duckdb::InternalException("Coordinator not initialized for CalculateExactDistance");
	return CalculateDistance<T_QUERY, T_NODE>(query_ptr, reinterpret_cast<const T_NODE *>(node_vector_ptr),
	                                          coordinator_->GetConfig());
}

/**
 * @brief Converts a raw node vector (potentially int8_t) to a float vector.
 * @param raw_node_vector Pointer to the raw node vector data.
 * @param float_vector_out Pointer to the output buffer for the float vector.
 */
void LmDiskannIndex::ConvertNodeVectorToFloat(::duckdb::const_data_ptr_t raw_node_vector, float *float_vector_out) {
	if (!coordinator_)
		throw ::duckdb::InternalException("Coordinator not initialized for ConvertNodeVectorToFloat");
	const auto &current_config = coordinator_->GetConfig();
	if (current_config.node_vector_type == common::LmDiskannVectorType::FLOAT32) {
		memcpy(float_vector_out, raw_node_vector, current_config.dimensions * sizeof(float));
	} else if (current_config.node_vector_type == common::LmDiskannVectorType::INT8) {
		core::ConvertToFloat<int8_t>(reinterpret_cast<const int8_t *>(raw_node_vector), float_vector_out,
		                             current_config.dimensions);
	} else {
		throw ::duckdb::InternalException("Unsupported node vector type in ConvertNodeVectorToFloat.");
	}
}

// Explicitly instantiate templates used within this file
template float LmDiskannIndex::CalculateExactDistance<float, float>(const float *, ::duckdb::const_data_ptr_t);
template float LmDiskannIndex::CalculateExactDistance<float, int8_t>(const float *, ::duckdb::const_data_ptr_t);

// --- Insertion Helper --- //
// FindAndConnectNeighbors and its helpers like RobustPrune (member version) are
// removed.

// --- Deletion Helper --- //
// EnqueueDeletion (member version) and ProcessDeletionQueue are removed.

} // namespace db
} // namespace diskann
