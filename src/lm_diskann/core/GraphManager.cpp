/**
 * @file GraphManager.cpp
 * @brief Implements the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "GraphManager.hpp"

#include "../common/ann.hpp"
#include "../common/duckdb_types.hpp"
#include "ISearcher.hpp"
#include "IStorageManager.hpp"
#include "distance.hpp" // For diskann::Distance* functions
#include "duckdb/common/limits.hpp"
#include "duckdb/common/random_engine.hpp"
#include "duckdb/execution/index/index_pointer.hpp" // For duckdb::IndexPointer definition
#include "index_config.hpp"
#include "ternary_quantization.hpp" // For diskann::ternary_quantization functions

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map> // Include map for RobustPrune internal logic if needed
#include <vector>

namespace diskann {
namespace core {

// Define a constant for an invalid IndexPointer (default constructed is fine as its data is 0)
const common::IndexPointer INVALID_INDEX_POINTER = common::IndexPointer();

// --- Constructor & Destructor ---

GraphManager::GraphManager(const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout,
                           common::idx_t block_size_bytes, IStorageManager *storage_manager, ISearcher *searcher)
    : config_(config), node_layout_(node_layout), block_size_bytes_(block_size_bytes),
      storage_manager_(storage_manager), searcher_(searcher),
      graph_entry_point_ptr_(INVALID_INDEX_POINTER), // Initialize with our defined invalid pointer
      graph_entry_point_rowid_(common::NumericLimits<common::row_t>::Maximum()) {
	if (!storage_manager_) {
		throw std::runtime_error("GraphManager: StorageManager pointer cannot be null.");
	}
	if (!searcher_) {
		throw std::runtime_error("GraphManager: Searcher pointer cannot be null.");
	}
}

// Destructor is defaulted in the header, no definition needed here.

// --- IGraphManager Interface Implementation ---

void GraphManager::InitializeEmptyGraph(const LmDiskannConfig &config_param, common::IndexPointer &entry_point_ptr_out,
                                        common::row_t &entry_point_rowid_out) {
	[[maybe_unused]] const auto &unused_config = config_param;
	rowid_to_node_ptr_map_.clear();
	graph_entry_point_ptr_ = INVALID_INDEX_POINTER; // Assign defined invalid pointer
	graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();

	entry_point_ptr_out = graph_entry_point_ptr_;
	entry_point_rowid_out = graph_entry_point_rowid_;
	// Comment: Coordinator is responsible for ensuring StorageManager also
	// initializes its state.
}

bool GraphManager::AddNode(common::row_t row_id, const float *vector_data, common::idx_t dimensions,
                           common::IndexPointer &node_ptr_out) {
	if (dimensions != config_.dimensions) {
		// Consider logging this error or making it a D_ASSERT
		// throw common::NotImplementedException("AddNode: Provided dimensions mismatch with index configuration.");
		return false; // Mismatch in dimensions
	}

	common::data_ptr_t new_node_block_data_ptr = nullptr;

	// 1. Call StorageManager to allocate a new node block, get IndexPointer (node_ptr_out)
	//    and a direct pointer to the block's data.
	bool allocation_success = storage_manager_->AllocateNodeBlock(row_id, node_ptr_out, new_node_block_data_ptr);

	if (!allocation_success || !new_node_block_data_ptr) {
		// Failed to allocate block or get a valid data pointer
		return false;
	}

	// 2. Initialize the node block (zero out, set neighbor count to 0).
	NodeAccessors::InitializeNodeBlock(new_node_block_data_ptr, block_size_bytes_, node_layout_);

	// 3. Get mutable raw vector pointer and copy/quantize vector_data.
	unsigned char *raw_vec_storage_ptr = NodeAccessors::GetRawNodeVectorMutable(new_node_block_data_ptr, node_layout_);
	if (!raw_vec_storage_ptr) {
		// This should not happen if InitializeNodeBlock and layout are correct.
		// If it does, the block might be unusable. Consider deallocating.
		// storage_manager_->DeallocateNodeBlock(node_ptr_out); // Potentially
		return false;
	}

	if (config_.node_vector_type == common::LmDiskannVectorType::FLOAT32) {
		std::memcpy(raw_vec_storage_ptr, vector_data, dimensions * sizeof(float));
	} else if (config_.node_vector_type == common::LmDiskannVectorType::INT8) {
		// TODO: Implement proper quantization from float to int8.
		// This requires scale/zero-point parameters, usually derived from the dataset.
		// For now, this is a placeholder direct cast which is incorrect for real use.
		// diskann::core::quantize_vector(vector_data, reinterpret_cast<int8_t*>(raw_vec_storage_ptr), dimensions, ...);
		for (common::idx_t i = 0; i < dimensions; ++i) {
			// Placeholder: simplistic cast, NOT proper quantization
			reinterpret_cast<int8_t *>(raw_vec_storage_ptr)[i] = static_cast<int8_t>(vector_data[i]);
		}
		// throw common::NotImplementedException("AddNode: INT8 quantization not fully implemented.");
	} else {
		// Unsupported vector type for storage.
		// storage_manager_->DeallocateNodeBlock(node_ptr_out); // Potentially
		// throw common::NotImplementedException("AddNode: Unsupported vector type for storage.");
		return false;
	}

	// 4. Update internal map.
	rowid_to_node_ptr_map_[row_id] = node_ptr_out;

	// 5. If this is the first node, set it as the entry point.
	if (rowid_to_node_ptr_map_.size() == 1) {
		SetEntryPoint(node_ptr_out, row_id);
	}

	// 6. Perform graph connection:
	//    - Find candidate neighbors.
	std::vector<common::row_t> initial_candidate_row_ids;
	if (graph_entry_point_ptr_ != INVALID_INDEX_POINTER) {
		// The number of candidates to fetch (e.g., L_search parameter for RobustPrune, often related to R)
		// Using config_.r * 2 as a placeholder for a search list size parameter.
		common::idx_t num_initial_candidates = config_.l_insert; // Use L_insert from config
		searcher_->SearchForInitialCandidates(vector_data, dimensions, config_, this, num_initial_candidates,
		                                      graph_entry_point_ptr_, initial_candidate_row_ids);
	}
	// else: No entry point yet, RobustPrune will handle an empty candidate list if SearchForInitialCandidates doesn't
	// run.

	//    - Call RobustPrune to select final neighbors.
	//      RobustPrune expects the `vector_data` (float* query) and modifies `initial_candidate_row_ids` in place.
	RobustPrune(node_ptr_out, vector_data, initial_candidate_row_ids, config_);

	// 7. Update the new node's neighbor list with the results from RobustPrune.
	NodeAccessors::SetNeighborCount(new_node_block_data_ptr, node_layout_,
	                                static_cast<uint16_t>(initial_candidate_row_ids.size()));
	common::row_t *new_node_neighbors_ptr =
	    NodeAccessors::GetNeighborIDsPtrMutable(new_node_block_data_ptr, node_layout_);
	if (!new_node_neighbors_ptr) {
		// This implies a serious issue with node layout or block data.
		// storage_manager_->DeallocateNodeBlock(node_ptr_out); // Potentially
		return false;
	}

	for (size_t i = 0; i < initial_candidate_row_ids.size(); ++i) {
		new_node_neighbors_ptr[i] = initial_candidate_row_ids[i];
	}
	// Initialize remaining neighbor slots if RobustPrune returned fewer than R.
	for (size_t i = initial_candidate_row_ids.size(); i < config_.r; ++i) {
		new_node_neighbors_ptr[i] = common::NumericLimits<common::row_t>::Maximum(); // Mark as invalid/empty
	}

	// 8. Mark the new node block as dirty.
	storage_manager_->MarkBlockDirty(node_ptr_out);

	// --- Step 8: Interconnect - Update neighbors to potentially link back ---
	// TODO: Review DiskANN/Vamana algorithm details for optimal Interconnect strategy.
	//       Simply calling RobustPrune symmetrically might lead to oscillations or suboptimal graph structure.
	//       This implementation is a basic placeholder for reciprocal connection attempt.

	// Use the final list of neighbors selected by RobustPrune for the new node.
	// Note: initial_candidate_row_ids was modified in-place by RobustPrune.
	const std::vector<common::row_t> &final_neighbors = initial_candidate_row_ids;
	std::vector<float> neighbor_vector_float_storage(config_.dimensions);

	for (const common::row_t &neighbor_rowid : final_neighbors) {
		if (neighbor_rowid == common::NumericLimits<common::row_t>::Maximum())
			continue;

		common::IndexPointer neighbor_ptr = INVALID_INDEX_POINTER;
		if (!TryGetNodePointer(neighbor_rowid, neighbor_ptr)) {
			// Neighbor might have been deleted concurrently? Log or continue.
			continue;
		}

		// Get neighbor's vector data (needed for its RobustPrune call)
		if (!GetNodeVector(neighbor_ptr, neighbor_vector_float_storage.data(), config_.dimensions)) {
			// Failed to get neighbor vector, cannot prune for it.
			continue;
		}

		// Create a candidate list for the neighbor, including the new node.
		std::vector<common::row_t> neighbor_candidates;
		// Add neighbor's current neighbors
		GetNeighbors(neighbor_ptr, neighbor_candidates);
		// Add the new node as a candidate
		neighbor_candidates.push_back(row_id); // row_id is the ID of the node we just added

		// Prune the neighbor's list (including the new node as a candidate)
		// This call modifies neighbor_candidates in-place.
		// It will also fetch mutable data for neighbor_ptr and update its block.
		RobustPrune(neighbor_ptr, neighbor_vector_float_storage.data(), neighbor_candidates, config_);

		// RobustPrune already marked neighbor_ptr dirty if changes were made.
	}

	return true;
}

bool GraphManager::GetNodeVector(common::IndexPointer node_ptr, float *vector_out, common::idx_t dimensions) const {
	if (dimensions != config_.dimensions) {
		return false;
	}
	if (node_ptr == INVALID_INDEX_POINTER) { // Compare with defined invalid pointer
		return false;
	}

	common::const_data_ptr_t node_block_data = storage_manager_->GetNodeBlockData(node_ptr);
	if (!node_block_data) {
		return false;
	}

	const unsigned char *raw_vector_ptr = NodeAccessors::GetRawNodeVector(node_block_data, node_layout_);
	if (!raw_vector_ptr) {
		return false;
	}

	// Use the common utility function, passing the node_vector_type from config_
	if (!diskann::common::ConvertRawVectorToFloat(raw_vector_ptr, vector_out, dimensions, config_.node_vector_type)) {
		return false;
	}
	return true;
}

bool GraphManager::GetNeighbors(common::IndexPointer node_ptr, std::vector<common::row_t> &neighbor_row_ids_out) const {
	neighbor_row_ids_out.clear();
	if (node_ptr == INVALID_INDEX_POINTER) { // Compare with defined invalid pointer
		return false;
	}

	common::const_data_ptr_t node_block_data = storage_manager_->GetNodeBlockData(node_ptr);
	if (!node_block_data) {
		return false;
	}

	uint16_t count = NodeAccessors::GetNeighborCount(node_block_data, node_layout_);
	const common::row_t *ids_ptr = NodeAccessors::GetNeighborIDsPtr(node_block_data, node_layout_);

	if (!ids_ptr) {
		// Should not happen if layout is correct and block is valid
		return false;
	}

	neighbor_row_ids_out.reserve(count);
	for (uint16_t i = 0; i < count; ++i) {
		if (ids_ptr[i] != common::NumericLimits<common::row_t>::Maximum()) { // Filter invalid/empty slots
			neighbor_row_ids_out.push_back(ids_ptr[i]);
		}
	}
	// No need to explicitly release node_block_data.
	return true;
}

void GraphManager::RobustPrune(common::IndexPointer node_to_connect_ptr, const float *node_to_connect_vector_data,
                               std::vector<common::row_t> &candidate_row_ids, const LmDiskannConfig &config_param) {

	if (candidate_row_ids.empty()) {
		return;
	}
	uint32_t max_neighbors = config_param.r; // Use 'r' from config

	// --- Step 1: Calculate distances and create initial candidate list ---
	std::vector<std::pair<float, common::row_t>> current_candidates_with_dist;
	current_candidates_with_dist.reserve(candidate_row_ids.size());
	std::vector<float> temp_candidate_vector(config_param.dimensions);

	for (common::row_t candidate_rowid : candidate_row_ids) {
		common::IndexPointer candidate_ptr_val = INVALID_INDEX_POINTER;
		if (!TryGetNodePointer(candidate_rowid, candidate_ptr_val) || candidate_ptr_val == INVALID_INDEX_POINTER) {
			continue;
		}

		// Check for self-connection BEFORE fetching data
		if (candidate_ptr_val == node_to_connect_ptr) {
			continue;
		}

		common::const_data_ptr_t candidate_block_data = storage_manager_->GetNodeBlockData(candidate_ptr_val);
		if (!candidate_block_data)
			continue;
		const unsigned char *candidate_raw_vector_ptr = NodeAccessors::GetRawNodeVector(candidate_block_data, node_layout_);

		if (!candidate_raw_vector_ptr ||
		    !diskann::common::ConvertRawVectorToFloat(candidate_raw_vector_ptr, temp_candidate_vector.data(),
		                                              config_param.dimensions, config_.node_vector_type)) {
			continue;
		}

		float dist = this->CalculateDistanceInternal(node_to_connect_vector_data, temp_candidate_vector.data(),
		                                             config_param.dimensions);
		current_candidates_with_dist.emplace_back(dist, candidate_rowid);
	}

	// --- Step 2: Sort candidates by distance and remove duplicates by row_id ---
	std::sort(current_candidates_with_dist.begin(), current_candidates_with_dist.end());

	current_candidates_with_dist.erase(std::unique(current_candidates_with_dist.begin(),
	                                               current_candidates_with_dist.end(),
	                                               [](const auto &a, const auto &b) { return a.second == b.second; }),
	                                   current_candidates_with_dist.end());

	// --- Step 3: Alpha Pruning ---
	std::vector<common::row_t> final_selected_neighbors_rowids;
	final_selected_neighbors_rowids.reserve(max_neighbors);

	// Need temporary storage for the float vectors of selected neighbors for the alpha check
	// Store as map for easier distance lookup? Or flat vector + lookup? Flat vector is simpler.
	std::vector<float> selected_neighbor_float_vectors_flat;
	selected_neighbor_float_vectors_flat.reserve(max_neighbors * config_param.dimensions);
	// Map to store distances from node_to_connect to selected neighbors for alpha check
	// This avoids recalculating or searching current_candidates_with_dist repeatedly
	std::map<common::row_t, float> selected_neighbor_distances;

	for (const auto &candidate_pair : current_candidates_with_dist) {
		if (final_selected_neighbors_rowids.size() >= max_neighbors) {
			break;
		}

		common::row_t p_rowid = candidate_pair.second; // Candidate 'p' to consider
		float dist_node_p = candidate_pair.first;      // Distance node_to_connect -> p

		bool p_should_be_added = true;

		// Fetch p's data (only needed if we perform alpha check)
		common::IndexPointer p_ptr_val = INVALID_INDEX_POINTER;
		if (!TryGetNodePointer(p_rowid, p_ptr_val) || p_ptr_val == INVALID_INDEX_POINTER) {
			continue; // Should not happen if candidate list was valid
		}
		common::const_data_ptr_t p_block_data = storage_manager_->GetNodeBlockData(p_ptr_val);
		if (!p_block_data)
			continue;
		const unsigned char *p_raw_vector_ptr = NodeAccessors::GetRawNodeVector(p_block_data, node_layout_);
		std::vector<float> p_float_vector(config_param.dimensions);
		if (!p_raw_vector_ptr ||
		    !diskann::common::ConvertRawVectorToFloat(p_raw_vector_ptr, p_float_vector.data(), config_param.dimensions,
		                                              config_.node_vector_type)) {
			continue;
		}

		// Alpha check: compare candidate 'p' with already selected neighbors 'r'
		for (size_t r_idx = 0; r_idx < final_selected_neighbors_rowids.size(); ++r_idx) {
			common::row_t r_rowid = final_selected_neighbors_rowids[r_idx];
			const float *r_float_vector_ptr = selected_neighbor_float_vectors_flat.data() + (r_idx * config_param.dimensions);

			// Distance between candidate p and selected neighbor r
			float dist_p_r =
			    this->CalculateDistanceInternal(p_float_vector.data(), r_float_vector_ptr, config_param.dimensions);

			// Get distance from node_to_connect to selected neighbor r (should be stored)
			float dist_node_r = selected_neighbor_distances[r_rowid]; // Use map lookup

			// Vamana pruning condition: Check if r makes p redundant relative to node_to_connect
			// If alpha * dist(p, r) < dist(node_to_connect, p)
			if (config_param.alpha * dist_p_r < dist_node_p) {
				p_should_be_added = false;
				break; // Stop checking p against other r's, p is pruned
			}
			// Symmetrical check (Optional but part of original Vamana): Check if p makes r redundant
			// If alpha * dist(p, r) < dist(node_to_connect, r)
			// This would require removing r from final_selected_neighbors if true, making the loop more complex.
			// Let's stick to the one-way check for now (if r prunes p).
		}

		if (p_should_be_added) {
			final_selected_neighbors_rowids.push_back(p_rowid);
			// Store p's float vector and its distance from node_to_connect for future alpha checks
			selected_neighbor_float_vectors_flat.insert(selected_neighbor_float_vectors_flat.end(), p_float_vector.begin(),
			                                            p_float_vector.end());
			selected_neighbor_distances[p_rowid] = dist_node_p;
		}
	}
	// Update the input/output parameter with the final pruned list of row IDs
	candidate_row_ids = final_selected_neighbors_rowids;

	// --- Step 4: Update node_to_connect_ptr's neighbor list in storage ---
	// (This part was missing from the previous merge attempt but was in GraphOperations::RobustPrune)
	// NOTE: This assumes RobustPrune is *only* called from AddNode before the node block is written,
	// OR that AddNode passes a mutable block pointer. The interface version assumes it modifies
	// the state associated with node_to_connect_ptr.

	common::data_ptr_t node_to_connect_mutable_data = storage_manager_->GetMutableNodeBlockData(node_to_connect_ptr);
	if (!node_to_connect_mutable_data) {
		// Cannot update the node if we can't get mutable data
		// Maybe throw an error or log? This indicates a problem.
		std::cerr << "Warning: RobustPrune could not get mutable block data for node " << node_to_connect_ptr.Get()
		          << std::endl;
		return;
	}

	uint16_t final_count = static_cast<uint16_t>(final_selected_neighbors_rowids.size());
	NodeAccessors::SetNeighborCount(node_to_connect_mutable_data, node_layout_, final_count);
	common::row_t *dest_ids_ptr = NodeAccessors::GetNeighborIDsPtrMutable(node_to_connect_mutable_data, node_layout_);
	// TODO: Need to handle compressed edge data if use_ternary_quantization is true.
	// The logic from GraphOperations::RobustPrune that copied compressed data needs
	// to be integrated here, using CompressFloatVectorForEdge.

	if (config_param.use_ternary_quantization) {
		// Placeholder: Need to fetch vector data for each final neighbor, compress it, and store it.
		// common::data_ptr_t dest_pos_planes_base = node_to_connect_mutable_data + node_layout_.neighbor_pos_planes_offset;
		// common::data_ptr_t dest_neg_planes_base = node_to_connect_mutable_data + node_layout_.neighbor_neg_planes_offset;
		// idx_t plane_size_bytes = node_layout_.ternary_edge_size_bytes / 2; // Assuming equal size planes
		// std::vector<float> temp_final_neighbor_vec(config_param.dimensions);
		// std::vector<uint8_t> temp_compressed_edge(node_layout_.ternary_edge_size_bytes);

		// for (uint16_t i = 0; i < final_count; ++i) {
		//     dest_ids_ptr[i] = final_selected_neighbors_rowids[i];
		//     common::IndexPointer final_neighbor_ptr = INVALID_INDEX_POINTER;
		//     if (TryGetNodePointer(final_selected_neighbors_rowids[i], final_neighbor_ptr)) {
		//         if (GetNodeVector(final_neighbor_ptr, temp_final_neighbor_vec.data(), config_param.dimensions)) {
		//             if (CompressFloatVectorForEdge(temp_final_neighbor_vec.data(), temp_compressed_edge.data(),
		//             config_param.dimensions)) {
		//                 // Assuming compressed edge data is structured [pos_plane | neg_plane]
		//                 memcpy(dest_pos_planes_base + i * plane_size_bytes, temp_compressed_edge.data(),
		//                 plane_size_bytes); memcpy(dest_neg_planes_base + i * plane_size_bytes,
		//                 temp_compressed_edge.data() + plane_size_bytes, plane_size_bytes);
		//             } else {
		//                  // Handle compression failure - zero out planes?
		//                 memset(dest_pos_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//                 memset(dest_neg_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//             }
		//         } else {
		//              // Handle vector fetch failure
		//                 memset(dest_pos_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//                 memset(dest_neg_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//         }
		//     } else {
		//          // Handle node pointer lookup failure
		//         dest_ids_ptr[i] = common::NumericLimits<common::row_t>::Maximum(); // Mark ID as invalid
		//         memset(dest_pos_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//         memset(dest_neg_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//     }
		// }
		// Fill remaining slots
		// for (uint16_t i = final_count; i < config_param.r; ++i) {
		//     dest_ids_ptr[i] = common::NumericLimits<common::row_t>::Maximum();
		//     memset(dest_pos_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		//     memset(dest_neg_planes_base + i * plane_size_bytes, 0, plane_size_bytes);
		// }
		throw std::runtime_error("RobustPrune with ternary edge compression not fully implemented yet.");
	} else {
		// Just store RowIDs if not using ternary quantization for edges
		for (uint16_t i = 0; i < final_count; ++i) {
			dest_ids_ptr[i] = final_selected_neighbors_rowids[i];
		}
		// Clear out remaining neighbor slots
		for (uint16_t i = final_count; i < config_param.r; ++i) {
			dest_ids_ptr[i] = common::NumericLimits<common::row_t>::Maximum();
		}
	}

	storage_manager_->MarkBlockDirty(node_to_connect_ptr);
}

void GraphManager::SetEntryPoint(common::IndexPointer node_ptr, common::row_t row_id) {
	if ((node_ptr == INVALID_INDEX_POINTER) != (row_id == common::NumericLimits<common::row_t>::Maximum())) {
		// It's an inconsistent state if one is valid and the other is not.
		// Consider D_ASSERT or logging a warning here.
	}
	graph_entry_point_ptr_ = node_ptr;
	graph_entry_point_rowid_ = row_id;
}

common::IndexPointer GraphManager::GetEntryPointPointer() const {
	return graph_entry_point_ptr_;
}

common::row_t GraphManager::GetEntryPointRowId() const {
	return graph_entry_point_rowid_;
}

void GraphManager::HandleNodeDeletion(common::row_t row_id) {
	if (row_id == graph_entry_point_rowid_) {
		graph_entry_point_ptr_ = INVALID_INDEX_POINTER;
		graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
	}
	// Comment: Actual removal from neighbor lists of other nodes is complex and
	// usually part of a separate process (e.g., vacuuming, or search-time
	// filtering of deleted nodes). This function primarily ensures a deleted node
	// isn't the active entry point and removes it from rowid_to_node_ptr_map_.
	// The actual freeing of the block in storage is handled by StorageManager via Coordinator.
	rowid_to_node_ptr_map_.erase(row_id);
}

void GraphManager::FreeNode(common::row_t row_id) {
	// This is a logical removal from GraphManager's tracking.
	// Coordinator must inform StorageManager to actually reclaim/mark the block
	// as free.
	if (rowid_to_node_ptr_map_.erase(row_id) > 0) {
		// If the freed node was the entry point, reset entry point to invalid.
		if (row_id == graph_entry_point_rowid_) {
			SetEntryPoint(INVALID_INDEX_POINTER, common::NumericLimits<common::row_t>::Maximum());
		}
	} // else: node was not found in map, nothing to erase.
}

common::idx_t GraphManager::GetNodeCount() const {
	return rowid_to_node_ptr_map_.size();
}

common::idx_t GraphManager::GetInMemorySize() const {
	common::idx_t size = sizeof(*this);
	size += rowid_to_node_ptr_map_.size() * (sizeof(common::row_t) + sizeof(common::IndexPointer));
	// Add size of storage_manager_ pointer, searcher_ pointer (negligible for pointers themselves)
	// The actual node data is managed by storage_manager_ and not part of GraphManager's direct in-memory size.
	return size;
}

void GraphManager::Reset() {
	rowid_to_node_ptr_map_.clear();
	graph_entry_point_ptr_ = INVALID_INDEX_POINTER;
	graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
	// Comment: Coordinator is responsible for ensuring StorageManager also
	// resets its state (e.g., clear all blocks).
}

// --- Non-interface public methods ---

bool GraphManager::TryGetNodePointer(common::row_t row_id, common::IndexPointer &node_ptr_out) const {
	auto iter = rowid_to_node_ptr_map_.find(row_id);
	if (iter != rowid_to_node_ptr_map_.end()) {
		node_ptr_out = iter->second;
		return true;
	}
	node_ptr_out = INVALID_INDEX_POINTER;
	return false;
}

common::row_t GraphManager::GetRandomNodeID(common::RandomEngine &engine) {
	if (rowid_to_node_ptr_map_.empty()) {
		return common::NumericLimits<common::row_t>::Maximum();
	}
	auto map_size_val = rowid_to_node_ptr_map_.size();
	if (map_size_val == 0) {
		return common::NumericLimits<common::row_t>::Maximum();
	}
	// Use NextRandom to get a double in [0,1), then scale and cast.
	// This ensures a value in [0, map_size_val - 1].
	common::idx_t random_idx = static_cast<common::idx_t>(engine.NextRandom() * map_size_val);
	// Defensive check for the case where NextRandom() is 1.0 (should be < 1.0)
	if (random_idx >= map_size_val) {
		random_idx = map_size_val - 1;
	}

	auto iter = rowid_to_node_ptr_map_.begin();
	std::advance(iter, random_idx);
	return iter->first;
}

// --- Entry Point Management Logic (Moved from GraphOperations) ---

/**
 * @brief Selects a valid entry point for starting a search.
 * @details If the current entry point is invalid or deleted, selects a new random one.
 *          The caller (Coordinator/Orchestrator) is responsible for marking the index
 *          as dirty if the entry point changes.
 * @param engine A random engine to use for selecting a random node if needed.
 * @return A valid row_t to be used as a search entry point, or NumericLimits<row_t>::Maximum() if no nodes exist.
 */
common::row_t GraphManager::SelectEntryPointForSearch(common::RandomEngine &engine) {
	bool entry_point_updated = false;
	if (graph_entry_point_rowid_ != common::NumericLimits<common::row_t>::Maximum()) {
		common::IndexPointer ptr_check;
		// Verify current entry point is still valid in the node manager's map
		if (this->TryGetNodePointer(graph_entry_point_rowid_, ptr_check)) {
			// Current entry point is valid, return it.
			graph_entry_point_ptr_ = ptr_check; // Ensure pointer is cached correctly
			return graph_entry_point_rowid_;
		}
		// Entry point rowid was set but is no longer in the map (deleted)
		graph_entry_point_ptr_ = INVALID_INDEX_POINTER;
		graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
		entry_point_updated = true;
	}

	if (this->GetNodeCount() == 0) {
		// No nodes exist, ensure entry point is invalid and return invalid rowid.
		if (graph_entry_point_rowid_ != common::NumericLimits<common::row_t>::Maximum()) {
			graph_entry_point_ptr_ = INVALID_INDEX_POINTER;
			graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
			entry_point_updated = true; // Technically it changed to invalid
		}
		// Inform caller? Or let caller handle the invalid rowid return.
		return common::NumericLimits<common::row_t>::Maximum();
	}

	// Current entry point is invalid, and nodes exist, so select a new random one.
	common::row_t random_id = this->GetRandomNodeID(engine);
	if (random_id != common::NumericLimits<common::row_t>::Maximum()) {
		common::IndexPointer random_ptr;
		// Must call TryGetNodePointer again as GetRandomNodeID only returns row_id
		if (this->TryGetNodePointer(random_id, random_ptr)) {
			this->SetEntryPoint(random_ptr, random_id);
			entry_point_updated = true;
			// Caller (Coordinator) needs to mark dirty if entry_point_updated is true.
			return random_id;
		}
	}

	// Fallback: If random selection somehow failed to get a valid pointer.
	// This indicates an inconsistency. Should ideally not happen.
	// Try iterating the map - this is slow and discouraged.
	if (graph_entry_point_rowid_ == common::NumericLimits<common::row_t>::Maximum()) {
		if (!rowid_to_node_ptr_map_.empty()) {
			auto first_node = rowid_to_node_ptr_map_.begin();
			this->SetEntryPoint(first_node->second, first_node->first);
			entry_point_updated = true;
			return first_node->first;
		}
	}

	// If absolutely no entry point could be found (e.g., map became empty concurrently?)
	// ensure state is invalid.
	graph_entry_point_ptr_ = INVALID_INDEX_POINTER;
	graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
	return common::NumericLimits<common::row_t>::Maximum();
}

// --- Private Helper Methods ---

float GraphManager::CalculateDistanceInternal(common::IndexPointer node1_ptr, common::IndexPointer node2_ptr) const {
	std::vector<float> vec1(config_.dimensions);
	std::vector<float> vec2(config_.dimensions);

	if (!GetNodeVector(node1_ptr, vec1.data(), config_.dimensions) ||
	    !GetNodeVector(node2_ptr, vec2.data(), config_.dimensions)) {
		return common::NumericLimits<float>::Maximum();
	}
	return CalculateDistanceInternal(vec1.data(), vec2.data(), config_.dimensions);
}

float GraphManager::CalculateDistanceInternal(const float *vec1, const float *vec2, common::idx_t dimensions) const {
	if (dimensions == 0)
		return 0.0f;

	// Use the unified ComputeExactDistanceFloat function from distance.hpp
	return diskann::core::ComputeExactDistanceFloat(vec1, vec2, dimensions, config_.metric_type);

	// The old switch statement is replaced by the call above.
	// switch (config_.metric_type) {
	// case LmDiskannMetricType::L2:
	// 	return diskann::DistanceL2(vec1, vec2, dimensions); // Removed core:: assuming in diskann:: or global
	// case LmDiskannMetricType::IP:
	// 	return diskann::DistanceInnerProduct(vec1, vec2, dimensions); // Removed core::
	// case LmDiskannMetricType::COSINE:
	// 	return diskann::DistanceCosine(vec1, vec2, dimensions); // Removed core::
	// default:
	// 	return common::NumericLimits<float>::Maximum();
	// }
}

float GraphManager::CalculateDistanceInternal(common::IndexPointer node_ptr, const float *query_vec,
                                              common::idx_t dimensions) const {
	std::vector<float> node_vec(dimensions);
	common::const_data_ptr_t node_block_data = storage_manager_->GetNodeBlockData(node_ptr);
	if (!node_block_data) {
		return common::NumericLimits<float>::Maximum();
	}
	const unsigned char *raw_node_vector_ptr = NodeAccessors::GetRawNodeVector(node_block_data, node_layout_);
	// Use the common utility function, passing the node_vector_type from config_
	if (!raw_node_vector_ptr || !diskann::common::ConvertRawVectorToFloat(raw_node_vector_ptr, node_vec.data(),
	                                                                      dimensions, config_.node_vector_type)) {
		return common::NumericLimits<float>::Maximum();
	}
	return CalculateDistanceInternal(node_vec.data(), query_vec, dimensions);
}

bool GraphManager::CompressFloatVectorForEdge(const float *float_vector_in, common::data_ptr_t compressed_edge_data_out,
                                              common::idx_t dimensions) const {
	if (!config_.use_ternary_quantization) {
		return false;
	}
	if (!float_vector_in || !compressed_edge_data_out) {
		return false;
	}
	if (dimensions != config_.dimensions) {
		return false;
	}

	if (!config_.ternary_planes_ptr) {
		// Cannot compress without ternary planes defined in config
		return false;
	}

	// TODO: Reimplement this function using diskann::core::EncodeTernary.
	// Note that EncodeTernary outputs separate pos/neg planes, while this function
	// expects a single output buffer (compressed_edge_data_out). The layout
	// in compressed_edge_data_out needs to be defined (e.g., concatenated planes)
	// and the output of EncodeTernary needs to be written accordingly.
	// The function vectors_to_uint8_codebook called previously does not exist in the provided headers.

	/*
diskann::ternary_quantization::vectors_to_uint8_codebook( // Changed to diskann::ternary_quantization
	  float_vector_in,
	  1,
	  config_.dimensions,
	  config_.ternary_vector_alpha,
	  reinterpret_cast<const float *>(config_.ternary_planes_ptr),
	  compressed_edge_data_out
	  );
	*/

	// Placeholder return until implemented correctly
	return false;
}

} // namespace core
} // namespace diskann
