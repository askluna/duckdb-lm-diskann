/**
 * @file GraphManager.cpp
 * @brief Implements the GraphManager class for managing node allocations,
 *        RowID mappings, and raw node data access for the LM-DiskANN index.
 */
#include "GraphManager.hpp"

#include "../common/types.hpp"      // Ensure this is first for common::IndexPointer definition
#include "ISearcher.hpp"            // Added: Interface for search operations
#include "IStorageManager.hpp"      // Added: Interface for storage operations
#include "distance.hpp"             // For calculate_distance, ConvertToFloat
#include "duckdb/common/limits.hpp" // For NumericLimits
#include "duckdb/common/random_engine.hpp"
#include "duckdb/common/types/vector.hpp"
#include "index_config.hpp"         // For LmDiskannConfig, NodeLayoutOffsets, WordsPerPlane, LmDiskannMetricType
#include "ternary_quantization.hpp" // For WordsPerPlane (if not fully in index_config.hpp)

#include <algorithm> // For std::remove_if, std::find, std::fill, std::min, std::max
#include <cstring>   // For std::memcpy, std::memset
#include <iostream>  // For potential debugging
#include <set>       // For std::set in RobustPrune
#include <stdexcept> // For std::runtime_error
#include <vector>

namespace diskann {
namespace core {

// --- Constructor & Destructor ---

GraphManager::GraphManager(const LmDiskannConfig &config, const NodeLayoutOffsets &node_layout,
                           common::idx_t block_size_bytes, IStorageManager *storage_manager, ISearcher *searcher)
    : config_(config), node_layout_(node_layout), block_size_bytes_(block_size_bytes),
      storage_manager_(storage_manager), searcher_(searcher), // Initialize new members
      graph_entry_point_ptr_(),                               // Default construct
      graph_entry_point_rowid_(common::NumericLimits<common::row_t>::Maximum()), init_vector_search_L_(-1),
      current_max_points_(-1), current_max_graph_points_(-1), current_graph_build_epochs_(-1),
      search_list_size_alpha_(-1.0F) {
	graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID; // Explicitly set members
	graph_entry_point_ptr_.offset = 0;

	if (!storage_manager_) {
		throw std::runtime_error("GraphManager: StorageManager pointer cannot be null.");
	}
	if (!searcher_) {
		throw std::runtime_error("GraphManager: Searcher pointer cannot be null.");
	}

	// Initialize any other members based on config if needed
	// For example, if using a specific distance function based on metric
	if (config_.metric_type == LmDiskannMetricType::L2) {
		// distance_func_ = &L2DistanceFloat; // Example
	} else if (config_.metric_type == LmDiskannMetricType::IP) {
		// distance_func_ = &InnerProductDistanceFloat; // Example
	} else if (config_.metric_type == LmDiskannMetricType::COSINE) {
		// distance_func_ = &CosineDistanceFloat; // Example
	}
	// init_vector_search_L_ = config_param.L_build; // Or some other logic
	// current_max_points_ = 0; // Or from loaded index
	// ...
}

// Destructor is defaulted in the header, no definition needed here.

// --- IGraphManager Interface Implementation ---

void GraphManager::InitializeEmptyGraph(const LmDiskannConfig &config_param, common::IndexPointer &entry_point_ptr_out,
                                        common::row_t &entry_point_rowid_out) {
	[[maybe_unused]] const auto &unused_config = config_param; // Mark as unused
	rowid_to_node_ptr_map_.clear();
	graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
	graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();

	entry_point_ptr_out = graph_entry_point_ptr_;
	entry_point_rowid_out = graph_entry_point_rowid_;
	// Comment: Coordinator is responsible for ensuring StorageManager also
	// initializes its state.
}

bool GraphManager::AddNode(common::row_t row_id, const float *vector_data, common::idx_t dimensions,
                           common::IndexPointer &node_ptr_out) {
	if (dimensions != config_.dimensions) {
		throw common::NotImplementedException("AddNode: Provided dimensions mismatch with index configuration.");
	}

	common::data_ptr_t new_node_block_data = nullptr;

	// Placeholder: Actual implementation requires StorageManager integration.
	// Steps involved:
	// 1. Call StorageManager to allocate a new node block, get IndexPointer
	// (node_ptr_out).
	//    - If allocation fails (e.g., disk full), return false.
	// 2. Call StorageManager to get a mutable data_ptr_t to the allocated block.
	// 3. Use NodeAccessors::InitializeNodeBlock to prepare the block (zero, set
	// neighbor count to 0).
	// 4. Get mutable raw vector pointer: auto* raw_vec_ptr =
	// NodeAccessors::GetRawNodeVectorMutable(...);
	// 5. Convert and copy vector_data to raw_vec_ptr:
	//    - If config_.node_vector_type == LmDiskannVectorType::FLOAT32,
	//    std::memcpy.
	//    - Else (e.g., INT8), perform quantization and copy.
	// 6. Update internal map: rowid_to_node_ptr_map_[row_id] = node_ptr_out;
	// 7. If GetNodeCount() == 1 (this is the first node), call
	// SetEntryPoint(node_ptr_out, row_id).
	// 8. Perform graph connection (the complex part):
	//    - Find candidate neighbors (e.g., using Searcher::SearchForNeighbors or
	//    similar initial search).
	//    - Call this->RobustPrune(node_ptr_out, vector_data, candidates,
	//    config_);
	//    - For each final neighbor from RobustPrune, update their neighbor lists
	//    to include this new node,
	//      and update this new node's neighbor list. This involves:
	//      - Getting neighbor block data via StorageManager.
	//      - Using NodeAccessors to modify neighbor lists.
	//      - Telling StorageManager the neighbor block is dirty.
	// 9. Tell StorageManager the new node block (node_ptr_out) is dirty.

	bool allocation_success = storage_manager_->AllocateNodeBlock(row_id, node_ptr_out, new_node_block_data);

	if (!allocation_success || !new_node_block_data) {
		// Failed to allocate block or get a valid data pointer
		return false;
	}

	NodeAccessors::InitializeNodeBlock(new_node_block_data, block_size_bytes_, node_layout_);

	// Copy vector data
	unsigned char *raw_vec_ptr = NodeAccessors::GetRawNodeVectorMutable(new_node_block_data, node_layout_);
	if (!raw_vec_ptr) {
		// This should not happen if InitializeNodeBlock and layout are correct
		// Consider how to handle this error; perhaps deallocate the block?
		return false;
	}
	if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
		std::memcpy(raw_vec_ptr, vector_data, dimensions * sizeof(float));
	} else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
		// TODO: Implement quantization from float to int8
		// For now, throw or return false, as this is critical
		// diskann::core::quantize_vector(vector_data,
		// reinterpret_cast<int8_t*>(raw_vec_ptr), dimensions, ...);
		throw common::NotImplementedException("AddNode: INT8 quantization not implemented.");
		// return false;
	} else {
		throw common::NotImplementedException("AddNode: Unsupported vector type for storage.");
		// return false;
	}

	rowid_to_node_ptr_map_[row_id] = node_ptr_out;

	// If this is the first node, set it as the entry point
	if (rowid_to_node_ptr_map_.size() == 1) {
		SetEntryPoint(node_ptr_out, row_id);
	}

	// Search for initial candidates for RobustPrune
	// The number of candidates to fetch (e.g., L_search parameter, often related
	// to R) Using config_.r * 2 as a placeholder for L_search or similar search
	// list size parameter.
	common::idx_t num_initial_candidates = config_.r * 2; // Example search parameter
	std::vector<common::row_t> initial_candidate_row_ids;

	// The searcher might need the current entry point to start its search
	// Or it might have its own strategy (e.g., random nodes if graph is too
	// sparse)
	if (graph_entry_point_ptr_.block_id != common::INVALID_BLOCK_ID) {
		searcher_->SearchForInitialCandidates(vector_data, dimensions, config_, this, num_initial_candidates,
		                                      graph_entry_point_ptr_, initial_candidate_row_ids);
	} else {
		// Handle case with no entry point (e.g. very first nodes being added, or
		// searcher handles it)
		// For now, if no entry point, we can't find neighbors this way. RobustPrune
		// will handle empty list.
		// Alternatively, the searcher could be called with an invalid entry point
		// if it can handle that:
		searcher_->SearchForInitialCandidates(vector_data, dimensions, config_, this, num_initial_candidates,
		                                      graph_entry_point_ptr_, // Pass invalid ptr
		                                      initial_candidate_row_ids);
	}

	// Prune and connect the new node
	RobustPrune(node_ptr_out, vector_data, initial_candidate_row_ids, config_);

	// After RobustPrune, initial_candidate_row_ids contains the final R
	// neighbors. Update the new node's neighbor list.
	NodeAccessors::SetNeighborCount(new_node_block_data, node_layout_,
	                                static_cast<uint16_t>(initial_candidate_row_ids.size()));
	common::row_t *new_node_neighbors_ptr = NodeAccessors::GetNeighborIDsPtrMutable(new_node_block_data, node_layout_);
	if (!new_node_neighbors_ptr)
		return false; // Should not happen

	for (size_t i = 0; i < initial_candidate_row_ids.size(); ++i) {
		new_node_neighbors_ptr[i] = initial_candidate_row_ids[i];
	}
	// Initialize remaining neighbor slots if any (RobustPrune should return
	// exactly R or fewer)
	for (size_t i = initial_candidate_row_ids.size(); i < config_.r; ++i) {
		new_node_neighbors_ptr[i] = common::NumericLimits<common::row_t>::Maximum(); // Mark as
		                                                                             // invalid/empty
	}

	// TODO: Bidirectional linking: For each final neighbor in
	// initial_candidate_row_ids, update *their* neighbor lists to include
	// 'row_id' (the new node). This also involves RobustPrune on those neighbors
	// with 'row_id' as a candidate. This is a complex part of the Vamana
	// algorithm (Interconnect step). For now, this stub focuses on one-way
	// connection from new_node to its neighbors. Example for one neighbor: if
	// (!initial_candidate_row_ids.empty()) {
	//   common::row_t first_neighbor_row_id = initial_candidate_row_ids[0];
	//   common::IndexPointer neighbor_ptr;
	//   if (TryGetNodePointer(first_neighbor_row_id, neighbor_ptr)) {
	//     common::data_ptr_t neighbor_block_data =
	//     storage_manager_->GetMutableNodeBlockData(neighbor_ptr); if
	//     (neighbor_block_data) {
	//        // ... logic to add row_id to this neighbor's list, potentially
	//        calling RobustPrune for it ...
	//        // NodeAccessors::... to update neighbor_block_data
	//        storage_manager_->MarkBlockDirty(neighbor_ptr);
	//     }
	//   }
	// }

	storage_manager_->MarkBlockDirty(node_ptr_out);
	return true;
}

// Static helper function to get node data - REMOVED, use StorageManager instead
// static common::const_data_ptr_t GetNodeData_Placeholder(...)

bool GraphManager::GetNodeVector(common::IndexPointer node_ptr, float *vector_out, common::idx_t dimensions) const {
	if (dimensions != config_.dimensions) {
		return false;
	}
	if (node_ptr.block_id == common::INVALID_BLOCK_ID) {
		return false;
	}

	common::const_data_ptr_t node_block = storage_manager_->GetNodeBlockData(node_ptr);
	if (!node_block) {
		// StorageManager could not provide the block data (e.g., IO error, invalid
		// pointer)
		return false;
	}

	const unsigned char *raw_vector = NodeAccessors::GetRawNodeVector(node_block, node_layout_);
	if (!raw_vector) {
		// This implies an issue with layout or the block itself.
		return false;
	}

	if (config_.node_vector_type == LmDiskannVectorType::FLOAT32) {
		std::memcpy(vector_out, raw_vector, dimensions * sizeof(float));
	} else if (config_.node_vector_type == LmDiskannVectorType::INT8) {
		// TODO: Implement dequantization from int8 to float.
		// diskann::core::dequantize_vector(reinterpret_cast<const
		// int8_t*>(raw_vector), vector_out, dimensions, ...);
		throw common::NotImplementedException("GetNodeVector: Dequantization from INT8 not implemented.");
		// return false;
	} else {
		throw common::NotImplementedException("GetNodeVector: Unsupported vector type for retrieval.");
		// return false;
	}
	// No need to delete node_block, StorageManager handles its lifecycle.
	return true;
}

bool GraphManager::GetNeighbors(common::IndexPointer node_ptr, std::vector<common::row_t> &neighbor_row_ids_out) const {
	neighbor_row_ids_out.clear();
	if (node_ptr.block_id == common::INVALID_BLOCK_ID) {
		return false;
	}

	common::const_data_ptr_t node_block = storage_manager_->GetNodeBlockData(node_ptr);
	if (!node_block) {
		return false;
	}

	uint16_t count = NodeAccessors::GetNeighborCount(node_block, node_layout_);
	const common::row_t *ids_ptr = NodeAccessors::GetNeighborIDsPtr(node_block, node_layout_);

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
	// No need to delete node_block, StorageManager handles its lifecycle.
	return true;
}

void GraphManager::RobustPrune(common::IndexPointer node_to_connect_ptr, const float *node_to_connect_vector_data,
                               std::vector<common::row_t> &candidate_row_ids, // Input: initial candidates, Output:
                                                                              // final selected neighbors
                               const LmDiskannConfig &config_param) {         // Use config_param for R, alpha

	if (candidate_row_ids.empty()) {
		return; // No candidates to prune or connect to.
	}

	// Retrieve the current neighbors of node_to_connect_ptr if it's an existing
	// node being updated.
	// This RobustPrune is designed for connecting a NEW node or for updating an
	// existing node's connections where the *initial* candidate_row_ids list
	// is the complete set to prune from.
	// If this node_to_connect_ptr already has neighbors, and we want to consider
	// them
	// in the pruning process, they should have been included in candidate_row_ids
	// by the caller.

	std::vector<std::pair<float, common::row_t>> current_candidates_with_dist;
	std::vector<float> temp_candidate_vector(config_param.dimensions);

	for (common::row_t candidate_rowid : candidate_row_ids) {
		if (candidate_rowid == NodeAccessors::GetRowId(node_to_connect_ptr, node_layout_)) {
			continue; // Don't connect to self
		}

		common::IndexPointer candidate_ptr = GetNodePtrByRowId(candidate_rowid);
		if (candidate_ptr == common::INVALID_INDEX_POINTER) {
			// std::cerr << "Warning: RobustPrune found invalid candidate pointer for
			// rowid " << candidate_rowid << std::endl;
			continue;
		}
		common::const_data_ptr_t candidate_raw_vector = NodeAccessors::GetRawNodeVector(candidate_ptr, node_layout_);

		if (!ConvertRawVectorToFloat(candidate_raw_vector, temp_candidate_vector.data(), config_param.dimensions)) {
			// std::cerr << "Warning: RobustPrune failed to convert vector for rowid "
			// << candidate_rowid << std::endl;
			continue;
		}

		float dist =
		    CalculateDistanceInternal(node_to_connect_vector_data, temp_candidate_vector.data(), config_param.dimensions);
		current_candidates_with_dist.emplace_back(dist, candidate_rowid);
	}

	// Sort candidates by distance
	std::sort(current_candidates_with_dist.begin(), current_candidates_with_dist.end());

	// Remove duplicates by row_id, keeping the one with the smallest distance
	// (already handled by sort)
	current_candidates_with_dist.erase(std::unique(current_candidates_with_dist.begin(),
	                                               current_candidates_with_dist.end(),
	                                               [](const auto &a, const auto &b) { return a.second == b.second; }),
	                                   current_candidates_with_dist.end());

	std::vector<common::row_t> new_neighbor_row_ids;
	std::vector<common::const_data_ptr_t> new_neighbor_raw_vectors; // To store raw vector pointers for alpha
	                                                                // pruning
	std::vector<float> new_neighbor_float_vectors_flat;             // Store all float vectors contiguously
	new_neighbor_float_vectors_flat.reserve(config_param.r * config_param.dimensions);

	// Alpha pruning logic
	for (const auto &candidate_pair : current_candidates_with_dist) {
		if (new_neighbor_row_ids.size() >= config_param.r) {
			break; // Reached max neighbors
		}

		common::row_t p_rowid = candidate_pair.second;
		bool p_is_better_neighbor = true;

		common::IndexPointer p_ptr = GetNodePtrByRowId(p_rowid);
		// We assume p_ptr is valid as it was processed above
		common::const_data_ptr_t p_raw_vector = NodeAccessors::GetRawNodeVector(p_ptr, node_layout_);
		// Convert p_raw_vector to float for distance calculations with other new
		// neighbors
		// This temporary vector is for p_vector itself during this iteration
		std::vector<float> p_float_vector(config_param.dimensions);
		if (!ConvertRawVectorToFloat(p_raw_vector, p_float_vector.data(), config_param.dimensions)) {
			// std::cerr << "Warning: RobustPrune failed to convert p_vector for rowid
			// " << p_rowid << std::endl;
			continue; // Skip this candidate if conversion fails
		}

		for (size_t r_idx = 0; r_idx < new_neighbor_row_ids.size(); ++r_idx) {
			// common::row_t r_rowid = new_neighbor_row_ids[r_idx];
			// common::const_data_ptr_t r_raw_vector =
			// new_neighbor_raw_vectors[r_idx]; Get r's float vector from the flat
			// storage
			const float *r_float_vector_ptr = new_neighbor_float_vectors_flat.data() + (r_idx * config_param.dimensions);

			// Original Vamana condition: alpha * dist(p, r) < dist(node_to_connect,
			// r) dist(node_to_connect, r) is essentially the distance that got r into
			// new_neighbor_row_ids.
			// We need dist(p, r) and dist(node_to_connect, p) (which is
			// candidate_pair.first)

			float dist_p_r = CalculateDistanceInternal(p_float_vector.data(), r_float_vector_ptr, config_param.dimensions);
			float dist_node_r = 0; // This needs to be the distance from node_to_connect to r

			// Find dist_node_r: Iterate through current_candidates_with_dist to find
			// r_rowid and its distance
			// This is inefficient. It's better to store distances for selected
			// neighbors.
			// For now, let's re-calculate for simplicity, or find a way to access it.
			// The current_candidates_with_dist are sorted by distance TO
			// node_to_connect.
			// Let's find r in current_candidates_with_dist
			auto it_r = std::find_if(current_candidates_with_dist.begin(), current_candidates_with_dist.end(),
			                         [&](const auto &pair) { return pair.second == new_neighbor_row_ids[r_idx]; });
			if (it_r != current_candidates_with_dist.end()) {
				dist_node_r = it_r->first;
			} else {
				// This shouldn't happen if r_rowid came from
				// current_candidates_with_dist Or, if r was an original neighbor not in
				// current_candidates_with_dist initially, we need its distance to
				// node_to_connect_vector_data This logic path needs to be very clear
				// about origin of r. For now, assuming all selected neighbors `r`
				// originated from `current_candidates_with_dist`. std::cerr << "Error:
				// Could not find original distance for selected neighbor r." <<
				// std::endl;
				continue;
			}

			if (config_param.alpha * dist_p_r < dist_node_r) {
				p_is_better_neighbor = false;
				break;
			}
			// Symmetrically, from p's perspective to keep r:
			// alpha * dist(p,r) < dist(node_to_connect, p)
			// dist(node_to_connect, p) is candidate_pair.first
			if (config_param.alpha * dist_p_r < candidate_pair.first) {
				// This means r would be pruned by p if p was already in the set.
				// This is the condition to prune r if p is chosen.
				// The original loop structure is:
				// for each p in sorted_candidates:
				//   if p is not occluded by any r in current_selection:
				//     add p to current_selection
				//     if current_selection.size > R: prune from current_selection (the
				//     worst one)
				// This seems to be what is intended.
				// The check `alpha * dist(p, r) < dist_node_r` means:
				// if path node->p->r is shorter than node->r, then p occludes r from
				// node's perspective. This means p is a candidate to *replace* r or
				// make r redundant. No, this is to decide if P should be added. If
				// `alpha * dist(p,r) < dist_node_r`, it means `p` is "behind" `r`
				// relative to `node_to_connect` or too close to `r` along a path that
				// is not much better than going directly to `r`. So, `p` would be
				// pruned.
			}
		}

		if (p_is_better_neighbor) {
			new_neighbor_row_ids.push_back(p_rowid);
			new_neighbor_raw_vectors.push_back(p_raw_vector); // Store for reference if needed, though float used
			                                                  // next
			// Store p's float vector
			new_neighbor_float_vectors_flat.insert(new_neighbor_float_vectors_flat.end(), p_float_vector.begin(),
			                                       p_float_vector.end());

			if (new_neighbor_row_ids.size() >= config_param.R_neighbor_connections) {
				// If we add p and exceed R, we might need to prune from
				// new_neighbor_row_ids. The original paper's algorithm for robust
				// prune:
				// 1. V <- S U {p} // S is current selection, p is candidate
				// 2. if |V| > R_max: V <- V \ argmax_{q in V} dist(orig_node, q) //
				// prune farthest
				// 3. For q in V: if alpha * dist(p,q) < dist(orig_node, q) AND p != q:
				//       V <- V \ {q} // p occludes q
				// 4. For q in V: if alpha * dist(p,q) < dist(orig_node, p) AND p != q:
				//       V <- V \ {p}; break from inner loop // q occludes p

				// The current loop iterates through sorted candidates (p).
				// If p is added, then we check if p occludes any existing r in
				// new_neighbor_row_ids. And if any existing r occludes p.

				// Let's simplify: add p if it's not occluded by current selection.
				// Then, if size > R, remove the one farthest from
				// node_to_connect_vector_data. The current loop is: for each candidate
				// `p`, check if any `r` in `new_neighbor_row_ids` occludes `p`. If not,
				// add `p`. If size overflows `R`, break. This means we take the first R
				// non-occluded items.

				// This is effectively: keep adding to new_neighbor_row_ids if
				// p_is_better_neighbor and stop once R_neighbor_connections is reached.
				// The initial sort of current_candidates_with_dist ensures we consider
				// closer candidates first.
			}
		}
	}
	// The final list of neighbors is in new_neighbor_row_ids
	candidate_row_ids = new_neighbor_row_ids;
}

void GraphManager::SetEntryPoint(common::IndexPointer node_ptr, common::row_t row_id) {
	// Basic validation: if one is valid, the other should be too (or both invalid
	// for reset)
	if ((node_ptr.block_id == common::INVALID_BLOCK_ID) != (row_id == common::NumericLimits<common::row_t>::Maximum())) {
		// Log a warning about inconsistent entry point state if desired.
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
		graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
		graph_entry_point_rowid_ = common::NumericLimits<common::row_t>::Maximum();
	}
	// Comment: Actual removal from neighbor lists of other nodes is complex and
	// usually part of a separate process (e.g., vacuuming, or search-time
	// filtering of deleted nodes). This function primarily ensures a deleted node
	// isn't the active entry point.
}

void GraphManager::FreeNode(common::row_t row_id) {
	// This is a logical removal from GraphManager's tracking.
	// Coordinator must inform StorageManager to actually reclaim/mark the block
	// as free.
	if (rowid_to_node_ptr_map_.erase(row_id) > 0) {
		// If the freed node was the entry point, reset entry point to invalid.
		if (row_id == graph_entry_point_rowid_) {
			common::IndexPointer invalid_ptr;
			invalid_ptr.block_id = common::INVALID_BLOCK_ID;
			invalid_ptr.offset = 0;
			SetEntryPoint(invalid_ptr, common::NumericLimits<common::row_t>::Maximum());
		}
	} // else: node was not found in map, nothing to erase.
}

common::idx_t GraphManager::GetNodeCount() const {
	return rowid_to_node_ptr_map_.size();
}

common::idx_t GraphManager::GetInMemorySize() const {
	common::idx_t size = sizeof(*this);
	size += rowid_to_node_ptr_map_.size() * (sizeof(common::row_t) + sizeof(common::IndexPointer));
	// Add size of storage_manager_ if it's significant and measurable here
	// The actual node data is managed by storage_manager_
	return size;
}

void GraphManager::Reset() {
	rowid_to_node_ptr_map_.clear();
	graph_entry_point_ptr_.block_id = common::INVALID_BLOCK_ID;
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
	node_ptr_out.block_id = common::INVALID_BLOCK_ID; // Ensure it's invalid if not found
	node_ptr_out.offset = 0;                          // Also zero out offset for an invalid pointer
	return false;
}

common::row_t GraphManager::GetRandomNodeID(common::RandomEngine &engine) {
	if (rowid_to_node_ptr_map_.empty()) {
		return common::NumericLimits<common::row_t>::Maximum();
	}
	auto map_size_val = rowid_to_node_ptr_map_.size();
	if (map_size_val == 0) { // Should be caught by empty() check, but good for safety
		return common::NumericLimits<common::row_t>::Maximum();
	}
	common::idx_t random_idx = static_cast<common::idx_t>(engine.NextRandom() * map_size_val);
	// Ensure random_idx is within bounds [0, map_size_val - 1]
	if (random_idx >= map_size_val) {
		random_idx = map_size_val - 1;
	}
	auto iter = rowid_to_node_ptr_map_.begin();
	std::advance(iter, random_idx);
	return iter->first;
}

// --- Private Helper Methods ---

// Internal distance calculation helpers, using config_ members consistently.
float GraphManager::CalculateDistanceInternal(common::IndexPointer node1_ptr, common::IndexPointer node2_ptr) const {
	std::vector<float> vec1(config_.dimensions);
	std::vector<float> vec2(config_.dimensions);

	// GetNodeVector is const, so no const_cast needed anymore if it's correctly
	// implemented.
	if (!GetNodeVector(node1_ptr, vec1.data(), config_.dimensions) ||
	    !GetNodeVector(node2_ptr, vec2.data(), config_.dimensions)) {
		// Error handling: could not retrieve one or both vectors.
		// Depending on policy, could return infinity, throw, or a special error
		// value.
		throw std::runtime_error("CalculateDistanceInternal: Failed to get node vectors.");
	}
	return CalculateDistanceInternal(vec1.data(), vec2.data(), config_.dimensions);
}

float GraphManager::CalculateDistanceInternal(const float *vec1, const float *vec2, common::idx_t dimensions) const {
	if (dimensions == 0)
		return 0.0f; // Or handle error
	switch (config_.metric) {
	case MetricType::L2:
		return diskann::core::DistanceL2(vec1, vec2, dimensions);
	case MetricType::INNER_PRODUCT:
		// Note: DiskANN typically maximizes inner product. If using for distance,
		// often 1 - IP or -IP. Assuming direct IP for now as "distance".
		return diskann::core::DistanceInnerProduct(vec1, vec2, dimensions);
	case MetricType::COSINE:
		return diskann::core::DistanceCosine(vec1, vec2, dimensions);
	default:
		// Should not happen if config is validated
		return common::NumericLimits<float>::Maximum();
	}
}

float GraphManager::CalculateDistanceInternal(common::IndexPointer node_ptr, const float *query_vec,
                                              common::idx_t dimensions) const {
	std::vector<float> node_vec(dimensions);
	common::const_data_ptr_t raw_node_vec = NodeAccessors::GetRawNodeVector(node_ptr, node_layout_);
	if (!ConvertRawVectorToFloat(raw_node_vec, node_vec.data(), dimensions)) {
		// std::cerr << "Error: Failed to convert node vector for distance
		// calculation." << std::endl;
		return common::NumericLimits<float>::Maximum();
	}
	return CalculateDistanceInternal(node_vec.data(), query_vec, dimensions);
}

bool GraphManager::ConvertRawVectorToFloat(common::const_data_ptr_t raw_vector_data, float *float_vector_out,
                                           common::idx_t dimensions) const {
	if (!raw_vector_data || !float_vector_out) {
		return false;
	}

	switch (config_.quantization_type) {
	case QuantizationType::NONE: // Assuming float if no quantization
	case QuantizationType::FP32:
		memcpy(float_vector_out, raw_vector_data, dimensions * sizeof(float));
		return true;
	case QuantizationType::FP16: {
		const uint16_t *fp16_data = reinterpret_cast<const uint16_t *>(raw_vector_data);
		for (common::idx_t i = 0; i < dimensions; ++i) {
			float_vector_out[i] = duckdb_fp16_ieee_to_float(fp16_data[i]);
		}
		return true;
	}
	case QuantizationType::INT8: {
		const int8_t *int8_data = reinterpret_cast<const int8_t *>(raw_vector_data);
		for (common::idx_t i = 0; i < dimensions; ++i) {
			float_vector_out[i] = static_cast<float>(int8_data[i]);
		}
		// This is a placeholder. Proper INT8 dequantization needs scale/zero-point.
		// Consider if this type is actually used for primary node vectors or just edges.
		return true;
	}
	default:
		// Unsupported quantization type
		return false;
	}
}

bool GraphManager::CompressFloatVectorForEdge(const float *float_vector_in, common::data_ptr_t compressed_edge_data_out,
                                              common::idx_t dimensions) const {
	if (!config_.use_ternary_quantization) {
		// If not using ternary quantization for edges, this might be a no-op,
		// or copy float data if edge storage supports it (node_layout_ should
		// reflect this) Currently, node_layout_ implies ternary if
		// use_ternary_quantization is true.
		// This function specifically handles ternary quantization.
		return false; // Or true if no compression is a valid state and handled by
		              // caller
	}
	if (!float_vector_in || !compressed_edge_data_out) {
		return false;
	}
	if (dimensions != config_.dimensions) {
		return false; // Mismatch in dimensions
	}

	// Assuming ternary_quantization.hpp provides the necessary function
	// The original code used:
	// diskann::ternary_quantization::vectors_to_uint8_codebook(
	//    vector_data_const_ptr, 1, config_.dimensions,
	//    config_.ternary_vector_alpha, planes_raw_ptr,
	//    ternary_codes_for_node_ptr);

	// We need the planes_raw_ptr which are part of LmDiskannConfig
	// and should be loaded/initialized.
	if (!config_.ternary_planes_ptr) {
		// std::cerr << "Error: Ternary planes not available for compression." <<
		// std::endl;
		return false;
	}

	diskann::ternary_quantization::vectors_to_uint8_codebook(
	    float_vector_in, 1, config_.dimensions, config_.ternary_vector_alpha,
	    reinterpret_cast<const float *>(config_.ternary_planes_ptr), compressed_edge_data_out);
	return true;
}

} // namespace core
} // namespace diskann
