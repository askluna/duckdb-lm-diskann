/**
 * @file index_config.hpp
 * @brief Defines configuration structures, enums, constants, and functions for
 * the LM-DiskANN index.
 */
#pragma once

#include "duckdb.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/enums/vector_type.hpp"
#include "duckdb/common/types.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/index/index_pointer.hpp"

#include <cstdint>
#include <string>

namespace diskann {
namespace core {

// --- Enums for LM-DiskANN Parameters --- //

/**
 * @brief Metric types supported by the index.
 * Corresponds to nDistanceFunc / VECTOR_METRIC_TYPE_PARAM_ID
 */
enum class LmDiskannMetricType : uint8_t {
  UNKNOWN = 0, // Default / Unset state
  L2 = 1,      // Euclidean distance
  COSINE = 2,  // Cosine similarity (often converted to distance: 1 - cosine)
  IP = 3,      // Inner product
  HAMMING = 4  // Hamming distance (Potentially for binary vectors)
};

/**
 * @brief Data types supported for the main node vectors.
 * Corresponds to nNodeVectorType / VECTOR_TYPE_PARAM_ID
 */
enum class LmDiskannVectorType : uint8_t {
  UNKNOWN = 0, // Default / Unset state
  FLOAT32 = 1, // 32-bit floating point
  INT8 = 2,    // 8-bit integer (requires quantization parameters potentially)
};

// --- Configuration Constants --- //

/**
 * @brief Option keys used in the CREATE INDEX ... WITH (...) clause.
 */
struct LmDiskannOptionKeys {
  static constexpr const char *METRIC = "METRIC";
  static constexpr const char *R = "R";
  static constexpr const char *L_INSERT = "L_INSERT";
  static constexpr const char *ALPHA = "ALPHA";
  static constexpr const char *L_SEARCH = "L_SEARCH";
};

/**
 * @brief Default values for LM-DiskANN configuration parameters.
 */
struct LmDiskannConfigDefaults {
  static constexpr LmDiskannMetricType METRIC = LmDiskannMetricType::L2;
  static constexpr uint32_t R = 64;
  static constexpr uint32_t L_INSERT = 128;
  static constexpr float ALPHA = 1.2f;
  static constexpr uint32_t L_SEARCH = 100;
};

/**
 * @brief Current on-disk format version for the index metadata.
 */
inline constexpr uint8_t LMDISKANN_CURRENT_FORMAT_VERSION = 3;

// --- Configuration Struct --- //
/**
 * @brief Holds the core configuration parameters for the LM-DiskANN index.
 */
struct LmDiskannConfig {
  LmDiskannMetricType metric_type =
      LmDiskannConfigDefaults::METRIC; // Distance metric used for comparisons.
  uint32_t r = LmDiskannConfigDefaults::R; // Max neighbors (degree) per node in
                                           // the graph (R).
  uint32_t l_insert =
      LmDiskannConfigDefaults::L_INSERT; // Size of the candidate list during
                                         // index insertion (L_insert).
  float alpha = LmDiskannConfigDefaults::ALPHA; // Alpha parameter for pruning
                                                // during insertion.
  uint32_t l_search =
      LmDiskannConfigDefaults::L_SEARCH; // Size of the candidate list during
                                         // search (L_search).

  idx_t dimensions =
      0; // Dimensionality of the vectors. Derived from column type.
  LmDiskannVectorType node_vector_type =
      LmDiskannVectorType::UNKNOWN; // Data type of the node vectors. Derived
                                    // from column type.
};

// --- Struct to hold calculated layout offsets --- //
/**
 * @brief Stores the calculated byte offsets for different data sections within
 * a node's disk block.
 * @details Assumes TERNARY compressed neighbors. Crucial for low-level node
 * accessors.
 */
struct NodeLayoutOffsets {
  idx_t neighbor_count_offset =
      0; // Offset of the neighbor count (uint16_t) - Typically 0
  idx_t node_vector_offset =
      0; // Offset of the start of the node's full vector data
  idx_t neighbor_ids_offset =
      0; // Offset of the start of the neighbor row_t array
  idx_t neighbor_pos_planes_offset =
      0; // Offset of the start of the positive ternary planes array
  idx_t neighbor_neg_planes_offset =
      0; // Offset of the start of the negative ternary planes array
  idx_t total_node_size = 0; // Total size *before* final block alignment (used
                             // for allocation/memcpy)
  idx_t ternary_edge_size_bytes =
      0; // Calculated size of one neighbor's compressed ternary representation
};

/**
 * @brief Non-owning view of constant ternary bit planes.
 */
struct TernaryPlanesView {
  ::duckdb::const_data_ptr_t positive_plane =
      nullptr; // Pointer to the positive plane data.
  ::duckdb::const_data_ptr_t negative_plane =
      nullptr;          // Pointer to the negative plane data.
  idx_t dimensions = 0; // Dimensionality of the vector these planes represent.
  idx_t words_per_plane =
      0; // Pre-calculated size (in uint64_t words) based on dimensions.

  /**
   * @brief Basic validity check.
   * @return True if the view points to valid data structures, false otherwise.
   */
  bool IsValid() const {
    return positive_plane != nullptr && negative_plane != nullptr &&
           words_per_plane > 0;
  }
};

/**
 * @brief Non-owning view of mutable ternary bit planes.
 */
struct MutableTernaryPlanesView {
  ::duckdb::data_ptr_t positive_plane =
      nullptr; // Pointer to the mutable positive plane data.
  ::duckdb::data_ptr_t negative_plane =
      nullptr;          // Pointer to the mutable negative plane data.
  idx_t dimensions = 0; // Dimensionality of the vector these planes represent.
  idx_t words_per_plane =
      0; // Pre-calculated size (in uint64_t words) based on dimensions.

  /**
   * @brief Basic validity check.
   * @return True if the view points to valid data structures, false otherwise.
   */
  bool IsValid() const {
    return positive_plane != nullptr && negative_plane != nullptr &&
           words_per_plane > 0;
  }
};

/**
 * @brief Non-owning view of a batch of contiguous ternary bit planes.
 * @details Used to describe the layout of pre-encoded database vectors for
 * search.
 */
struct TernaryPlaneBatchView {
  const uint64_t *positive_planes_start =
      nullptr; // Pointer to the start of ALL positive planes for N vectors.
  const uint64_t *negative_planes_start =
      nullptr; // Pointer to the start of ALL negative planes for N vectors.
  size_t num_vectors = 0; // Number of vectors (N) in the batch.
  size_t words_per_plane =
      0; // Pre-calculated words per single plane (from WordsPerPlane(dims)).

  /**
   * @brief Basic validity check.
   * @return True if the batch view seems valid, false otherwise.
   */
  bool IsValid() const {
    return positive_planes_start != nullptr &&
           negative_planes_start != nullptr && num_vectors > 0 &&
           words_per_plane > 0;
  }
};

// --- Configuration Functions --- //

/**
 * @brief Parses index options from the WITH clause into a config struct.
 * @param options Map of option keys to values from CREATE INDEX.
 * @return LmDiskannConfig struct populated with parsed or default values.
 * @throws Exception if an unknown option is provided.
 */
LmDiskannConfig
ParseOptions(const ::duckdb::case_insensitive_map_t<::duckdb::Value> &options);

/**
 * @brief Validates the combination of parameters within the config struct.
 * @param config The configuration struct to validate.
 * @throws Exception if validation fails (e.g., required parameters unset,
 * incompatible values).
 */
void ValidateParameters(const LmDiskannConfig &config);

/**
 * @brief Gets the size in bytes for a given vector data type.
 * @param type The LmDiskannVectorType enum value.
 * @return Size in bytes.
 * @throws InternalException for unsupported types.
 */
idx_t GetVectorTypeSizeBytes(LmDiskannVectorType type);

/**
 * @brief Gets the size in bytes for *one* compressed ternary plane (pos or neg)
 * for one neighbor.
 * @param dimensions The vector dimensionality.
 * @return Size in bytes.
 * @throws InternalException if dimensions is 0.
 */
idx_t GetTernaryPlaneSizeBytes(idx_t dimensions);

/**
 * @brief Calculates the internal layout offsets within a node block based on
 * config.
 * @details Requires dimensions and node_vector_type to be set in the config.
 * @param config The index configuration.
 * @return NodeLayoutOffsets struct containing the calculated offsets and sizes.
 */
NodeLayoutOffsets CalculateLayoutInternal(const LmDiskannConfig &config);

// --- Utility Functions (Potentially move later) ---

/**
 * @brief Helper function to convert enum LmDiskannMetricType to string
 */
const char *LmDiskannMetricTypeToString(LmDiskannMetricType type);

/**
 * @brief Helper function to convert enum LmDiskannVectorType to string
 */
const char *LmDiskannVectorTypeToString(LmDiskannVectorType type);

// --- Metadata Struct --- //
/**
 * @brief Holds all parameters persisted in the index metadata block.
 */
struct LmDiskannMetadata {
  uint8_t format_version = 0; // Internal format version for compatibility
  LmDiskannMetricType metric_type =
      LmDiskannMetricType::UNKNOWN; // Distance metric used
  LmDiskannVectorType node_vector_type =
      LmDiskannVectorType::UNKNOWN; // Type of vectors stored in nodes
  // Edge type is implicitly Ternary, no need to store explicitly
  idx_t dimensions = 0;       // Vector dimensionality
  uint32_t r = 0;             // Max neighbors per node (graph degree)
  uint32_t l_insert = 0;      // Search list size during insertion
  float alpha = 0.0f;         // Pruning factor during insertion
  uint32_t l_search = 0;      // Search list size during query
  idx_t block_size_bytes = 0; // Size of each node block on disk
  ::duckdb::IndexPointer
      graph_entry_point_ptr; // Pointer to the entry node block
  ::duckdb::IndexPointer
      delete_queue_head_ptr; // Pointer to the head of the delete queue block
  // IndexPointer rowid_map_root_ptr; // TODO: Add when ART is integrated
};

} // namespace core
} // namespace diskann
