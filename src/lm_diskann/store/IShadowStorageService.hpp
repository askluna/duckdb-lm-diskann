#pragma once

#include "../common/types.hpp" // For common::row_t, common::IndexPointer

namespace diskann {
namespace store {

/**
 * @brief Interface for abstracting transactional operations related to a shadow
 * store.
 *
 * This service allows the core DiskANN logic to request logging of
 * modifications (inserts, deletes, updates) in a way that can be
 * transactionally managed by the underlying DuckDB system (or a mock for
 * testing).
 */
class IShadowStorageService {
public:
  virtual ~IShadowStorageService() = default;

  /**
   * @brief Logs an insertion operation.
   *
   * @param row_id The RowID of the inserted vector.
   * @param node_ptr Pointer to the newly allocated node in the main graph
   * store. This might be used to fetch vector data if the shadow store needs to
   * persist it directly from the main graph block. Alternatively, serialized
   * vector data could be passed.
   */
  virtual void LogInsert(common::row_t row_id,
                         common::IndexPointer node_ptr) = 0;

  /**
   * @brief Logs a deletion operation.
   *
   * @param row_id The RowID of the vector to be deleted.
   */
  virtual void LogDelete(common::row_t row_id) = 0;

  // Add other methods as needed, e.g., for updates, checkpointing, recovery.
  // virtual void LogUpdate(common::row_t old_row_id, common::row_t new_row_id,
  // common::IndexPointer new_node_ptr) = 0; virtual void CommitChanges() = 0;
  // // Or apply changes virtual void RollbackChanges() = 0; virtual void
  // LoadPersistentState() = 0; // To load state from diskann_store.duckdb on
  // startup
};

} // namespace store
} // namespace diskann