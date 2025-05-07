export module diskann.core.Orchestrator;

// Forward declarations for now; these will be replaced with 'import' statements
// as the respective modules are created.
// Example: import diskann.IStorageManager;
//          import diskann.store.IShadowStorageService;

namespace diskann {
namespace core {

// Forward declare interfaces and types that Orchestrator will depend on.
// These will eventually be imported from their own modules.
class IStorageManager;
class IGraphManager;
class ISearcher;
class IndexConfig; // Assuming this will be a class or struct in
                   // diskann.IndexConfig

namespace store {
class IShadowStorageService; // This interface will be in
                             // diskann.store.IShadowStorageService
}

/**
 * @brief The Orchestrator class is the central component for core DiskANN
 * logic.
 *
 * It owns and controls the DiskANN graph's state, implements high-level
 * indexing operations, and coordinates tasks with other diskann modules. Its
 * dependencies are injected.
 */
export class Orchestrator {
public:
  /**
   * @brief Constructs the Orchestrator.
   *
   * @param storage_manager A unique pointer to an IStorageManager
   * implementation.
   * @param graph_manager A unique pointer to an IGraphManager implementation.
   * @param searcher A unique pointer to an ISearcher implementation.
   * @param config The index configuration.
   * @param shadow_service A unique pointer to an IShadowStorageService
   * implementation.
   */
  explicit Orchestrator(
      // We'll use placeholder types for now, to be replaced with actual
      // std::unique_ptr<IStorageManager> storage_manager,
      // std::unique_ptr<IGraphManager> graph_manager,
      // std::unique_ptr<ISearcher> searcher,
      // const IndexConfig& config,
      // std::unique_ptr<store::IShadowStorageService> shadow_service
  );

  // High-level operations as per the design document:
  // void BuildIndex(/* parameters */);
  // void Search(/* parameters */);
  // void Insert(/* parameters */);
  // void Update(/* parameters */);
  // void Delete(/* parameters */);

  // Lifecycle management or other necessary public methods

private:
  // Member variables to hold injected dependencies and internal state
  // std::unique_ptr<IStorageManager> storage_manager_;
  // std::unique_ptr<IGraphManager> graph_manager_;
  // std::unique_ptr<ISearcher> searcher_;
  // IndexConfig index_config_; // Or a copy, depending on ownership semantics
  // std::unique_ptr<store::IShadowStorageService> shadow_storage_service_;

  // Internal state related to the graph, configuration, etc.
};

} // namespace core
} // namespace diskann
