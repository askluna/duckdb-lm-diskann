#pragma once

#include "index_config.hpp" // Added include for IndexConfig definition
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
class ClientContext;
} // namespace duckdb

// Forward declare types that might be used in method signatures
// For example, if you have specific structs/classes for queries, results, data
// points
// class QueryType;
// class ResultType;
// class DataType;

class IGraphManager;
class IStorageManager;
class ISearcher;

namespace store {
class IShadowStorageService;
}

namespace diskann {
namespace core {

class Orchestrator {
public:
  // Constructor expecting injected dependencies
  Orchestrator(
      std::unique_ptr<IStorageManager> storage_manager,
      std::unique_ptr<IGraphManager> graph_manager,
      std::unique_ptr<ISearcher> searcher,
      std::unique_ptr<store::IShadowStorageService> shadow_storage_service,
      const IndexConfig &config
      // duckdb::ClientContext& context // If direct context is needed
  );

  ~Orchestrator();

  // Core high-level operations
  // Parameters are illustrative and will need to be refined
  void BuildIndex(const std::string &data_path /*, other params */);

  // std::vector<ResultType> Search(const QueryType& query, int top_k);
  void Search(/* illustrative params: const float* query_vector, size_t query_dim, uint32_t top_k, std::vector<int64_t>& results */);

  void Insert(const float *data_vector, size_t data_dim, int64_t label);

  void Delete(int64_t label);

  void Update(int64_t label, const float *new_data_vector, size_t data_dim);

  void LoadIndex(const std::string &index_path);

  void SaveIndex(const std::string &index_path);

  // Other potential methods
  // void Consolidate();
  // IndexStats GetStats();

private:
  std::unique_ptr<IStorageManager> storage_manager_;
  std::unique_ptr<IGraphManager> graph_manager_;
  std::unique_ptr<ISearcher> searcher_;
  std::unique_ptr<store::IShadowStorageService> shadow_storage_service_;
  IndexConfig config_;
  // duckdb::ClientContext& context_; // If direct context is needed

  // Internal state
  std::string index_path_;
  bool index_loaded_ = false;
  // Potentially other state variables like graph entry point, metadata cache,
  // etc.
};

} // namespace core
} // namespace diskann
