#ifndef UNUM_USEARCH_HPP
#define UNUM_USEARCH_HPP

#include <cstddef> // For size_t
#include <cstdint> // For uint8_t, int64_t
#include <vector>  // Included because hnsw_index.hpp uses std::vector<...stats_t>

// Define row_t if it's not standard and expected by HNSW code referencing usearch types
// Assuming row_t might be int64_t based on DuckDB usage
// Forward declare if needed, or ensure definition is available before HNSW includes this.
namespace duckdb {
using row_t = int64_t;
} // namespace duckdb


namespace unum {
namespace usearch {

// --- Forward declarations for types potentially used ---
using byte_t = char;
using row_t = duckdb::row_t; // Use DuckDB's row_t definition

// --- Minimal enums used by HNSW ---
enum struct metric_kind_t : std::uint8_t {
  unknown_k = 0,
  l2sq_k = 'e',
  ip_k = 'i',
  cos_k = 'c',
  divergence_k = 'd',
  hamming_k = 'b',
  jaccard_k = 'j',
  haversine_k = 'h',
  pearson_k = 'p',
  sorensen_k = 's',
  tanimoto_k = 't'
};
enum struct scalar_kind_t : std::uint8_t {
  unknown_k = 0,
  f32_k = 11,
  f16_k = 12,
  i8_k = 23,
  b1x8_k = 1,
  f64_k = 10,
  i16_k = 22,
  i32_k = 21,
  i64_k = 20,
  u8_k = 17,
  u16_k = 16,
  u32_k = 15,
  u64_k = 14
};

// --- Stub Limits struct (used by reserve) ---
struct index_limits_t {
    size_t members = 0;
    size_t threads_add = 0;
    size_t threads_search = 0;

    // Constructor to allow initialization like {count, threads}
    index_limits_t(size_t mem = 0, size_t threads = 1) 
        : members(mem), threads_add(threads), threads_search(threads) {}
};

// --- Stub configuration struct ---
struct index_dense_config_t {
  bool enable_key_lookups = false;
  size_t expansion_add = 0;
  size_t expansion_search = 0;
  size_t connectivity = 0;
  size_t connectivity_base = 0;
};

// --- Stub metric type ---
struct metric_punned_t {
  metric_punned_t(size_t dimensions = 0, metric_kind_t kind = metric_kind_t::unknown_k, scalar_kind_t scalar = scalar_kind_t::unknown_k)
    : dimensions_(dimensions), kind_(kind), scalar_(scalar) {}

  metric_kind_t metric_kind() const noexcept { return kind_; }
  size_t dimensions() const noexcept { return dimensions_; }
  scalar_kind_t scalar_kind() const noexcept { return scalar_; }

private:
    size_t dimensions_ = 0;
    metric_kind_t kind_ = metric_kind_t::unknown_k;
    scalar_kind_t scalar_ = scalar_kind_t::unknown_k;
};

// --- Stub error handling ---
struct error_t {
  error_t(char const* = nullptr) noexcept {}
  explicit operator bool() const noexcept { return true; }
  void raise() const noexcept {}
  const char *what() const noexcept { return ""; }
};

// --- Stub Add Result ---
struct add_result_t {
    error_t error{};
    explicit operator bool() const noexcept { return !error; }
};

// --- Stub search result types ---
struct match_t {
  row_t key = -1;
  float distance = 0.0f;
  match_t() = default;
  match_t(row_t k, float d) : key(k), distance(d) {}
  match_t(const match_t&) = default;
  match_t(match_t&&) = default;
  match_t& operator=(const match_t&) = default;
  match_t& operator=(match_t&&) = default;
};

struct search_result_t {
  size_t count = 0;
  error_t error;
  search_result_t() = default;
  search_result_t(const search_result_t&) = default;
  search_result_t(search_result_t&&) = default;
  search_result_t& operator=(const search_result_t&) = default;
  search_result_t& operator=(search_result_t&&) = default;

  size_t size() const noexcept { return count; }
  bool empty() const noexcept { return count == 0; }
  match_t operator[](size_t /*index*/) const noexcept { return {}; }
  match_t at(size_t /*index*/) const noexcept { return {}; }
  size_t dump_to(row_t* keys, float* /*distances*/ = nullptr) const noexcept {
      for(size_t i = 0; i < count; ++i) {
          if (keys) keys[i] = -1;
      }
      return count;
  }
};


// --- Stub index class template ---
template <typename KeyType = row_t>
struct index_dense_gt {

  // --- Nested types needed by HNSW ---
  struct stats_t {
      std::size_t nodes{};
      std::size_t edges{};
      std::size_t max_edges{};
      std::size_t allocated_bytes{};
  };
  using search_result_t = unum::usearch::search_result_t;
  using add_result_t = unum::usearch::add_result_t;

private:
  // Declare private members used by methods
  index_dense_config_t config_ = {};
  metric_punned_t metric_ = {};
  size_t size_ = 0;
  size_t capacity_ = 0;
  size_t connectivity_ = 0;
  size_t max_level_ = 0;
  size_t dimensions_ = 0;

public:
  // --- Static factory method used by HNSW ---
  static index_dense_gt<KeyType> make(metric_punned_t const& metric, index_dense_config_t const& config) {
    index_dense_gt<KeyType> idx;
    idx.config_ = config;           // Assign to member
    idx.metric_ = metric;           // Assign to member
    idx.connectivity_ = config.connectivity; // Assign to member
    idx.dimensions_ = metric.dimensions(); // Assign to member
    return idx;
  }

  // --- Static member for thread index ---
  static constexpr size_t any_thread() { return 0; }

  // --- Methods called by HNSW ---
  bool reserve(index_limits_t limits) { capacity_ = limits.members; return true; } // Assign to member

  template <typename VectorType>
  add_result_t add(KeyType /*key*/, VectorType const& /*vector*/, size_t /*thread_idx*/ = 0) {
      size_++; // Increment member
      return {};
  }

  template <typename VectorType, typename PredicateType = bool(*)(KeyType)>
  unum::usearch::search_result_t search(
      VectorType const& /*query_vector*/,
      size_t /*k*/,
      size_t /*ef_search*/ = 0,
      PredicateType /*predicate*/ = [](KeyType){ return true; }
  ) const {
      unum::usearch::search_result_t res;
      return res;
  }

  template <typename StreamFn>
  void save_to_stream(StreamFn&& /*saver*/) const { /* No-op */ }

  template <typename StreamFn>
  void load_from_stream(StreamFn&& /*loader*/) { /* No-op */ }

  void reset() noexcept { size_ = 0; /* clear other state */ } // Assign to member
  add_result_t compact() noexcept { return {}; }
  add_result_t remove(KeyType /*key*/) noexcept { if (size_ > 0) size_--; return {}; }

  template <typename QType>
  search_result_t ef_search(QType const& query_vector, size_t k, size_t ef_search_val) const noexcept {
      return search(query_vector, k, ef_search_val);
  }

  // --- Accessors called by HNSW ---
  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return capacity_; }
  size_t connectivity() const noexcept { return connectivity_; }
  size_t max_level() const noexcept { return max_level_; }
  size_t dimensions() const noexcept { return dimensions_; }
  size_t memory_usage() const noexcept { return size_ * 100; /* Dummy value using member */ }
  size_t expansion_search() const noexcept { return config_.expansion_search; }
  const metric_punned_t& metric() const noexcept { return metric_; }
  const index_dense_config_t& config() const noexcept { return config_; }

  // Stub for stats() method called by HNSWIndex::VerifyAndToString
  stats_t stats() const noexcept {
      stats_t s;
      s.nodes = size_;
      s.edges = size_ * connectivity_;
      s.max_edges = size_ * connectivity_ * 2;
      s.allocated_bytes = memory_usage();
      return s;
  }
  stats_t stats(size_t /*level*/) const noexcept { return stats(); }
  stats_t stats(stats_t* /*stats_per_level*/, size_t /*max_level*/) const noexcept { return stats(); }

};

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_HPP