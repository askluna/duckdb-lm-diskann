#ifndef UNUM_USEARCH_INDEX_DENSE_HPP
#define UNUM_USEARCH_INDEX_DENSE_HPP

// Include the main stub header which defines the necessary types and classes
#include "index.hpp"

// You might need to define specific aliases or types used only from index_dense.hpp
// if HNSW specifically includes/uses them, but typically the main index.hpp
// should contain the primary `index_dense_gt` template.

namespace unum {
namespace usearch {

// Define the non-templated alias used in hnsw_index.cpp
using index_dense_t = index_dense_gt<>; // Assuming default KeyType (row_t)

} // namespace usearch
} // namespace unum

#endif // UNUM_USEARCH_INDEX_DENSE_HPP 