#pragma once

#include "duckdb/storage/data_pointer.hpp" // For data_ptr_t, const_data_ptr_t
#include "duckdb/common/types.hpp" // For idx_t

namespace duckdb {

//! Non-owning view of constant ternary bit planes.
struct TernaryPlanesView {
    const_data_ptr_t positive_plane = nullptr;
    const_data_ptr_t negative_plane = nullptr;
    idx_t words_per_plane = 0; // Pre-calculated size based on dimensions

    // Basic validity check
    bool IsValid() const {
        return positive_plane != nullptr && negative_plane != nullptr && words_per_plane > 0;
    }
};

//! Non-owning view of mutable ternary bit planes.
struct MutableTernaryPlanesView {
    data_ptr_t positive_plane = nullptr;
    data_ptr_t negative_plane = nullptr;
    idx_t words_per_plane = 0; // Pre-calculated size based on dimensions

    // Basic validity check
    bool IsValid() const {
        return positive_plane != nullptr && negative_plane != nullptr && words_per_plane > 0;
    }
};


} // namespace duckdb
