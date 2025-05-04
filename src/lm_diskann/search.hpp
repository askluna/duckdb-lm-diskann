#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LMDiskannIndex; // Forward declare to access members/types
struct LMDiskannScanState; // Forward declare scan state struct

// --- Search Algorithm ---
// Implements the core graph traversal (beam search) logic.

// Performs the beam search.
// - scan_state: Holds query info, candidates, visited set, and results.
// - index: Provides access to index parameters, storage, distance functions.
// - find_exact_distances: If true, calculates exact distances for top candidates found.
void PerformSearch(LMDiskannScanState &scan_state, LMDiskannIndex &index, bool find_exact_distances);


} // namespace duckdb

