## DuckDB VSS Extension with HNSW Index: A Deep Dive

This document explains the inner workings of the DuckDB Vector Similarity Search (VSS) extension, specifically focusing on its Hierarchical Navigable Small World (HNSW) index implementation. We'll examine the roles of the different source files and how the HNSW index integrates with DuckDB's core systems via its index API.

### 1. Extension Entry Point (`vss_extension.cpp`, `vss_extension.hpp`)

- **Purpose:** This is the main entry point for the entire VSS extension. It's responsible for registering all the components (functions, types, settings, index types) with DuckDB when the extension is loaded (`LOAD vss;`).
- **Key Actions:**
  - **`VssExtension::Load(DuckDB &db)`:** This is the core function called by DuckDB.
  - **Registering Distance Functions:** It registers scalar functions like `l2_distance`, `inner_product`, `cosine_distance`, etc., which are implemented using the `simsimd` library for efficient computation.
  - **Registering the HNSW Index Type:** Crucially, it registers the "HNSW" index type. This involves associating the name "HNSW" with the specific functions and classes that handle its creation, planning, and scanning. It uses `db.instance->GetCatalog().CreateIndexType(...)` to register the `HNSWIndex` class.
  - **Registering Optimizer Rules:** It adds custom optimizer rules (`HNSWOptimizeScan`, `HNSWOptimizeTopK`, `HNSWOptimizeJoin`, `HNSWOptimizeExpr`) to DuckDB's query optimizer pipeline. These rules are essential for rewriting query plans to utilize the HNSW index effectively.
  - **Registering Pragmas:** It registers custom PRAGMA statements (`hnsw_index_scan.cpp`) for inspecting or potentially tuning the HNSW index (e.g., `PRAGMA hnsw_index_info('index_name')`).
  - **Registering Macros:** Helper table macros like `hnsw_index_scan` (`hnsw_index_macros.cpp`) might be registered to simplify querying the index directly.

### 2. HNSW Index Definition (`hnsw_index.cpp`, `hnsw_index.hpp`)

- **Purpose:** Defines the `HNSWIndex` class, which represents an HNSW index within DuckDB's catalog and execution system. It acts as a bridge between DuckDB's generic index interface and the underlying `usearch` library (which provides the actual HNSW implementation).
- **Key Components:**
  - **`HNSWIndex` Class:** Inherits from DuckDB's `Index` base class.
  - **Metadata Storage:** Stores essential metadata about the index, such as:
    - `index_name`: Name of the index.
    - `table_io_manager`: Handles storage for the index.
    - `column_ids`: IDs of the columns indexed (usually just the vector column).
    - `unbound_expressions`: The original expressions used to create the index.
    - `index_bind_data`: Contains parsed and validated information derived from the `CREATE INDEX` statement options (like `metric`, `dim`, `M`, `ef_construction`, `ef_search`). This is populated during the binding phase.
  - **`usearch` Instance:** Holds an instance (or manages the loading/saving) of the actual HNSW graph structure, likely using the `unum::usearch::index_dense_t` template provided by the `usearch` library (`usearch/index_dense.hpp`, `usearch/index.hpp`). This is the core data structure performing the approximate nearest neighbor search.
  - **Index Parameter Handling:** Parses and validates the parameters provided in the `CREATE INDEX ... WITH (...)` clause (e.g., `metric='l2'`, `ef_search=100`).
  - **Serialization/Deserialization:** Implements methods (like `Serialize`, potentially leveraging `BlockManager`) to save the `usearch` index structure to disk and load it back when DuckDB restarts or the index is needed. The actual graph structure from `usearch` needs to be persisted.
  - **Binding Function (`GetFunction`):** Provides the `IndexBindData` containing parsed parameters.
  - **Planner Function (`GetPlanFunction`):** Returns the `HNSWIndexScanPlanner` responsible for integrating the index scan into the query plan.

### 3. Index Creation (`hnsw_index_physical_create.cpp`, `hnsw_index_physical_create.hpp`)

- **Purpose:** Handles the physical process of building the HNSW index when a `CREATE INDEX ... USING HNSW` statement is executed.
- **Key Components:**
  - **`PhysicalCreateHNSWIndex` Class:** Inherits from `PhysicalOperator`. This is the execution engine component responsible for building the index.
  - **`GetData` Method:** This method is executed by DuckDB's pipeline executor.
    - **Scan Base Table:** It scans the base table to retrieve the vector data and corresponding `rowid`s for the column being indexed.
    - **Initialize `usearch` Index:** Creates an instance of the `usearch` index (`unum::usearch::index_dense_t`) configured with the parameters specified (`metric`, `dim`, `M`, `ef_construction`, etc.).
    - **Insert Vectors:** Iterates through the retrieved vectors and inserts them one by one into the `usearch` index instance. The `rowid` is typically used as the key/label for the vector within the `usearch` index. This is where the HNSW graph is actually constructed.
    - **Error Handling:** Includes checks for vector dimensions, data types, and potential errors during insertion.
    - **Persist Index:** Once all vectors are inserted, it triggers the serialization mechanism (defined in `HNSWIndex`) to save the constructed `usearch` index graph to disk, managed by the `TableIOManager`.

### 4. Index Scanning (`hnsw_index_scan.cpp`, `hnsw_index_scan.hpp`)

- **Purpose:** Defines how DuckDB retrieves data (nearest neighbor `rowid`s and distances) from an existing HNSW index during query execution.
- **Key Components:**
  - **`HNSWIndexScanFunction`:** This is the core function registered with DuckDB for the HNSW index type. It's called by the `PhysicalIndexScan` operator during query execution.
  - **`IndexScanInitialize`:** Sets up the scan state (`HNSWIndexScanState`). This likely involves loading the persisted `usearch` index from disk if it's not already in memory and extracting the query vector and `k` (number of neighbors) from the scan operator's bindings.
  - **`IndexScan` (Main Logic):**
    - Takes the query vector and `k` from the scan state.
    - Calls the `search` method of the loaded `usearch` index instance. This method performs the actual ANN search on the HNSW graph.
    - The `usearch::index::search` method returns a list of matches (typically pairs of `rowid`s and distances).
    - Populates the output `DataChunk` with the retrieved `rowid`s and potentially the calculated distances. DuckDB uses these `rowid`s to fetch other columns from the base table if needed.
  - **`HNSWIndexScanBindData`:** Stores information needed specifically for the scan, like the query vector expression and the value of `k`.
  - **`HNSWIndexScanState`:** Holds the runtime state for a scan operator, including the materialized query vector, `k`, the loaded `usearch` index instance, and the results returned by `usearch`.

### 5. Query Planning and Optimization

- **Purpose:** To automatically rewrite user queries involving distance calculations and ordering/limiting to use the efficient HNSW index scan instead of a full table scan followed by sorting.
- **`hnsw_index_plan.cpp`:**
  - **`HNSWIndexScanPlanner`:** This class is responsible for creating the initial `PhysicalIndexScan` operator node in the physical query plan *if* an HNSW index is deemed potentially usable for a table involved in the query. It doesn't yet guarantee the index *will* be used, but it sets up the possibility.
- **Optimizer Rules (`hnsw_optimize_\*.cpp`):** These are crucial for making the index useful. They inspect the *logical* query plan tree and transform it.
  - **`HNSWOptimizeScan` / `HNSWOptimizeTopK` / `HNSWOptimizeExpr`:** These optimizers look for specific patterns in the logical plan, typically:
    1. A filter or ordering clause involving a registered distance function (e.g., `l2_distance(table.vector_col, ?)`).
    2. An `ORDER BY` clause on the distance function result.
    3. A `LIMIT` clause (specifying `k`).
    4. A reference to a table that has an HNSW index on `vector_col`.
  - **Transformation:** If the pattern matches, the optimizer rewrites the relevant part of the plan:
    - It replaces the `LogicalGet` (table scan) and subsequent `Filter`/`Order`/`Limit` operators with a `PhysicalIndexScan` operator configured to use the `HNSWIndexScanFunction`.
    - It binds the query vector and `k` (from the `LIMIT` clause) to the `PhysicalIndexScan` operator's state (`HNSWIndexScanBindData`).
  - **`hnsw_optimize_join.cpp`:** This likely handles optimizing joins where one side involves finding nearest neighbors, potentially transforming it into an index-accelerated join.

### 6. DuckDB Index Hooks and Interoperability

The HNSW index integrates with DuckDB through a well-defined C++ API for custom indexes:

1. **Registration (`VssExtension::Load`):** The extension tells DuckDB about the "HNSW" index type using `CreateIndexType`. This registration links the name "HNSW" to:
   - The `HNSWIndex` class definition.
   - Functions to handle binding (`HNSWIndex::GetFunction`), planning (`HNSWIndex::GetPlanFunction`), etc.
2. **Binding (`CREATE INDEX`):** When `CREATE INDEX ... USING HNSW WITH (...)` is parsed, DuckDB calls the registered binding function. This function validates the `WITH` parameters (metric, M, etc.) and stores them in `HNSWIndexBindData` within the `HNSWIndex` object created in the catalog.
3. **Physical Creation (`CREATE INDEX` Execution):** DuckDB's planner creates a plan containing the `PhysicalCreateHNSWIndex` operator. When executed, this operator builds the index using the logic described in section 3.
4. **Query Planning (Optimizer Phase):**
   - The `HNSWIndexScanPlanner` might initially propose an index scan.
   - The custom optimizer rules (`HNSWOptimize...`) inspect the logical plan. If a suitable pattern (distance + order + limit) is found matching an HNSW index, they rewrite the plan to *force* the use of `PhysicalIndexScan` with the `HNSWIndexScanFunction`, passing the query vector and `k`.
5. **Query Execution (Index Scan):** When the execution engine encounters the `PhysicalIndexScan` operator configured for HNSW, it calls the registered `HNSWIndexScanFunction`. This function (via its initialize/scan steps) loads the `usearch` data, performs the search using the query vector and `k`, and returns the resulting `rowid`s.

### 7. Role of Supporting Libraries/Files

- **`usearch` (`include/usearch/`):** This is the core, header-only C++ library providing the high-performance HNSW algorithm implementation, distance functions, and index data structures (`index_dense.hpp`). The DuckDB extension is essentially a wrapper around `usearch`.
- **`simsimd` (`include/simsimd/`):** A header-only C library used by `usearch` (and potentially directly by the extension's scalar functions) to provide SIMD-accelerated distance calculations (L2, cosine, inner product, etc.) for various CPU architectures.
- **`fp16` (`include/fp16/`):** Provides support for half-precision floating-point numbers (float16), potentially used for more memory-efficient vector storage or calculations if supported by `usearch` configurations.
- **`hnsw_index_macros.cpp`:** Contains implementations for any table-producing functions (macros) related to the index (e.g., `PRAGMA hnsw_index_scan(...)`).
- **`hnsw_index_pragmas.cpp`:** Implements custom PRAGMA statements for interacting with the index.
- **`hnsw_topk_operator.cpp`:** Might contain specialized physical operators related to top-k processing that could work in conjunction with or as an alternative to the standard index scan under certain conditions, although the primary mechanism seems to be the `HNSWOptimize...` rules rewriting to `PhysicalIndexScan`.
- **`CMakeLists.txt`:** Defines how to build the HNSW-specific parts of the extension.

### Adapting for LM-DiskANN

To create a DuckDB extension for LM-DiskANN, you would follow a similar pattern, replacing the HNSW-specific components:

1. **Core Index Class:** Create an `LMDiskANNIndex` class inheriting from `duckdb::Index`. It would store metadata relevant to LM-DiskANN (parameters like R, L, B, M) and manage the LM-DiskANN index structure on disk.
2. **Underlying Library:** Integrate the LM-DiskANN library instead of `usearch`. Your `LMDiskANNIndex` class would hold handles or references to the LM-DiskANN index structure.
3. **Physical Creation:** Implement `PhysicalCreateLMDiskANNIndex`. This operator would scan the base table and call the LM-DiskANN library's functions to build the Vamana graph and associated structures, saving them to disk.
4. **Index Scan Function:** Implement `LMDiskANNIndexScanFunction`. This function would:
   - Load the LM-DiskANN index structure.
   - Receive the query vector and `k`.
   - Call the LM-DiskANN library's search function.
   - Return the resulting `rowid`s.
5. **Optimizer Rules:** Create `LMDiskANNOptimize...` rules that recognize distance functions (potentially the same ones if metrics overlap) combined with `ORDER BY` and `LIMIT`, and rewrite the plan to use your `PhysicalIndexScan` configured with `LMDiskANNIndexScanFunction`.
6. **Registration:** Register the "LM_DISKANN" index type, your optimizer rules, and any specific functions or pragmas in your extension's `Load` method.
7. **Serialization:** Implement robust serialization/deserialization for the LM-DiskANN index structure, ensuring it integrates with DuckDB's storage (`TableIOManager`, `BlockManager`).

The key is to map the concepts (index definition, creation logic, search logic, planning hooks, optimizer rules) from the HNSW implementation to their equivalents using the LM-DiskANN library and its specific parameters and API calls. The DuckDB index interface provides the necessary hooks, and the VSS extension serves as an excellent example of how to use them.
