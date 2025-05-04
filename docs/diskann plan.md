Great. I’ll design a detailed and structured implementation plan to rewrite `vectordiskann.c` into a modern `lm_diskann.cpp` C++ module that integrates with DuckDB’s storage model, supports dynamic and bulk vector operations (insert, delete, search, clear), and uses DuckDB’s native vector types (int8, fp16, fp32).

The plan will prioritize compatibility with HNSW-style operations and DuckDB’s indexing/storage APIs, ensuring everything is disk-resident with zero RAM reliance like LM-DiskANN. I’ll explore necessary type abstractions, suggest file/module breakdowns, and document architectural interfaces for `hnsw_index` integration.

I’ll get started and let you know once the structured plan is ready for review.


# Implementation Plan: DuckDB LM-DiskANN Index

This plan details rewriting the `vectordiskann.c` implementation (LM-DiskANN in libSQL) as a native DuckDB extension module (`lm_diskann.cpp`) that fully integrates with DuckDB’s vector index framework.  We will port **all** functionality (index creation, opening, closing, search, insert, delete, and serialization) into C++ using DuckDB’s storage model (e.g. `LinkedBlock` and `FixedSizeAllocator`). The new index will use DuckDB’s fixed-length `ARRAY` vector types (e.g. `TINYINT[N]` for int8, `FLOAT[N]` for fp32, etc.) and expose the same APIs that the existing `HNSWIndex` expects (scan state, serialize/deserialize, insert, delete, query).  We will replace the old SQLite/Usearch I/O and distance code with DuckDB abstractions and built-in distance functions (e.g. `array_distance`, `array_cosine_distance`). The design follows the LM-DiskANN algorithm (Pan 2023) as described in Turso’s libSQL blog.

## 1. Architecture and Functional Components

### Data Layout (Disk Storage)

* **Fixed-size blocks (nodes):**  Each index entry (vector) is stored in a *fixed-size binary block* on disk (e.g. 4 KiB, as in LM-DiskANN).  A block contains:

  * **Node ID:** the unique identifier (row ID or key) of the vector.
  * **Raw vector data:** the uncompressed vector coordinates (fp32, fp16, or int8) of the node.
  * **Neighbor IDs list:** the list of neighbor node IDs in the graph.
  * **Compressed neighbor vectors:** the neighbors’ vector data in a compressed format (e.g. quantized to fp16 or int8).
    (We may include padding to align to block boundaries, similar to Fig.2 of LM-DiskANN【96†】.)

  &#x20;*Figure: Each LM-DiskANN index node is a disk-resident block containing the node’s ID and raw vector, plus its neighbors’ IDs and compressed vectors.*

* **Storage management:** We will allocate these blocks using DuckDB’s **FixedSizeAllocator** (for fixed-record allocation) and link them via **LinkedBlock** chains if multiple blocks per node are needed (e.g. if dimension is very large or many neighbors). DuckDB’s `BlockManager` will manage I/O to the on-disk database file; all reads/writes go through these abstractions, not manual file I/O. This ensures the index is *disk-resident* (no WAL) and managed by DuckDB’s buffer manager.

* **Metadata:** The index metadata (dimensions, metric type, block size, total nodes, etc.) will be stored in a small header block or a dedicated catalog table. For example, we can reserve the first few bytes of the file for a header with fields like `dim`, `dType` (e.g. int8/fp16/fp32), `metric`, and an index identifier. DuckDB’s catalog should store an entry (e.g. in `TableCatalogEntry` or `IndexCatalogEntry`) linking the table/column to this DiskANN index, similar to how HNSW indexes are stored.

* **RowID ↔ Block mapping:**  We need a fast way to map a table row (vector) to its block in the index. We can choose a convention: e.g. block number = rowid or use a secondary map (e.g. a DuckDB ART index) from rowid to block offset. Since each block includes the ID, we may simplify by storing blocks sequentially in insertion order and remembering an offset. For simplicity, a `LinkedBlock` list or DuckDB’s persistent segment can serve as the storage for blocks, keyed by an internal node ID equal to the table’s rowid.

### Search Engine (Query Execution)

* **Best-First Graph Search:**  The query algorithm is **Best-First Search** on the DiskANN graph. We will implement the standard DiskANN search procedure:

  1. **Initialization:** choose a random entry point or use a fixed entry node. Initialize a *candidate set* (priority queue) of size `L` containing the entry.
  2. **Search loop:** Pop the closest candidate, mark it visited, then read its block from disk. Compute distances between the query vector and that node’s **compressed neighbors** (which are already on the block). For each neighbor, if unvisited, insert into the candidate set.
  3. **Pruning:** Keep only the top-`L` candidates in the queue (discard farther ones).
  4. **Termination:** After exhausting or reaching a stopping condition, the best candidates correspond to the approximate top-k. The algorithm typically starts at a random node and rapidly converges on the nearest region of the graph.
     Key point: because each block contains its neighbors’ compressed vectors, no extra I/O is needed to fetch neighbors for distance computations. All neighbor data is local to the block, enabling purely **local block inspection** for each visited node.

* **Distance Metrics:**  DuckDB’s built-in distance UDFs will be used to measure closeness. For example, for cosine distance we call `array_cosine_distance`, for L2 use `array_distance` (which is L2-norm squared). These functions can operate on DuckDB array vectors; we will convert or load the query vector into a `data_ptr_t` buffer and compute distances in C++ code using these routines. This replaces any usearch SIMD routines.

### Index Operations (Create/Open/Drop/Modify)

* **Create Index:** When a user issues `CREATE INDEX … USING DiskANN(col)`, the extension will call a function (e.g. `LMDiskANNIndex::Create`) that:

  * Validates the column type (fixed-length ARRAY of compatible numeric type).
  * Reads index parameters from SQL (e.g. block size `bs`, candidate size `L`, metric, `efConstruction`, etc.) from the `WITH (...)` clause.
  * Initializes an empty index structure: allocates the header block, prepares the allocator, and writes metadata.
  * *Bulk Load (optional):* If the table already has data, we may initially insert all rows into the new index. This could be done by scanning the table and calling the insert routine on each vector. For very large tables, we could also allow a bulk-load path (batch building neighbors), though a simple sequential insert is acceptable initially.

* **Open Index:** When DuckDB opens the database or uses the index for a query, it should load the metadata and make the index available. We do **not** necessarily load the entire index graph into memory; by design (LM-DiskANN) only parts of the index are read on demand. We will create an `LMDiskANNIndex` object in C++ that retains the metadata and underlying BlockManager references so it can read blocks during queries. No buffer pool loading is needed beyond normal on-demand page reads.

* **Drop Index:** Corresponds to deleting the underlying blocks and metadata, similar to dropping a table or index. We must free all blocks via the allocator (which informs the storage manager to reclaim space). Then remove catalog entries.

* **Insert (Add Node):** For dynamic workloads, support inserting a new vector:

  1. Assign a new node ID (typically the table’s new rowid).
  2. Compress its vector coordinates (if applicable) into fp16/int8 format for storage.
  3. Use the existing index graph to find its neighbors: e.g. run a top-k search on the current index to get approximate nearest neighbors of the new point.
  4. Update the graph: for each neighbor found, possibly add each other to their neighbor lists (subject to maximum degree) and record the new node’s neighbors.
  5. Write the new node’s block with ID, vector, neighbor list, compressed neighbor vectors.
  6. Update metadata (increment node count, etc).
     (If immediate insertion into neighbors is too costly for large scale, we may adopt lazy strategies as HNSW does: mark new connections and prune later.)

* **Delete (Remove Node):** For removing a vector:

  1. Locate the node’s block (via its ID). Mark it deleted (e.g. set a tombstone flag or simply ignore it in searches).
  2. Optionally, remove the ID from neighbors’ lists (or mark those entries invalid). This can be deferred or handled during a periodic compaction.
  3. Reclaim the block’s space via the allocator if needed.

* **Bulk Updates:** Because streaming workloads are important, the index must allow *many* inserts/deletes without full rebuild. We should support incremental updates as above. For large bulk inserts, we may provide a faster mode (e.g. a specialized bulk loader that rebuilds large portions of the graph in parallel), but even naive per-row inserts should be possible.

### Index Metadata and Serialization

* **Metadata Layout:** All index parameters (dimensionality, metric, block size, etc.) should be persisted in a header and read on index open. We may also store: `max_degree`, `quantization_type`, `version`, etc.

* **Persistence:** On a DuckDB disk-backed database, checkpoints will persist all changes including our index blocks. Unlike HNSW (which currently writes the entire index on each checkpoint), our disk-based design naturally writes incremental updates to the index blocks.  Serialization support is minimal: after every insert/delete the underlying blocks are already on disk (via DuckDB’s buffer manager), so normal checkpoint/flush semantics suffice. For completeness, we will implement (and test) explicit `Serialize()` and `Deserialize()` methods that write/read the header and optionally pre-load an initial portion of the index if needed.

* **Scan State:** Similar to HNSW, we will define an `LM_DiskANNScanState` struct to hold the state of an ongoing query: it contains the query vector (in DuckDB’s vector format), the priority queue of candidates, visited flags, and parameters like `k` and `L`. Each invocation of the vector scan will initialize this state and run the search algorithm as above.

## 2. DuckDB Integration

### Storage Model (LinkedBlock, FixedSizeAllocator)

* **LinkedBlock:**  Use DuckDB’s `LinkedBlock` to chain multiple blocks for large entries. In practice, if one block per node (e.g. 4096 bytes) is sufficient for typical dimensions and degrees, we may only need single-block entries. If a node’s data (vector + neighbors) exceeds one block, `LinkedBlock` allows linking extra blocks seamlessly. This ensures we adhere to DuckDB’s block-based I/O model.

* **FixedSizeAllocator:**  We will create a `FixedSizeAllocator` instance configured with our chosen block size. This allocator allocates and frees fixed-size chunks from DuckDB’s block manager, exactly what our index needs for fixed-length blocks. On drop, we free all chunks back to the database.

* **No SQLite I/O:**  Replace any `sqlite3_blob_open` or file I/O calls from the original C code with calls to DuckDB’s storage APIs. For example, to read a node block we fetch the block via DuckDB’s `BlockManager::Read` and parse its bytes; to write, we allocate a block from `FixedSizeAllocator`, write into it, and flush via `BlockManager::Write`. DuckDB’s storage model will handle caching and persistence.

* **WAL Bypass:**  We explicitly disable any WAL logging for the index operations (no need for a separate WAL file). All writes go directly into the on-disk database file (in the same way duckdb writes table data). This satisfies the “purely disk-resident, no WAL” requirement.

### Vector Types (int8, fp16, fp32)

* **DuckDB Array Types:**  We use DuckDB’s fixed-length `ARRAY` (e.g. `FLOAT[dim]`, `SMALLINT[dim]`, etc.) for storing and passing vectors. For **fp32**, use `FLOAT`. For **int8**, use `TINYINT`. DuckDB does not natively support a 16-bit float type, so for **fp16** we will store data either as `SMALLINT[dim]` and reinterpret bits, or convert to `FLOAT` on load and compress to `int16` on disk. We will treat fp16 as a storage/compression format; queries still operate in `FLOAT` (fp32).
* **USearch Replacement:**  The old VSS/HNSW implementation used Unum’s **uSearch** library for SIMD distance kernels. We will **not** embed uSearch here. Instead, for each neighbor’s compressed vector we simply read it into a local `double` or `float` buffer and use DuckDB’s distance functions (`array_distance`, etc.) in C++ to compute distances. This cleanly eliminates an external dependency and keeps the code in DuckDB’s vector framework. DuckDB’s distance functions are efficient and sufficient for our needs.

### DuckDB Extension Hooks

* **Extension Registration:** In the `duckdb_vss` extension initialization code, we will register a new index type (e.g. `IndexType::DISKANN`) alongside HNSW. This involves adding parser support (so that `CREATE INDEX ... USING DiskANN` is recognized) and linking it to our C++ `LMDiskANNIndex` class. The extension C entry-point (e.g. `LMDiskANN_Init`) will call `ExtensionUtil::RegisterIndex(IndexType::DISKANN, ...)`.

* **Planner and Physical Operator:**  We will create a new physical operator node (e.g. `PHYSICAL_DISKANN_SCAN`) akin to `HNSW_INDEX_SCAN`. In the planner, queries of the form `ORDER BY array_distance(col, :query) LIMIT k` on a table with a DiskANN index should be rewritten to use our operator. The operator’s `GetChunkInternal` will run the LM-DiskANN search (using the scan state) to retrieve the top-k IDs, then fetch actual rows. This mirrors how HNSWIndex is integrated. The query executor thus sees the disk-based index transparently as an accelerated method for nearest-neighbor queries.

* **Scan State and API Compatibility:**  To ensure compatibility, our `LMDiskANNIndex` class will implement the same interface that `HNSWIndex` expects. For example, if HNSWIndex has methods like `InitializeScan(DuckDbHNSWScanState &state)`, `Scan(DuckDbHNSWScanState &state, DataChunk &result)`, our DiskANN index will have analogous methods. We may refactor common code into shared abstractions if possible. The important point is that the rest of the VSS extension (planner, executor) can treat both index types uniformly.

### Integration with HNSW Entry Points

* **Index APIs:**  The DiskANN index class will support the same APIs that the HNSW index uses: methods to insert a vector (called during `INSERT INTO table` on a indexed table), delete a vector (if supported), and query the index. We will reuse or mimic the hooks in `hnsw_index.cpp` for updates so that inserts on the table trigger our `LMDiskANNIndex::Insert()` instead of writing to an ART/BTREE index. In practice, this means adding a `case` in the VSS `IndexInsert` dispatcher for our new index type.

* **Serialization/Deserialization:**  HNSWIndex uses serialization on checkpoint; similarly, we provide `Serialize()` and `Deserialize()` methods for our index. The serialize operation will write any in-memory caches (if any) and the index header; the deserialize will read metadata. However, because the graph is on-disk already, there is little to do besides reopening the allocator and possibly verifying checksum or version. This complements the built-in buffer cache.

## 3. Code Organization (Files/Modules)

To keep the implementation modular and clear, we propose the following file split under the DuckDB VSS extension (`src/`):

* **`lm_diskann_index.hpp / lm_diskann_index.cpp`**:
  Defines the `LMDiskANNIndex` class and related types. Contains the entry-point methods for *index lifecycle*: `Create()`, `Open()`, `Drop()`, plus `Insert()`, `Delete()`, and methods to register the index type with DuckDB. This file wraps DuckDB storage calls (allocators) and high-level control flow.

* **`lm_diskann_search.cpp`**:
  Implements the core search algorithm. Contains helper functions and classes like `LMDiskANNScanState`, `DistanceHeap` (priority queue), and the best-first search loop. Also includes conversion between DuckDB vectors and raw buffers, and distance computation using DuckDB UDFs.

* **`lm_diskann_serializer.cpp`**:
  Handles reading/writing index blocks to disk. Functions here include `ReadNodeBlock(node_id)` and `WriteNodeBlock(node_id, node_data)`, using `FixedSizeAllocator`. Also the index header (metadata) read/write. This isolates the low-level I/O from the logic.

* **`lm_diskann_types.hpp`**:
  Defines any shared types, constants, and enums: e.g. `enum DiskANNMetric { L2, COSINE, IP };`, a struct for `LMDiskANNParams` (dim, bs, L, etc.), and perhaps `union CompressedVector` if needed.

* **`lm_diskann_bind.cpp`**:
  Integration glue. Contains the extension load/unload functions (e.g. `LMDiskANN_Init`) which register the index type and any SQL functions. Also the physical operator implementation for the DiskANN scan node (if not inlined in the index class). It wires our code into DuckDB’s extension framework.

Within these files, we will clearly separate interface (DuckDB-facing) from implementation (C-style algorithm). For example, `lm_diskann_index.cpp` may contain the `sqlInterface_*` functions called by the DuckDB parser (similar to how `hnsw_index.cpp` has them), and call into the C++ class methods.

## 4. Type and Interface Mappings

We translate C structs and SQLite types into DuckDB C++ equivalents. A representative mapping is summarized below:

| **Old (SQLite/libSQL)**                            | **New (DuckDB C++/Extension)**                    | **Notes**                               |
| -------------------------------------------------- | ------------------------------------------------- | --------------------------------------- |
| `DiskAnnIndex *pIndex`                             | `unique_ptr<LMDiskANNIndex>` or `LMDiskANNIndex*` | Use a C++ class instance                |
| `VectorIdxParams *pParams`                         | `LMDiskANNParams` struct                          | Holds index parameters (dim, metric)    |
| `sqlite3_blob *`                                   | `duckdb::block_id_t` / `LinkedBlock`              | Use DuckDB storage for blocks           |
| `sqlite3_int64 rowid` (node ID)                    | `idx_t rowid`                                     | DuckDB’s `idx_t` (typically 64-bit)     |
| Raw C `float *` or `double *`                      | `duckdb::data_ptr_t` or `vector<float>`           | Copy to/from DuckDB vectors             |
| SQLite BLOB column                                 | DuckDB `ARRAY` column (e.g. `FLOAT[dim]`)         | Use fixed-size arrays                   |
| `sqlite3_mprintf`, `sqlite3_free`                  | Throw `duckdb::Exception`                         | Use DuckDB error handling               |
| `sqlite3_malloc`                                   | `DuckDB::Allocator` (or C++ `new`)                | C++ memory allocation                   |
| Custom function interface (e.g. OP\_OpenVectorIdx) | DuckDB index registration APIs                    | Hook into DuckDB’s extension mechanisms |
| Usearch distance/compression calls                 | DuckDB UDFs (`array_distance`, etc.) or custom    | Replace SIMD code with builtin funcs    |

By using DuckDB’s native abstractions, the code is cleaner and safer. For example, rather than managing raw byte pointers for blocks, we will treat each block as a `duckdb::data_ptr_t` returned by `LinkedBlock::Ptr()`. Vector math will leverage DuckDB’s optimized kernels rather than custom loops.

## 5. Integration with DuckDB and HNSW

* **Extension System:**  We rely on DuckDB’s extension framework to introduce the new index type. As documented, DuckDB allows indexes via extensions. In the VSS extension, we will add an entry for `DiskANN` so that `CREATE INDEX ... USING DiskANN` is valid SQL. At runtime, DuckDB will instantiate our `LMDiskANNIndex` object for the relevant table and column.

* **HNSW-Compatible API:**  The VSS executor expects vector indexes to support scanning via a `ScanState` struct and certain methods (e.g. open, next, close). Our `LMDiskANNIndex` will offer identical entry points. For instance, if `HNSWIndex` has a method `ScanInitialize`, we implement `DiskANNScanInitialize` that sets up the LM-DiskANN algorithm. Likewise, insertion and deletion APIs in VSS (triggered by `INSERT`/`DELETE` SQL) will call our `InsertNode`/`DeleteNode` methods. This makes DiskANN a drop-in alternative to HNSW within DuckDB’s planner and executor.

* **Substituting usearch:**  The old VSS code used Unum’s USearch for fast vector math. We will **remove** any direct use of USearch. Instead:

  * For **vector distance**, we call DuckDB’s array functions (e.g. `DoubleVector::InnerProduct`, or simply evaluate `array_distance` UDF on each neighbor vector we read).
  * For **vector compression**, if the old code used USearch PQ (product quantization) or other schemes, we will implement a similar quantizer ourselves or use a simple scheme (e.g. linear quantization to fp16 or int8). This can be done inline in C++ using standard libraries. Over time, one could integrate an efficient open-source compressor, but that is beyond the initial plan. The key is to remove any dependency on USearch in our codebase so that the index lives entirely within DuckDB’s code.

## 6. Component Summary

Putting it together, the implementation will have the following components:

* **Index Metadata:** A header structure on disk (and in DuckDB catalog) storing dimension, metric, block size, etc. Mapped to C++ in `LMDiskANNParams`.
* **Block Manager:** A `FixedSizeAllocator` for disk blocks, used by `LinkedBlock`s to store each node.
* **Index Class (`LMDiskANNIndex`):** C++ class with methods:

  * `Create()`, `Drop()`: manage the entire index lifecycle.
  * `Open()`: load metadata on index usage.
  * `Insert()`, `Delete()`: update graph for dynamic data.
  * `Search(query_vec, k, L)`: perform best-first search and return top-k node IDs.
  * `Serialize()`, `Deserialize()`: optional persistence steps.
* **Search Module:** Implements the search loop and distance computations. Operates on DuckDB data (e.g. using `duckdb::Vector` for the query vector).
* **Integration Code:** Registers the index type and implements the physical scan operator. Hooks into the VSS extension system.
* **Type Mappings:** We systematically replace C-level constructs with DuckDB equivalents as shown above.

Throughout the implementation, we will refer to the LM-DiskANN design described in Pan et al. (2023) and summarized in Turso’s blog. In particular, we ensure each algorithmic step (block format, best-first search with local decisions) matches that description. For example, as noted, “with this layout, all LM-DiskANN algorithm steps can make local decisions by inspecting only a single node block, as it already has all neighbors’ identifiers and compressed vectors”. We replicate this by designing our block layout and read routines accordingly.

## 7. Bibliography

* Pan et al., *LM-DiskANN: Low-Memory Disk-Resident ANN Search*, Turso blog (describes algorithm and block layout).
* DuckDB VSS Extension Documentation (context on custom indexes).

This plan ensures a thorough, modular reimplementation of `vectordiskann.c` in DuckDB C++ style, fully aligned with DuckDB’s architecture and the existing VSS/HNSW framework. All external dependencies (SQLite I/O, Usearch) are replaced with DuckDB-native facilities, and the index will work seamlessly with DuckDB’s extension and indexing model.
