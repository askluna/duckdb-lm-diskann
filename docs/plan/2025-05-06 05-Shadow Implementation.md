# LM-DiskANN DuckDB Extension - Phased Implementation Plan (MVP Structure)

## I. Project Objective and Adopted Architectural Framework

**Project Objective:** The principal objective of this undertaking is the systematic development and implementation of LM-DiskANN as a robust, disk-native Approximate Nearest Neighbor (ANN) indexing extension tailored for the DuckDB database management system. This extension is engineered to facilitate high-performance similarity searches across extensive vector datasets, particularly those whose storage footprints may considerably surpass available system random-access memory (RAM). Design considerations of paramount importance include the provision of comprehensive support for dynamic data modifications—encompassing insertions, deletions, and updates—alongside the meticulous preservation of transactional consistency with the primary DuckDB database. Such consistency ensures that query results derived from the index accurately reflect the visibility rules pertinent to concurrent transactional operations. The designation "robust" signifies inherent resilience against system failures, encompassing predictable recovery behavior and a steadfast commitment to data integrity, thereby minimizing the risk of data loss or corruption. Concurrently, "disk-native" underscores the architectural tenet that the principal data repository resides on persistent disk storage, with RAM being strategically employed for caching mechanisms, active computational processes, and managing intermediate state, rather than for accommodating the entirety of the index structure, which could span terabytes.

**Adopted Architectural Framework: "Shadow Architecture V2"**

The project will realize a sophisticated shadow architecture, wherein each LM-DiskANN index instance is encapsulated within its own dedicated directory structure. This architectural paradigm promotes enhanced modularity, simplifies the complexities associated with index lifecycle management (creation, backup, drop), and isolates the operational concerns of individual indexes. The key constituent components of this framework are delineated as follows:

1. **Folder-per-Index:** Each index instance is allocated an isolated directory, ensuring separation of concerns and resources. This self-contained nature means that all files pertaining to a specific index (graph data, WAL-enabled store, metadata) are co-located, simplifying administrative tasks such as targeted backups or the complete removal of an index without impacting other database objects. The path to this directory will be managed by DuckDB and persisted within its catalog.
2. **Primary Graph File (`graph.lmd`):** This file serves as the persistent store for fixed-size `NodeBlock`s, which collectively represent the main graph structure. It is optimized for sequential read performance during graph traversals and for efficient random access writes during the merge process. The fixed-size nature of `NodeBlock`s is fundamental to this efficiency, allowing direct offset calculation for any given `node_id`.
3. **Consolidated Index Store (`diskann_store.duckdb`):** A Write-Ahead Logging (WAL)-enabled embedded DuckDB database is provisioned on a per-index basis. This subordinate database acts as the transactional backbone for all mutable auxiliary data associated with the index, thereby leveraging DuckDB's own proven mechanisms for atomicity, consistency, isolation, and durability (ACID properties). Its components include:
   - **Shadow Delta Table (`__lmd_blocks`):** Functions as a staging area for new or updated `NodeBlock` data prior to their merge into the primary graph file. Its schema is defined as: `(block_id BIGINT PRIMARY KEY, data BLOB, version BIGINT, commit_epoch BIGINT, tombstone BOOLEAN, checksum BIGINT)`. The `PRIMARY KEY` on `block_id` ensures efficient lookup and replacement of the latest version of a block in the shadow state. The `version` field is crucial for optimistic concurrency and for resolving potential write conflicts during merge operations.
   - **Lookup Mapping Table (`lmd_lookup`):** Maintains the crucial mapping from base table `row_id` values (which are stable identifiers for rows within DuckDB tables) to the index's internal `node_id` values (which are typically dense identifiers used for offset calculation within `graph.lmd`). Its schema is: `(row_id BIGINT PRIMARY KEY, node_id BIGINT UNIQUE NOT NULL)`. An additional unique index on `node_id` is essential for efficient reverse lookups (mapping `node_id` back to `row_id`), a common requirement when presenting search results.
   - **Metadata Table (`index_metadata`):** Persistently stores critical index configuration parameters (e.g., vector dimensionality, graph degree `R`, distance metric) and dynamic state information (e.g., total number of nodes, current entry point `node_id`, merge sequence number) in a transactionally consistent manner. This obviates the need for a separate, custom metadata file and its associated complexities regarding atomic updates.
   - **Tombstone Table (`tombstoned_nodes`):** Provides a mechanism for tracking logically deleted nodes within the index, facilitating their exclusion from search results and their eventual physical reclamation. Its schema is: `(node_id BIGINT PRIMARY KEY, deletion_epoch BIGINT)`. The `deletion_epoch` can be used to determine when a tombstone is old enough to be considered for physical removal or to manage visibility for time-travel queries if such functionality were ever introduced.
4. **`NodeBlock` Structure:** Each `NodeBlock` is a meticulously defined, fixed-size data structure (e.g., 8KB or 16KB, determined at index creation based on dimensionality and `R`). It contains: `node_id` (its unique identifier within the graph), `row_id` (the corresponding base table row identifier, for verification and potential recovery), `version` (for optimistic concurrency control), `origin_txn_id` (the identifier of the transaction that created or last modified this block version), `commit_epoch` (the commit sequence number or timestamp, crucial for MVCC), a `tombstone` flag (boolean, indicating logical deletion), the raw vector data itself, a list of neighbor identifiers along with their compressed vector representations (to accelerate approximate distance calculations during traversal), a checksum for integrity verification against corruption, and requisite padding to meet the fixed block size. The choice of compression scheme for neighbor vectors (e.g., Product Quantization, Ternary Quantization) is a significant design decision impacting both storage efficiency and search performance.
5. **In-Memory LRU Cache:** An In-Memory Least Recently Used (LRU) cache is employed for `NodeBlock` objects. This cache stores deserialized `NodeBlock` instances to optimize access to frequently or recently utilized graph nodes, thereby reducing the latency associated with disk I/O operations. The cache will also manage dirty flags for modified blocks awaiting persistence.
6. **In-Memory Dirty Ring Buffer:** This buffer, implemented as a thread-safe circular queue, serves to stage `NodeBlock` modifications (represented as `DirtyBlockEntry` structures) prior to their asynchronous flushing to the `__lmd_blocks` table by a dedicated background daemon. This decoupling improves the responsiveness of foreground DML operations.

*(For comprehensive descriptions of these components and the overarching operational flow, including data lifecycle and interaction patterns, reference should be made to the "LM-DiskANN Shadow Architecture – Implementation & Mitigation Specification" (previously designated `2025-05-06 02-Diskann shadow design 2 mitigations.md`) and the "LM-DiskANN with Shadow Lookup Table – Technical Design & Implementation Specification" (previously `2025-05-06 01-Diskann shadow design 1 proposal.md`)).*

## II. Phased Implementation via Minimum Viable Products (MVPs)

The development of this project will be executed iteratively, progressing through a sequence of Minimum Viable Products (MVPs). Each MVP is designed to deliver a functionally complete and testable vertical slice of the system, systematically building upon the capabilities established in preceding MVPs. This methodological approach facilitates early validation of critical architectural decisions, allows for the incremental mitigation of technical risks, and enables the progressive integration of features, leading to a more robust and well-tested final product.

### MVP 0: Core Infrastructure and Static Index (Read-Only Capability)

**Objective:** To establish the foundational software components and infrastructure necessary for the creation, persistent storage, and execution of basic search operations on a static, read-only LM-DiskANN index. This initial MVP is strategically focused on the delineation and implementation of the core graph structure (`graph.lmd`) and its fundamental interactions with the DuckDB system, including index creation and basic scan invocation. Considerations related to dynamic updates (insertions, deletions occurring post-build) or complex transactional semantics (MVCC beyond a static snapshot) are deliberately deferred to subsequent phases to manage initial complexity.

**Key Components and Features:**

1. **Basic Directory and File Management System:**
   - Implementation of procedures for the creation of the index-specific directory structure (e.g., `database_name.lmd_idx/index_name/`). This includes robust path handling and error checking for filesystem operations.
   - Initialization of an empty `graph.lmd` file. This involves writing a requisite header structure containing a magic number for file type identification, a format version indicator (to support future layout changes), and static configuration parameters such as vector dimensionality, chosen distance metric, and configured block size.
   - Initialization of the `diskann_store.duckdb` database. This entails programmatically creating the embedded database file and executing Data Definition Language (DDL) statements to define the schemas for the `lmd_lookup` and `index_metadata` tables, as referenced in the "Shadow Architecture V2," Section I.3.
2. **`NodeBlock` Definition and Serialization Mechanisms:**
   - Definition of the C++ `struct NodeBlock` in strict accordance with Section I.4 of the "Shadow Architecture V2" specification. Internal helper structures related to `NodeBlock` (e.g., for neighbor entries) will adhere to C++ guidelines, such as the principle of parameter grouping for constructors or methods that would otherwise possess an excess of three parameters, promoting clarity and maintainability.
   - Implementation of rudimentary serialization routines for converting in-memory `NodeBlock` instances to `char*` buffers suitable for disk persistence. Corresponding deserialization routines for reconstructing `NodeBlock` instances from such buffers will also be developed. These routines must handle byte ordering consistently if cross-platform compatibility is a concern, although DuckDB's primary environments are typically little-endian.
   - Implementation of checksum calculation algorithms (e.g., CRC32 or xxHash) and associated verification mechanisms to ensure the integrity of `NodeBlock` data against corruption during storage or transmission.
3. **`IndexStoreManager` (Rudimentary Implementation):**
   - Development of the basic class structure for the `IndexStoreManager`. This class will encapsulate interactions with `diskann_store.duckdb`. Its initial responsibilities will include initializing the embedded database and programmatically creating its constituent tables (`lmd_lookup`, `index_metadata`).
   - Implementation of fundamental methods for populating the `lmd_lookup` table (mapping `row_id` to `node_id`) and the `index_metadata` table (storing configuration details) during the static index build process.
4. **Serial Build Process (Simplified for Static Index Construction):**
   - Adaptation of the Vamana graph construction algorithm for serial execution, as detailed in the "Technical Specification: LM-DiskANN Graph Index Integration with DuckDB," Section "Serial Graph Construction." This involves iterating through the source data, performing greedy searches for neighbors, and applying robust pruning.
   - This process, for MVP 0, will involve direct writes of finalized `NodeBlock` data to the `graph.lmd` file.
   - Population of the `lmd_lookup` and `index_metadata` tables within `diskann_store.duckdb` will be performed transactionally upon the successful completion of the entire build operation.
   - The build procedure shall direct its output to temporary files (e.g., `graph.lmd.tmp`, `diskann_store.duckdb.tmp`). Upon successful completion of all build stages, an atomic rename operation will be performed to replace any existing index files with the newly built ones. This strategy, outlined in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.3, "Build Pipeline Robustness," ensures that the index remains in a consistent state even if the build process is interrupted.
5. **Basic `ReadNodeBlock` Functionality:**
   - Implementation of a function to read a specified `NodeBlock` directly from the `graph.lmd` file, given its `node_id` (which maps to a file offset).
   - This function will be responsible for performing the necessary deserialization of the byte buffer into the `NodeBlock` C++ structure and for verifying the block's integrity using its checksum.
6. **Basic Search Logic (`PerformSearch` - V0 Implementation):**
   - Implementation of a standard beam search graph traversal algorithm.
   - This initial version will utilize the basic `ReadNodeBlock` function, which, for MVP 0, reads exclusively from the `graph.lmd` file. The complexities of querying shadow data or consulting a cache are deferred.
   - Filtering based on `allowed_ids` (predicate pushdown) will not be incorporated in this MVP; the search will operate over the entire graph.
   - Complex MVCC checks are deferred. All data within the statically built `graph.lmd` is assumed to be visible and valid for the purposes of any query in this MVP.
7. **DuckDB Integration (Core API Hooks):**
   - Implementation of parsing logic for the `CREATE INDEX ... USING LM_DISKANN (...)` SQL syntax. This will likely be managed via an `LMDiskannConfig::ParseOptions` method or an equivalent mechanism within the extension framework, responsible for interpreting `WITH` clause parameters.
   - Implementation of the `LMDiskannIndex` constructor. This constructor will handle index creation parameters (e.g., column to index, distance metric, graph construction parameters like `R` and `L_insert`) and will trigger the serial build process. Parameter handling will conform to C++ guidelines; for instance, if the constructor requires more than three parameters, these will be grouped into a dedicated options structure.
   - Implementation of a basic `LMDiskannIndex::Scan` method. This method will invoke the V0 `PerformSearch` function and will be responsible for returning the top-K results. A crucial part of this step is mapping the internal `node_id` values (returned by the search) back to base table `row_id` values, which is achieved by querying the `lmd_lookup` table.
   - Implementation of the `LMDiskannIndex::CommitDrop` method to ensure the clean and complete removal of the index directory and all associated files when a `DROP INDEX` command is executed.
8. **Basic Metadata Persistence (`Serialize`/`Deserialize` - V0 Implementation):**
   - `Serialize`: Implementation of the serialization hook to write essential metadata to DuckDB's checkpoint stream. This will include the relative path to the index directory, core configuration parameters (such as vector dimensions, `R`, `L_insert`, `alpha`, and the chosen distance metric), and the format version of the index files.
   - `Deserialize`: Implementation of the deserialization hook to read this metadata during database startup. Rudimentary validation procedures will be implemented at this stage, including checks for the existence of the index directory and files, and verification of the `graph.lmd` header (magic number, format version compatibility), as suggested in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.3, "Storage and Block Format Versioning."

**Implementation Tasks (Derived from Comprehensive Plan, Sections III & VII):**

- Development of Task III.1 (Directory and File Management System), modified for MVP 0 to exclude the `__lmd_blocks` and `tombstoned_nodes` tables from `diskann_store.duckdb` at this initial stage.
- Development of Task III.2 (`NodeBlock` Definition and Serialization Mechanisms), encompassing the `NodeBlock` C++ structure definition, basic serialization/deserialization routines, and checksum calculation/verification.
- Development of Task III.3 (Low-Level `graph.lmd` I/O Layer), providing functions for direct block reads from `graph.lmd`.
- Development of Task III.4 (`IndexStoreManager`), in its rudimentary MVP 0 form, responsible for `lmd_lookup` and `index_metadata` table schema creation and their initial population during the static build process.
- Development of Task VII (Serial Build Process), adapted to write directly to `graph.lmd.tmp` and to populate `diskann_store.duckdb.tmp`, followed by an atomic rename to finalize the build.
- Implementation of the `ReadNodeBlock` function (MVP 0 version: reads exclusively from `graph.lmd`, performs deserialization and checksum validation).
- Implementation of the `PerformSearch` function (MVP 0 version: utilizes a single candidate heap for beam search, without MVCC considerations or `allowed_ids` filtering capabilities).
- Implementation of DuckDB API hooks for `CREATE INDEX` (triggering the build), `Scan` (providing basic top-K functionality by invoking `PerformSearch` and mapping results), and `DROP INDEX` (ensuring cleanup).
- Implementation of `Serialize`/`Deserialize` methods (MVP 0 version: handling basic index metadata such as path, configuration, and format version for persistence and reload).

**Testing Focus:**

- Verification of the successful creation of an LM-DiskANN index on a representative vector dataset.
- Confirmation of the correct instantiation and population of the `graph.lmd` file and the `diskann_store.duckdb` database (specifically, the `lmd_lookup` and `index_metadata` tables within it).
- Assessment of the index's ability to be correctly loaded and recognized by DuckDB following a system restart.
- Evaluation of basic k-NN search operations to ensure they return plausible (though not necessarily perfectly accurate in terms of graph quality, nor MVCC-aware) results against the static graph.
- Verification that `DROP INDEX` operations result in the complete and clean removal of all associated index files and directories from the filesystem.

### MVP 1: Basic Dynamic Updates (Insertions) and Core Shadowing Mechanism Implementation

**Objective:** To introduce the core components of the shadow architecture, thereby enabling support for dynamic insertions of new vectors into a pre-existing, statically built index. This MVP is primarily focused on establishing and validating the essential elements of the write path, which involves the In-Memory LRU Cache for `NodeBlock`s, the In-Memory Dirty Ring Buffer for staging modifications, the background Flush Daemon for asynchronous persistence, and the `__lmd_blocks` table within `diskann_store.duckdb` for durable delta storage.

**Key Components and Features:**

1. **LRU Cache Implementation:**
   - Development of a thread-safe Least Recently Used (LRU) cache. This cache will be designed to store `std::shared_ptr<NodeBlock>` instances, facilitating efficient memory management of deserialized blocks.
   - The cache must effectively manage `is_dirty` flags associated with its entries, indicating whether a cached block has been modified and requires flushing to persistent storage.
2. **In-Memory Dirty Ring Buffer Implementation:**
   - Development of a thread-safe ring buffer (circular queue) intended for staging `DirtyBlockEntry` structures.
   - The `DirtyBlockEntry` structure will encapsulate essential information for processing dirty blocks, including the `block_id` (i.e., `node_id`), a `std::shared_ptr<NodeBlock> block_data` (referencing the modified block instance in the LRU cache), the `origin_txn_id` (identifier of the transaction that initiated the change), and an `std::atomic<uint64_t>* commit_epoch_target_ptr`. This latter pointer facilitates the late finalization of the `commit_epoch` on the in-memory `NodeBlock` by the Flush Daemon.
3. **Flush Daemon Implementation:**
   - Implementation of a background thread or a recurring DuckDB task responsible for asynchronously processing entries from the In-Memory Dirty Ring Buffer.
   - **Critical Note for MVP 1:** The mechanism for verifying the commit status of `origin_txn_id` may be simplified in this initial MVP. For instance, it might initially assume all transactions are committed, solely for the purpose of thoroughly testing the flush pathway itself. Full, robust integration with DuckDB's transaction manager for status checking is a more complex undertaking deferred to a subsequent MVP.
   - The daemon will be responsible for assigning a provisional `commit_epoch` to the blocks it processes (e.g., using a monotonically incrementing counter specific to this MVP, or a simplified timestamp).
   - It will serialize the `NodeBlock` instances (retrieved via `block_data` from `DirtyBlockEntry`) and write them to the `__lmd_blocks` table within `diskann_store.duckdb`, utilizing the `IndexStoreManager` for database interaction.
4. **`IndexStoreManager` Enhancements:**
   - Addition of methods to the `IndexStoreManager` class to perform `INSERT OR REPLACE` operations into the `__lmd_blocks` table. These methods will handle the binding of serialized `NodeBlock` data and associated metadata (version, epoch, tombstone status, checksum) to prepared statements.
   - Addition of methods for querying the `__lmd_blocks` table by `block_id`, enabling the retrieval of the latest shadow version of a given block.
5. **`ReadNodeBlock` Enhancements (V1 Specification):**
   - Implementation of an updated lookup hierarchy for retrieving `NodeBlock`s: the function will first check the In-Memory LRU Cache, then query the `__lmd_blocks` table (via `IndexStoreManager`), and finally, if not found in either, read from the `graph.lmd` file.
   - Incorporation of basic MVCC logic: The function will compare the `NodeBlock::commit_epoch` (retrieved from the block data) with a query's snapshot epoch. For this MVP, the concept and provision of a query's snapshot epoch may also be simplified (e.g., using the current provisional epoch counter).
6. **Insertion Logic (`LMDiskannIndex::Insert` - V1 Specification):**
   - The processing of new vectors (and their corresponding nodes) will entail the following sequence:
     - Creation of a new `NodeBlock` instance within the LRU cache; this block is immediately marked as dirty.
     - Performance of a neighbor search for the new vector, utilizing the updated `ReadNodeBlock` function (which now incorporates cache and shadow lookups).
     - Any existing neighbor `NodeBlock`s that are affected by the insertion (i.e., their neighbor lists are modified to include the new node) will be updated via a copy-on-write mechanism. These updated neighbor blocks are also placed in the LRU cache and marked as dirty.
     - `DirtyBlockEntry`s for all new and modified blocks (the new node itself and any updated neighbors) will be enqueued onto the In-Memory Dirty Ring Buffer.
     - The `lmd_lookup` table (mapping `row_id` to the new `node_id`) and `index_metadata` (e.g., incrementing `num_nodes`) will be updated transactionally via the `IndexStoreManager`. These database operations must occur as part of the user's encompassing DuckDB transaction to ensure atomicity.
7. **Basic Transactional Consistency Framework:**
   - Changes to the `lmd_lookup` and `index_metadata` tables within `diskann_store.duckdb` will be transactionally consistent with the main DuckDB database operations, by virtue of these operations being part of the same user transaction.
   - `NodeBlock` data persisted in the `__lmd_blocks` table will be written transactionally (within `diskann_store.duckdb`'s context) by the Flush Daemon.
   - Introduction of an `index_version` (alternatively termed `commit_epoch_watermark`) stored in `index_metadata`, and a `block_version` (or `commit_epoch`) stored within each `NodeBlock`. This mechanism is intended to facilitate rudimentary snapshot isolation during read operations, as per the strategy outlined in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.3, "Snapshot Isolation for Read Operations." The `ReadNodeBlock` function will be enhanced to respect these versions for visibility determination.

**Implementation Tasks:**

- Development of Task III.5 (In-Memory LRU Cache), ensuring thread-safety and correct dirty flag management.
- Development of Task III.6 (In-Memory Dirty Ring Buffer), ensuring thread-safe producer-consumer semantics.
- Development of Task III.7 (Flush Daemon), incorporating a simplified transaction commit check for MVP 1 and basic epoch assignment logic. The daemon must correctly interact with the Ring Buffer and `IndexStoreManager`.
- Enhancement of Task III.4 (`IndexStoreManager`) to include methods for interacting with the `__lmd_blocks` table (insertion and querying).
- Development of Task IV.1 (`Insert` logic for MVP 1), focusing on new node creation, updates to neighbor `NodeBlock`s (copy-on-write), and correct interaction with the LRU cache and Ring Buffer.
- Enhancement of the `ReadNodeBlock` function to implement the three-tiered lookup hierarchy (cache -> `__lmd_blocks` -> `graph.lmd`) and to perform basic epoch/version-based visibility checks.
- Enhancement of the `PerformSearch` function to utilize the V1 `ReadNodeBlock`, thereby enabling searches to see recently inserted (but potentially unmerged) data.
- Implementation of the `index_version` and `block_version` (or `commit_epoch`) mechanisms for rudimentary snapshot isolation, including their storage and checking logic.

**Testing Focus:**

- Verification of the capability to successfully insert new vectors into a pre-existing, statically built index.
- Confirmation that newly inserted vectors become searchable and are visible to subsequent queries, respecting the simplified epoch mechanism implemented in this MVP.
- Validation that the Flush Daemon correctly and durably writes `NodeBlock` data to the `__lmd_blocks` table within `diskann_store.duckdb`.
- Assessment of the `ReadNodeBlock` function's ability to correctly retrieve `NodeBlock` data from the cache, then from `__lmd_blocks`, and finally from `graph.lmd`, adhering to the prescribed lookup order and visibility rules.
- Evaluation of the basic stability and correct operational behavior of the background threads, particularly the Flush Daemon, under typical insertion workloads.

### MVP 2: Merge Process Implementation, Deletion Support, and Basic Maintenance Capabilities

**Objective:** To complete the primary data lifecycle within the shadow architecture by implementing the merge process, which integrates changes from the shadow store (`__lmd_blocks`) into the main graph file (`graph.lmd`). This MVP will also introduce foundational support for vector deletions, including tombstoning mechanisms, and provide rudimentary index maintenance functionalities, such as a basic vacuum operation to trigger merges and manage space.

**Key Components and Features:**

1. **Merge/Compaction Process Implementation:**
   - Development of a background thread or a schedulable DuckDB task dedicated to executing the merge process.
   - This process will systematically read `NodeBlock` data from the `__lmd_blocks` table (typically ordered by `block_id` for optimized writes to `graph.lmd`).
   - It will then write or overwrite these `NodeBlock`s into their designated locations within the `graph.lmd` file.
   - Critically, after writing blocks to `graph.lmd`, an `fsync()` operation must be performed on the `graph.lmd` file handle to ensure that these changes are durably persisted to the physical storage medium before any metadata updates are committed.
   - Following the successful sync of `graph.lmd`, the merge process will atomically (within a single transaction on `diskann_store.duckdb`):
     - Clear the successfully merged entries from the `__lmd_blocks` table.
     - Update relevant `index_metadata` entries, such as incrementing a `merge_sequence_number` to track merge progress and consistency.
2. **Deletion Logic (`LMDiskannIndex::Delete` - V1 Specification):**
   - Implementation of procedures for the creation and management of the `tombstoned_nodes` table within `diskann_store.duckdb`, as specified in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.2. This table will store `node_id`s of logically deleted nodes and their `deletion_epoch`.
   - When a `row_id` is targeted for deletion from the base table:
     - Its corresponding `node_id` will be retrieved from `lmd_lookup` and subsequently added to the `tombstoned_nodes` table.
     - The entry for this `row_id` in the `lmd_lookup` table will be deleted. Both this deletion and the insertion into `tombstoned_nodes` must occur transactionally.
     - The `NodeBlock` associated with the deleted `node_id`, if fetched from cache or storage (e.g., for updating its neighbors), can be explicitly marked with its internal `tombstone` flag set to true and assigned an appropriate `commit_epoch`. This modified `NodeBlock` (now a tombstone version) is then pushed to `__lmd_blocks` via the ring buffer, ensuring the tombstone status is persisted.
     - Neighbor `NodeBlock`s whose adjacency lists previously contained the now-deleted `node_id` must be updated. This involves creating new versions of these neighbor blocks (copy-on-write), removing the deleted `node_id` from their neighbor lists, marking them as dirty, and pushing them to the ring buffer for persistence.
3. **`ReadNodeBlock` Enhancements (V2 Specification):**
   - The `ReadNodeBlock` function will be enhanced to explicitly check the `tombstone` flag contained within the deserialized `NodeBlock` data structure.
   - Furthermore, it will consult the `tombstoned_nodes` table (via the `IndexStoreManager`). This check is particularly important if a block is read from `graph.lmd` (as this version might predate the logical deletion) or if its own `tombstone` flag is not definitive (e.g., if the block version in `graph.lmd` is older than the `deletion_epoch` in `tombstoned_nodes`). This can also serve as a primary check for deleted status before fully deserializing a block.
4. **Search Logic Enhancements (`PerformSearch` - V1 Specification):**
   - The core search algorithm (`PerformSearch`) will be modified to correctly interpret the results from the enhanced `ReadNodeBlock` function. Specifically, it will skip or ignore any nodes that are identified as tombstones, ensuring they are not considered as candidates or used for further graph traversal.
5. **Basic Vacuum Operation (`LMDiskannIndex::Vacuum` - V1 Specification):**
   - The `VACUUM INDEX` command (or an equivalent internal maintenance trigger) will, as a primary action, initiate a merge operation if the `__lmd_blocks` table has grown to a significant size or contains a substantial number of unmerged changes.
   - It will implement the "Simpler Compaction Strategy for V1," as detailed in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.2. For MVP 2, this strategy primarily involves:
     - Focusing on rudimentary free list management for `graph.lmd` slots. The merge process, when it encounters `NodeBlock`s from `__lmd_blocks` that correspond to `node_id`s already marked in `tombstoned_nodes`, will identify these slots in `graph.lmd` as free. This free list information will be persisted in `index_metadata`. Subsequent new node insertions (which are integrated into `graph.lmd` via the merge process) will then attempt to reuse these freed slots before appending to the end of the file.
     - Implementing tail trimming of `graph.lmd`: if a contiguous range of blocks at the physical end of the `graph.lmd` file are all identified as free or tombstoned, the file can be truncated to reclaim this trailing space.

**Implementation Tasks:**

- Development of Task III.8 (Merge/Compaction Process), including coordination with the Flush Daemon, durable writes to `graph.lmd`, and atomic updates to `diskann_store.duckdb`.
- Development of Task IV.3 (`Delete` logic for MVP 2), which encompasses management of the `tombstoned_nodes` table, transactional updates to `lmd_lookup`, generation of tombstone versions of `NodeBlock`s, and updates to neighbor lists.
- Enhancement of Task III.4 (`IndexStoreManager`) to include methods for interacting with the `tombstoned_nodes` table and for managing free list metadata within `index_metadata`.
- Enhancement of the `ReadNodeBlock` function to correctly consult the `tombstoned_nodes` table and to interpret the `NodeBlock.tombstone` flag for visibility decisions.
- Enhancement of the `PerformSearch` function to ensure that nodes identified as tombstones are correctly skipped during graph traversal and are not included in search results.
- Development of Task VI (`Vacuum` operation for MVP 2), which will trigger the merge process and implement basic free list management for merge-time reuse and tail trimming of `graph.lmd`.

**Testing Focus:**

- Verification that the Merge Process correctly and durably transfers data from the `__lmd_blocks` table to the `graph.lmd` file, and that it accurately updates all associated metadata in `diskann_store.duckdb`.
- Confirmation that vectors can be successfully deleted from the index and are subsequently not found in search results, nor are they considered during graph traversal by other searches.
- Validation that tombstones (both those explicitly marked in `NodeBlock`s and those recorded in the `tombstoned_nodes` table) are correctly identified and handled by the search logic.
- Assessment of the `Vacuum` operation's ability to trigger a merge and to perform rudimentary space management (e.g., reusing freed slots during merge, tail trimming).
- Evaluation of the overall index stability and data consistency under basic scenarios involving concurrent insertions, deletions, and merge operations.

### MVP 3: Advanced Filtering Capabilities and Query-Time Enhancements

**Objective:** To significantly enhance the search capabilities of the index by implementing effective and efficient filtering using `allowed_ids` sets (predicate pushdown), and to provide initial, functional integration with DuckDB's query optimizer through the introduction of a basic cost model for the LM-DiskANN scan operation.

**Key Components and Features:**

1. **Dual Candidate Heap Search Strategy Implementation:**
   - Implement the dual candidate heap search logic (Results Heap - RH, and Exploration Heap - EH) within the `PerformSearch` function, as specified in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.1. This strategy is designed to improve recall for filtered searches.
   - The Results Heap (RH) will be responsible for storing the top-K candidate nodes that definitively satisfy the `allowed_ids` filter criteria.
   - The Exploration Heap (EH) will store promising candidate nodes for graph traversal, irrespective of their immediate presence in the `allowed_ids` set, thereby allowing the search to navigate through non-matching nodes to reach valid targets.
   - Implement a refined stopping condition for the search loop, potentially based on a comparison of the best candidate in EH versus the worst in a full RH (e.g., `dist(EH_top) > gamma * dist(RH_kth)`).
2. **Interface for `allowed_ids` Set Consumption:**
   - Define the precise programmatic interface and data structures by which the `LMDiskannIndex::Scan` method receives the `allowed_ids` set from DuckDB's query planner. This could involve, for example, a `duckdb::SelectionVector`, a `duckdb::Validities` mask, or another optimized representation provided by DuckDB, as discussed in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.1, under "Memory Footprint of `allowed_ids`."
   - Implement efficient algorithms for checking membership within the received `allowed_ids` structure during the search process.
3. **Configurable `L_search` Parameter with Runtime Warnings:**
   - Establish `L_search` (the exploration beam width) as a configurable session parameter within DuckDB, thereby allowing users or administrators to adjust the breadth of the search based on workload characteristics or performance requirements.
   - Implement a runtime warning mechanism, potentially using `duckdb::ClientContext::Warn`, that is triggered if a filtered search operation (i.e., one with an active `allowed_ids` set) exhausts its `L_search` budget but the Results Heap (RH) contains fewer than the requested `K` qualified candidates. This feedback mechanism is proposed in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.1, under "Exploration Budget (`L_search` / `ef_search`)."
4. **Basic Cost Model for Optimizer Integration:**
   - Implement the `LMDiskannIndex::ConstructCost` method (or an equivalent API provided by DuckDB for index cost estimation). This method will provide the query optimizer with an estimated cost for performing a scan using this index.
   - The V1 Cost Model will be based on parameters such as `L_search_effective` (an estimate of unique nodes visited), `R_avg` (average effective graph degree), and a `Penalty_factor_selectivity`. This penalty factor will be applied when an `allowed_ids` set is active to account for the potentially increased search effort in sparse subgraphs, as outlined in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.4.

**Implementation Tasks:**

- Development of Task IV.2 (`PerformSearch` - specifically, the implementation of the dual-heap search strategy and the integration of `allowed_ids` filtering logic within the search loop).
- Definition and implementation of the interface for receiving and efficiently processing the `allowed_ids` set within the `LMDiskannScanState` structure and the `PerformSearch` function.
- Implementation of the configurable `L_search` parameter, including mechanisms for setting it at the session level, and the associated runtime warning logic for underpopulated Results Heaps in filtered searches.
- Development of Task VI (Optimizer Integration - specifically, the implementation of the basic cost model within the `ConstructCost` method or equivalent).

**Testing Focus:**

- Comprehensive evaluation of the performance (latency) and recall of filtered search operations under a wide range of filter selectivities (e.g., from 0.1% to 50% of the dataset).
- Verification of the correctness and effectiveness of the dual-heap search algorithm implementation, particularly its ability to find relevant results in sparsely filtered graph regions.
- Assessment of the utility and impact of `L_search` configuration options, and the clarity and usefulness of the runtime warnings for underpopulated results.
- Basic validation that DuckDB's query optimizer considers the LM-DiskANN index as a potential access path in relevant query scenarios, based on the cost estimates provided by the implemented cost model.

### MVP 4: Attainment of Robustness, Full Transactional Model, and Production Hardening

**Objective:** To elevate the LM-DiskANN extension to a production-ready state by implementing a full and robust transactional model aligned with DuckDB's MVCC, ensuring comprehensive crash recovery mechanisms, refining concurrency controls as dictated by performance and stability requirements, and incorporating extensive observability features for monitoring and diagnostics.

**Key Components and Features:**

1. **Full MVCC and Transaction Manager Integration:**
   - Refine the `commit_epoch` generation and assignment mechanism. This necessitates close and reliable integration with DuckDB's `TransactionManager` to obtain authentic global commit epochs upon transaction commit and to accurately ascertain the status (committed or aborted) of any given `origin_txn_id`. This integration is of critical importance for the correct and safe operation of the Flush Daemon.
   - Ensure that the `ReadNodeBlock` function meticulously utilizes precise snapshot epochs, derived from the query's active transaction context, for all visibility determination logic, thereby guaranteeing that queries only observe data consistent with their transactional snapshot.
2. **Comprehensive Rollback Handling Procedures:**
   - Ensure that the Flush Daemon rigorously and reliably discards any `DirtyBlockEntry`s that originate from transactions that have been identified by DuckDB's `TransactionManager` as aborted. This prevents data from rolled-back transactions from being persisted to `__lmd_blocks`.
   - Conduct thorough testing of complex rollback scenarios, particularly concerning the atomicity of changes to the `lmd_lookup` and `index_metadata` tables within `diskann_store.duckdb` when the main user transaction is aborted.
3. **Advanced Concurrency Control Mechanisms (If Deemed Necessary through Benchmarking):**
   - Systematically evaluate whether the default `IndexLock` (a coarse-grained mutex provided by `BoundIndex`) offers sufficient concurrency for high-throughput DML workloads.
   - If performance benchmarks indicate that `IndexLock` is a significant contention point, investigate, design, and potentially implement finer-grained locking strategies. Examples could include per-node locks for `NodeBlock` modifications, or sharded locks for shared data structures like the LRU cache or the Ring Buffer.
4. **Robust Crash Recovery and Validation Logic:**
   - Implement thorough and multi-faceted validation procedures within `LMDiskannIndex::Deserialize` (or an associated `LoadFromStorage` method invoked during index loading). This validation must include:
     - Comparison of the checkpointed `merge_sequence_number` (read from DuckDB's main checkpoint data) with the `merge_sequence_number` stored in `diskann_store.duckdb.index_metadata`.
     - Verification of the consistency of other critical metadata items (e.g., dimensions, block size) between the checkpointed data and the index's own metadata store.
     - Definition and implementation of a clear, predictable, and safe strategy for handling detected inconsistencies (e.g., marking the index as invalid and requiring a mandatory rebuild from base table data to ensure correctness), as discussed in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.3, "Snapshot Isolation."
   - Conduct extensive crash recovery testing, simulating failures at all critical phases of operation (e.g., during flush daemon activity, during merge process execution, during concurrent DML operations).
5. **Observability and Metrics Framework:**
   - Implement a comprehensive suite of detailed counters and metrics for monitoring search operations (e.g., number of nodes visited in EH and RH, number of blocks read from disk versus cache hits, number of `allowed_ids` filter checks performed, distances computed) and for tracking index maintenance activities (e.g., number of blocks flushed to shadow, number of blocks merged from shadow to main graph, duration of merge operations, cache hit/miss rates).
   - Expose these metrics through standard DuckDB system tables or table-producing functions to enable straightforward monitoring, analysis, and performance diagnostics by users and administrators, as suggested in "2025-05-06 04-Diskann algorithm mitigations.md," Section 3.5.
6. **Refined `Serialize`/`Deserialize` Procedures for Enhanced Robustness:**
   - Ensure that all state information absolutely necessary for robust crash recovery and comprehensive validation of index integrity is correctly and completely checkpointed during the `Serialize` phase.
   - Ensure that this information is accurately restored and meticulously validated during the `Deserialize` phase to guarantee a consistent and correct index state upon loading.

**Implementation Tasks:**

- Development of Task V (Transactional Consistency - focusing on full and reliable integration with DuckDB's Transaction Manager for obtaining commit epochs and querying transaction status).
- Development of Task V (Rollback Handling - including rigorous testing of various abort scenarios and their impact on index state).
- Development of Task V (Concurrency Control - systematic evaluation of existing locking and, if indicated by performance benchmarking under concurrent workloads, design and implementation of finer-grained locking mechanisms).
- Enhancement of Task VI (DuckDB API Integration - specifically, refining the `Serialize` and `Deserialize` methods to support robust validation procedures and reliable recovery from inconsistencies).
- Implementation of a comprehensive suite of observability metrics, covering both search performance and index maintenance activities.
- Execution of extensive stress testing protocols and carefully designed simulated crash recovery scenarios to validate the system's resilience and data integrity.

**Testing Focus:**

- Verification of the correctness and strictness of MVCC implementation under conditions of high transactional concurrency.
- Validation of the atomicity and durability of all index operations, with particular emphasis on behavior across various simulated system crash scenarios.
- Thorough assessment of index consistency and data integrity following diverse crash and recovery sequences.
- Evaluation of system performance, stability, and scalability under sustained high concurrent load (mixed read/write workloads).
- Confirmation of the accuracy, completeness, and utility of the implemented observability metrics for diagnostics and performance tuning.

### Post-MVP Enhancements (Slated for Future Iterations)

Upon the successful completion and rigorous validation of the foundational MVPs, subsequent development efforts may be directed towards the following advanced features and significant optimizations, aimed at further enhancing performance, scalability, and usability:

- **Parallel Graph Build Implementation:** The introduction of a parallel graph build process, potentially leveraging techniques such as prefix-doubling (as outlined in Section VII of the comprehensive implementation plan), to significantly accelerate the initial index creation phase for very large datasets, thereby reducing ingestion latency.
- **Development of Adaptive `L_search` Mechanisms:** Engineering mechanisms that allow the search breadth parameter (`L_search`) to be dynamically adjusted based on runtime query characteristics (e.g., filter selectivity, query vector location relative to data distribution) or real-time performance feedback, aiming for an optimal balance between recall and latency for diverse queries.
- **Implementation of Advanced Compaction with Full Graph Repair:** The development of a more sophisticated compaction strategy that enables the physical deletion of nodes from `graph.lmd` and potentially involves restructuring the graph to improve its density, connectivity, and overall search performance. This would address long-term internal fragmentation.
- **Introduction of a Lazy Neighbor Sweep Mechanism:** The implementation of a background process designed to proactively identify and clean stale links (i.e., references to tombstoned nodes) from the neighbor lists of live nodes, thereby incrementally improving graph quality and search efficiency over time without requiring a full compaction.
- **Creation of an Advanced, Potentially Self-Tuning, Cost Model:** The development of a more precise and adaptive cost model for integration with DuckDB's query optimizer. Such a model might incorporate machine learning techniques or leverage historical query performance data to refine its estimates and improve query plan selection.
- **Implementation of Full Multi-Version Concurrency Control (MVCC) for `graph.lmd` Block Versions:** Should the simpler epoch/versioning scheme implemented in the MVPs prove insufficient for highly complex concurrent workloads or advanced transactional requirements (e.g., time-travel queries on the index), a more comprehensive MVCC system for managing multiple historical versions of `NodeBlock`s within `graph.lmd` could be investigated.
- **Development of an Efficient Pathway for In-Place Updates of Vector Data:** For scenarios where vector embeddings change frequently but the underlying entity (and its `node_id`) remains the same, an optimized update pathway that modifies the vector data within an existing `NodeBlock` (while still adhering to copy-on-write principles via the shadow architecture) could offer performance benefits over a delete-then-insert approach.

This phased MVP approach is strategically designed to facilitate incremental delivery and continuous, rigorous testing, thereby mitigating development risks and enabling a responsive feedback loop throughout the project's lifecycle. Each successive MVP will build upon a validated and functional core, progressively incorporating additional features and enhancing system robustness, ultimately culminating in a comprehensive, performant, and production-ready LM-DiskANN extension for the DuckDB ecosystem.