# DuckdbLM-DiskANN Extension v1: Architecture and Design MVP

## Background and Evolution from LM-DiskANN Paper to DuckDB Extension

**LM-DiskANN Overview:** *LM-DiskANN* (Low-Memory DiskANN) is a dynamic graph-based Approximate Nearest Neighbor (ANN) index designed to reside primarily on disk while using minimal RAM. It builds on the DiskANN approach, which stores a navigable small-world graph of vectors for high recall and performance, but eliminates DiskANN’s need to keep a compressed copy of all vectors in memory. 

In LM-DiskANN, each graph node’s disk *block* contains *complete routing information* – the node’s vector and a list of neighbors (with compressed neighbor vectors) – enabling searches with only on-demand disk reads and negligible memory overhead. The original paper demonstrated that this design achieves recall-latency performance comparable to in-memory indexes while drastically reducing memory footprint.

**From Paper to DuckDB Integration:**  The extension must handle **transactional updates, concurrency, and recovery** in a DBMS context – areas beyond the paper’s scope. Initial integration attempts (using DuckDB’s in-memory allocators) revealed issues like **memory-pinning and poor eviction**, prompting a redesign to a *“shadow table” architecture* that fully embraces disk storage and DuckDB’s WAL/transaction mechanisms. Over successive design iterations (beta1 and beta2), the architecture evolved to ensure **ACID properties**, minimal write amplification, and synergy with DuckDB’s optimizer for filtered queries. The final design, described below, marries the LM-DiskANN graph structure with DuckDB’s MVCC model and storage framework, resulting in a robust, modular extension that supports billion-scale vector data with dynamic inserts/deletes.

## Architectural Overview

### LM-Diskann Index Overview

The LM-DiskANN variant of the algorithm we implemented is designed for low memory consumption by keeping only a small part of the vector index in memory while remaining efficient in search operations.  LM-DiskANN introduces a redundancy in the on-disk node format by storing both the graph and compressed neighbors on disk. So essentially, the LM-DiskANN algorithm trades memory usage for an increase in storage.

![Node layout](https://turso.tech/images/blog/approximate-nearest-neighbor-search-with-diskann-in-libsql/node-layout.png)

- **Node Chunk Structure**: Each node in the index is stored on disk as a fixed-size block (e.g., 4KB). This "node chunk" contains the node's identifier, its full-precision (uncompressed) vector, and the identifiers of its neighbors along with their *compressed* vectors.
- **Local Decisions**: This structure allows LM-DiskANN to make local decisions during the beam search. When a node chunk is read from disk, it has all the necessary information (neighbor IDs and their compressed vectors) to determine the next set of promising candidates without requiring additional I/O for those neighbors' compressed data. The compressed vectors of neighbors within the current node's chunk are used to calculate distances to the query, guiding the search path
- **Graph Traversal**: The search typically starts from one or more entry points (random nodes or pre-defined start nodes). It iteratively explores neighbors, using the beam search strategy to select which nodes' full data (including their neighbor lists) to fetch from disk.
- **Re-ranking**: After the graph traversal phase guided by compressed vectors, a re-ranking step is performed. The full-precision vectors of the final candidate nodes (those visited or identified as promising during the search) are fetched (if not already cached) and used to compute precise distances to the query, yielding the final top-k results.

### Key Design Goals

* **Dynamic Updates & Low Write Amplification:** Instead of expensive full-file rewrites on each update, use an incremental log-structured approach. Only modified blocks are written to the shadow delta store and periodically merged to the main file, making write cost proportional to changes (\~O(dirty\_blocks)) rather than total index size.
* **Correctness under Concurrency:** Avoid any shared in-place mutations that could break consistency. All updates produce new block copies (copy-on-write) and use versioning and transactional metadata to ensure readers see a consistent snapshot. This prevents issues like partial writes or stale pointers from “merged pages” or shared references.
* **Transactional Integrity (MVCC):** Integrate with DuckDB’s transaction system to handle **row\_id reuse and rollbacks** safely. Each node carries a commit timestamp/epoch to enforce visibility rules similar to MVCC, so an index never returns results from uncommitted or rolled-back transactions. A **commit epoch** and origin transaction ID in each node’s header allow the system to *discard nodes newer than the querying transaction’s snapshot*, preventing “ghost reads” when DuckDB reuses row identifiers after rollbacks. Aborted entries are later garbage-collected.
* **High Read Performance:** Support fast ANN search with minimal memory. Leverage a multi-tier caching and lookup hierarchy for node blocks (in-memory LRU cache, OS page cache, shadow table, then SSD). Each block read yields the vector and compressed neighbors for local distance computations, reducing random I/O during search.
* **Scalability to 10M–1B+ vectors:** The design should efficiently handle today’s target of \~100 million vectors per index, and remain forward-compatible with multi-billion scales at very cheap cost. This implies 64-bit identifiers and careful disk space management (free lists, compaction) to avoid overflow or fragmentation. The block ID space is monotonic (IDs 0..N-1) and never reused, enabling neighbor references to fit in 32 bits for large N (up to \~4B nodes) and simplifying pointer updates.
* **Isolation and Maintainability:** By isolating index storage in its own files, the design contains potential failures. The main DuckDB database is unaffected by index file corruption or crashes in the extension. Backup/restore of an index is as simple as copying its folder, and dropping an index just deletes that folder.

### Deficiencies in a buffer managed duckdb index

####  Main roadblocks

1. **"Index lacks access to WAL events"** This fundamental deficiency signifies that alterations to the index's state, particularly concerning auxiliary data structures such as lookup tables or metadata, are not safeguarded by a Write-Ahead Log that is synchronized with database transactions. The consequences include a loss of atomicity and durability for index operations, creating risks of data loss during system failures, inconsistent behavior during rollbacks, and non-atomic commits between the database and the index.
2. **"BufferManager fails to deallocate memory for index or ART data"** A "beta1" design's misuse of the database's buffer manager can prevent the deallocation of memory occupied by index structures (such as node data or ART-based lookup structures). This leads to persistent memory pinning, culminating in memory bloat and potential system resource starvation. An example cited was the `FixedSizeAllocator` pinning all index blocks.

####  Related Problems in the beta1 design

A rudimentary "beta1" index design, particularly one characterized by a monolithic structure and lacking robust mechanisms for the management of auxiliary data, typically encounters the following substantial challenges:

1. **Durability Deficits and Post-Crash Inconsistencies:** Index modifications may not exhibit atomicity in conjunction with database transactions.  Indexes have no events in WAL.  This creates a vulnerability wherein data loss or index desynchronization can occur should a system failure transpire subsequent to a database commit but prior to the durable persistence of index alterations.
2. **Inadequate Management of Transaction Rollbacks:** In the absence of stringent transaction integration, the rollback of a database transaction may result in the persistence of anomalous data artifacts (e.g., a vector subject to rollback remaining discoverable via search) within the index.
3. **Elevated Write Amplification and Suboptimal Update Performance:** Elementary designs can necessitate the rewriting of substantial index segments for minor updates, such as the addition of a node. This approach is demonstrably inefficient for dynamic workloads characterized by frequent modifications.
4. **Propagation of Stale Data:** Within graph-based indexes, alterations to a vector or the graph topology can lead to referencing nodes retaining outdated information, consequently diminishing search accuracy. The immediate and universal propagation of such changes is often prohibitively expensive.
5. **Complex and Resource-Intensive Rebuilds and Maintenance Operations:** Simpler designs tend to exhibit performance degradation over time, frequently mandating comprehensive and costly index reconstruction. The management of data fragmentation and the reclamation of storage space also present considerable difficulties. 

## Design of LM-DiskANN as a DuckDB extension

**Index-as-Folder Design:** Each LM-DiskANN index is self-contained in its own directory on disk, isolating its files from the main database. The index folder contains a shadow duckdb table and graph files.  From the user’s perspective, the LM-DiskANN index behaves like a native index in DuckDB. The typical usage is:

```sql
-- Create an index on the vector column using LM_DiskANN
CREATE INDEX myindex ON table_name USING LM_DiskANN(vector_column) 
WITH (dimensions=128, block_size=8192, distance_metric='Cosine', R=64, ... index_location =  'path/to/myindex_data_folder', 
-- The path will be relative to current index [mvp1], or with prefix (https/s3/etc...)
);
```

This statement causes DuckDB to call into the extension’s index creation routine.

### Directory Layout

This directory contains the following components:

- **Index Metadata file (`metadata.lmd`):** Stores global metadata such as vector dimension, graph parameters (e.g., max degree `R`, `alpha` for GSNG), distance metric, and the calculated `NodeBlock` size (e.g., 8192 bytes). These are typically fixed at index creation.

- **Primary Graph Files (`graph.lmd`):** A binary file storing the ANN graph’s nodes in fixed-size `NodeBlock` units. Each `NodeBlock` holds one vector, its adjacency list (neighbor IDs and their compressed vectors), and associated metadata. While logically growing, this file can have blocks updated (via copy-on-write to a new location or the delta store, with the old block eventually freed) or internal free space reused, so it's not strictly append-only after initial build. This is the primary on-disk representation of the graph structure.

- **Index Store Database (`diskann_store.duckdb`):** A DuckDB *secondary database* (with its own Write-Ahead Log - WAL) that maintains all auxiliary metadata for the LM-DiskANN index. Utilizing a DuckDB database file leverages its robust recovery mechanisms and transactional integrity for managing the index's evolving state. Key tables within this store include:

  - **Shadow Delta Table (`lmd_delta_blocks`):** A WAL-backed table that records new or modified `NodeBlocks` as opaque blobs. These are changes (e.g., new nodes, updated neighbor lists from healing) that have been committed but not yet merged into the `graph.lmd` file. It serves as a durable journal of pending updates, ensuring that recent changes are queryable and recoverable.

  - **RowID↔NodeID Mapping Table (`lmd_lookup`):** Maps each DuckDB base table `row_id` to its corresponding `node_id` in the ANN graph. This indirection is crucial for translating search results back to user-level tuples and for correctly handling scenarios where DuckDB might reuse `row_id`s after deletions.

  - **Tombstone Nodes Table (`lmd_tombstoned_nodes`):** Tracks `node_id`s that have been logically deleted from the graph. Each entry typically includes the `node_id` and a deletion epoch/timestamp. Nodes in this table are ignored by searches (for transactions after the deletion epoch) and are candidates for eventual reclamation by the sweeper process.  The `NodeBlocks` themselves have tombstone flag.

  - **Free Nodes Table (`lmd_free_nodes`):** Maintains a list of `node_id`s (and their corresponding block locations in `graph.lmd`) that have been fully processed after deletion and whose space is now available for reuse by new node insertions. This helps in managing disk space and mitigating fragmentation within `graph.lmd`.

  - **Sweep List (`lmd_sweep_list`):** A queue or list of `node_id`s that require processing by the background sweeper. Entries are added here to manage various stages of node lifecycle and maintenance:

    - When a node is initially tombstoned, its `node_id` might be added to signal the sweeper to begin processing its deletion (which includes initiating edge healing for its neighbors and eventually reclaiming its block).
    - When a node's block is updated (e.g., due to an insertion, its own vector changing, or its neighbor list being modified by an edge healing operation) and the new version is in `lmd_delta_blocks`, its `node_id` is added to signal the sweeper to merge this updated block into `graph.lmd`.
    - May also track nodes identified for other maintenance, like block recompaction due to high internal fragmentation (many deleted neighbor entries).

  - **Heal List (`lmd_heal_list`):** Tracks `node_id`s that have been identified by the sweeper or other maintenance processes as needing proactive graph connectivity adjustments or neighborhood optimization, not necessarily tied to an immediate deletion of one of their direct neighbors. This list supports tasks like:

    - Periodic graph quality checks where the sweeper identifies nodes whose neighborhoods have become suboptimal over time due to cumulative changes in the graph.

    - Rewiring or optimizing connections for nodes that might be critical for graph connectivity but whose current linkage could be improved based on broader graph heuristics.

      This list is distinct from the immediate, reactive healing of a deleted node's direct neighbors (which is typically handled when the sweeper processes a tombstone from the `lmd_sweep_list`).

#### Advantages

**Isolation and Maintainability:** By confining index data to a dedicated folder, the extension keeps failures localized. A crash or corruption in the index files does not affect the main DuckDB database. Backup and restore of an index is as simple as copying its folder, and dropping the index means deleting that folder. The main DuckDB database stores only a reference to the index location and perhaps minimal info (like the index name and config in the catalog), keeping the index largely decoupled.

**High-Level Operation:** When the index is in use, queries first consult in-memory caches, then the `diskann_store.duckdb` tables, and finally `graph.lmd` on SSD, to retrieve NodeBlocks. Updates (inserts or deletes) are applied by writing new NodeBlocks to the shadow table (`__lmd_blocks`) within the store DB, under transaction protection. A background process later *merges* these changes into `graph.lmd` in bulk. This design provides **transactional semantics** (via the store DB’s WAL and DuckDB’s transactions) and **high I/O performance** by batching writes.

## Node Block Layout and Graph Structure

### Node Block Layout

Each vector indexed by LM-DiskANN corresponds to a `NodeBlock` in the `graph.lmd` file. All `NodeBlock`s are of a fixed size (e.g., 8192 bytes), determined at index creation. This fixed size is crucial for direct offset-based access (`offset = node_id * BLOCK_SIZE`), efficient memory mapping, and zero-copy serialization (e.g., using Cista).

**Contents of a `NodeBlock`:**

Each block is a self-contained unit representing a graph node, its vector, its connections (forward links with compressed vectors for searching), and its referrers (backlinks).

1. **Header (e.g., ~48-56 bytes, `alignas(64)` for the whole block):**
   - `node_id` (uint64): Unique identifier for the node. Also typically determines its primary block position in `graph.lmd` when not using a free list slot. Assigned once and ideally never reused to simplify consistency.
   - `row_id` (int64): The DuckDB `row_id` of the base table tuple this vector corresponds to. Essential for mapping search results back to user data and for potential index rebuilds or verification.
   - `creation_epoch` (int64): A monotonically increasing timestamp or sequence number assigned when the transaction that created/last modified this block version committed. This is fundamental for MVCC visibility.
   - `flags` (uint8): A bitfield for various states:
     - Bit 0: `IS_TOMBSTONED` (1 if logically deleted, 0 otherwise).
     - Bit 1: `IS_OVERFLOW` (1 if backlinks for this node exist in `lmd_node_overflow`, 0 otherwise). This acts as the discriminator for the backlink union.
     - (Other bits reserved for future use, e.g., compaction status, pin status).
   - `num_forward_links` (uint16): Current count of active forward links stored in this block.
   - `checksum` (uint32/uint64): Checksum (e.g., xxHash32/64) of the block's content (excluding the checksum field itself) for detecting corruption.
2. **Vector Data:**
   - `vector` (float[DIMENSION], `alignas(32)`): The full-precision, uncompressed feature vector for this node. `DIMENSION` is fixed at index creation.
3. **Forward Links (Neighbors for Search):**
   - Capacity: `R_f` (e.g., 70, fixed at index creation).
   - `forward_link_node_ids` (uint32[R_f], `alignas(32)`): Array of `node_id`s of the neighbors this node points to. Unused slots are marked (e.g., `INVALID_NODE_ID` or 0 if 0 is not a valid `node_id`).
   - `forward_link_compressed_vectors` (uint8[R_f][CompressedVectorSize], `alignas(32)`): Array of compressed vector representations (e.g., using ternary quantization, PQ codes) for each corresponding forward neighbor. `CompressedVectorSize` depends on `DIMENSION` and the quantization scheme.
4. **Backlink Storage Area (Union for Inline vs. Overflow Pointer):**
   - This area is fixed in size, determined by the space needed for the inline backlink array.
   - The `flags.HAS_OVERFLOW_BACKLINKS` bit determines how this area is interpreted.
   - **If `flags.HAS_OVERFLOW_BACKLINKS == 0`:**
     - `inline_backlink_node_ids` (uint32[R_b], `alignas(32)`): Array storing `node_id`s of nodes that have this node in their forward links.
       - Capacity: `R_b`. Per your request, we'll set `R_b = 2 * R_f`.
       - Unused slots are marked. `num_inline_backlinks` tracks the count.
   - **If `flags.overflow == 1`:**
     - `overflow_reference` (uint64): This field occupies the first few bytes of the backlink storage area. It stores a reference (e.g., the `node_id` itself to be used as a key in `lmd_node_overflow`, table.
5. **Padding:**
   - Any remaining bytes to ensure the `NodeBlock` fills its allocated fixed size (e.g., 8192 bytes).

**Graph Structure and Identifier Management:** The overall graph structure (navigable small-world) and `node_id` management (64-bit, monotonic, no reuse) remain as previously described. The key change is the explicit, symmetric storage of backlinks within the node block itself, up to `R_b` capacity, with a defined overflow strategy.
#### MVCC Field in Node Header: `creation_epoch`

**`creation_epoch` (timestamp_t):** *(Note: `timestamp_t` is DuckDB's internal representation for timestamps. In your C++ struct for the Node Header, you'd use `duckdb::timestamp_t`).*

- **Purpose**: This field stores the timestamp representing the start time of the DuckDB transaction that created this `NodeBlock` version. This timestamp is obtained from DuckDB's active `MetaTransaction` and serves as an ordered epoch for Multi-Version Concurrency Control (MVCC) within the index.

- **Source**: When an LM-DiskANN operation (insert/update) occurs within a DuckDB transaction, the extension code will retrieve the current `MetaTransaction` using the active `ClientContext`. The `creation_epoch` for the new `NodeBlock` version is then set to `MetaTransaction::Get(context).start_timestamp`.

  - Example C++ (conceptual, within your extension's write path):

    ```
    cpp
    // Assuming 'context' is your available ClientContext&
    duckdb::MetaTransaction &meta_txn = duckdb::MetaTransaction::Get(context);
    duckdb::timestamp_t current_txn_start_time = meta_txn.start_timestamp; 
    // Or, using the convenience getter:
    // duckdb::timestamp_t current_txn_start_time = meta_txn.GetCurrentTransactionStartTimestamp();
    
    // When creating/updating a NodeBlock:
    my_node_block_header.creation_epoch = current_txn_start_time; 
    // ... then write the block to lmd_delta_blocks ...
    ```

- **Utility (Fundamental for MVCC)**:

  1. **Snapshot Isolation / Visibility**: Each active reader transaction `T_reader` operates with its own start timestamp (let's call it `reader_start_epoch`, obtained similarly: `MetaTransaction::Get(reader_context).start_timestamp`). `T_reader` can "see" `NodeBlock` versions where:
     - `NodeBlock.creation_epoch <= reader_start_epoch`
     - **AND** the transaction that started at `NodeBlock.creation_epoch` (and had the corresponding `global_transaction_id`) has successfully committed.
     - The "committed" status is handled by DuckDB's core transaction lifecycle. Data from `lmd_delta_blocks` associated with an aborted transaction will not be consolidated into the main index structure or will be eligible for cleanup. Readers querying the consolidated index inherently see only committed data.
  2. **Preventing Dirty Reads**: Transactions do not read data from other uncommitted transactions because uncommitted blocks (in `lmd_delta_blocks`) are not yet visible or consolidated based on their `creation_epoch`'s commit status.
  3. **Preventing Non-Repeatable Reads**: Ensures that if `T_reader` (with `reader_start_epoch`) reads a node, and another transaction `T_writer_new` later updates and commits that node (creating a new version with `NodeBlock_new.creation_epoch` where `NodeBlock_new.creation_epoch` is the start time of `T_writer_new`, and this start time is naturally after or concurrent but committed later than `T_reader`'s view), `T_reader` will still see the old version if `NodeBlock_new.creation_epoch > reader_start_epoch` or if `T_writer_new` is not yet committed from `T_reader`'s perspective.
  4. **Safe Tombstone Processing & Space Reclamation**: The sweeper process uses `creation_epoch` values. A tombstoned node (marked with its deletion's `creation_epoch`) can be reclaimed only if its `creation_epoch` corresponds to a committed transaction and is older than the `start_timestamp` of any currently active reader transaction (e.g., older than `DuckTransactionManager::LowestActiveStart()`). The `DuckTransactionManager::LowestActiveStart()` method seems particularly relevant here.
  5. **Rollback Handling**: If a transaction writing new `NodeBlock` versions aborts, those blocks (identifiable by their `creation_epoch` which matches the aborted transaction's `start_timestamp`) are effectively ignored. They will not be merged from `lmd_delta_blocks`.



### Overflow Management [Not MVP]

To maintain a strictly fixed `NodeBlock` size while accommodating nodes with a high number of incoming links (backlinks), an overflow mechanism is employed.  *This is only if needed.*

- **Triggering Overflow:** When a new backlink needs to be added to a node `N`, and its `inline_backlink_node_ids` array (of size `R_b`) is already full:

  1. The `flags.HAS_OVERFLOW_BACKLINKS` bit for node `N` is set to 1.
  2. All existing `R_b` backlinks from `N.inline_backlink_node_ids` are moved (inserted) into the `lmd_backlink_overflow` table in `diskann_store.duckdb`, associated with `N.node_id`.
  3. The new incoming backlink is also inserted directly into `lmd_backlink_overflow` for `N.node_id`.
  4. The `N.overflow_reference` field in `N.NodeBlock` is set to point to these overflowed entries (e.g., by storing `N.node_id`, which then serves as the query key for the overflow table).
  5. `N.num_inline_backlinks` is set to 0.
  6. Node `N`'s block is marked dirty and written to `lmd_delta_blocks`. `N.node_id` is added to `lmd_sweep_list` for merging.

- **Accessing Backlinks:** When the system needs the complete list of backlinks for a node `N`:

  1. Read `N.NodeBlock`.
  2. Check `N.flags.HAS_OVERFLOW_BACKLINKS`.
  3. If 0, all backlinks are in `N.inline_backlink_node_ids` (up to `N.num_inline_backlinks`).
  4. If 1, all backlinks must be retrieved by querying the `lmd_backlink_overflow` table using `N.overflow_reference` (e.g., `SELECT referrer_node_id FROM lmd_backlink_overflow WHERE target_node_id = N.node_id`).

- **Repatriation (Moving Backlinks from Overflow to Inline):**

  - The sweeper process can periodically attempt to move backlinks from `lmd_backlink_overflow` back into a node's inline storage.
  - **Trigger:** This can happen if, due to other nodes being deleted (which were referrers), the *actual* number of backlinks for node `N` (as tracked in `lmd_backlink_overflow`) drops below `R_b`.
  - **Process:**
    1. If `N.flags.HAS_OVERFLOW_BACKLINKS == 1` and the count of its backlinks in `lmd_backlink_overflow` is now `< R_b`:
    2. Read all backlinks for `N` from `lmd_backlink_overflow`.
    3. Write these backlinks into `N.inline_backlink_node_ids`.
    4. Update `N.num_inline_backlinks`.
    5. Set `N.flags.HAS_OVERFLOW_BACKLINKS = 0`.
    6. Clear the `N.overflow_reference` field (or set to a null/invalid value).
    7. Delete the corresponding entries from `lmd_backlink_overflow`.
    8. Mark `N.NodeBlock` dirty and schedule for merge.
  - This helps keep the `lmd_backlink_overflow` table smaller and restores faster inline access for nodes whose popularity wanes.

- **Cista and Union:** The backlink storage area can be implemented in C++ using a `union`

  ```
  // Within the NodeBlock struct
  union BacklinkArea {
      alignas(32) uint32_t inline_backlink_node_ids[R_B_PARAM]; // R_B_PARAM = 2 * R_F_PARAM
      uint32_t overflow_reference; // Uses the first 4 bytes if HAS_OVERFLOW_BACKLINKS is set
      // Potentially other members if different overflow strategies are needed later
  } backlinks;
  ```

  The `flags.HAS_OVERFLOW_BACKLINKS` bit acts as the external discriminator for this union. This structure ensures the `NodeBlock` remains fixed-size and Cista can serialize/deserialize it directly via `reinterpret_cast` or memory mapping.



## Storage Strategy: Shadow Delta Tables, Merges, and State Management

The primary graph is located in `graph.lmd` with the metadata in `metadata.lmd`.  All other components reside in the `diskann_store.duckdb` secondary database for each index:

To efficiently manage dynamic updates and ensure data integrity with high performance, LM-DiskANN employs a **Shadow Delta + Merge** strategy. This log-structured approach minimizes write amplification and integrates with DuckDB’s transactions. Instead of costly in-place modifications to the primary graph file (`graph.lmd`), changes are first recorded in a WAL-protected secondary store. A background process then periodically merges these accumulated changes into `graph.lmd` in optimized batches.

### Shadow Delta Table

**Shadow Delta Table (`lmd_delta_blocks`):** This table in `diskann_store.duckdb` is the initial persistence point for new or updated `NodeBlock`s, storing each as a BLOB keyed by its `block_id` (`node_id`). An illustrative schema is:

```
CREATE TABLE lmd_delta_blocks (
    block_id      BIGINT PRIMARY KEY,    -- Corresponds to node_id, unique.
    data          BLOB NOT NULL,         -- Serialized NodeBlock data (fixed size).
    version       BIGINT NOT NULL,       -- Version number of this NodeBlock.
    commit_epoch  BIGINT NOT NULL,       -- Commit epoch/timestamp of this block version.
    is_tombstone  BOOLEAN NOT NULL,      -- Flag indicating if this block represents a deleted node.
    checksum      BIGINT NOT NULL        -- Checksum for data integrity verification.
) WITHOUT ROWID;
```

Entries are the **latest committed versions** of `NodeBlock`s not yet merged into `graph.lmd`, acting as an upsert journal. WAL protection for `diskann_store.duckdb` ensures durability. Queries check `lmd_delta_blocks` first for the newest `NodeBlock`, effectively overriding older versions in `graph.lmd`.

### Lookup tables

**RowID ↔ NodeID Mapping Table (`lmd_lookup`):** Located in `diskann_store.duckdb`, `lmd_lookup` maps base table `row_id`s to internal `node_id`s. Insertions add mappings, and deletions remove them, all transactionally and consistent with the base table. A secondary index on `node_id` aids reverse lookups. This leverages DuckDB's recovery; `NodeBlock`s also store their `row_id` for redundancy. This system, combined with commit epochs, correctly handles `row_id` reuse, preventing stale data.

### Maintenace tables

**Tombstoned Nodes Table (`lmd_tombstoned_nodes`):** Deleted vector `node_id`s are recorded in `lmd_tombstoned_nodes` (in `diskann_store.duckdb`) with a `deletion_epoch`, providing a reliable log of deletions. This table ensures searches skip deleted nodes (especially older `graph.lmd` versions without the flag) and helps the merge process identify garbage blocks for reclamation, adding their slots to the free list. The `deletion_epoch` can track commit times for potential advanced features.

**Free List Management and Space Reuse (`lmd_free_nodes`):** Vacant `NodeBlock` slots in `graph.lmd` from deletions are managed by a free list (e.g., `lmd_free_nodes` table or tracked in `lmd_metadata`). During merges, slots of tombstoned nodes are added to this list. New `NodeBlock`s reuse these free slots before appending, mitigating fragmentation and controlling file size. The strategy is conservative: existing blocks are not moved to fill internal holes (compaction is deferred), but tail trimming of `graph.lmd` is possible if trailing blocks are free. Periodic merges ensure efficient space reuse.

**Durability and Recovery:** Modifications to `diskann_store.duckdb` tables (`lmd_delta_blocks`, `lmd_lookup`, etc.) are WAL-protected, providing standard DuckDB durability and atomicity. A committed vector insertion, for example, includes base table changes, `lmd_lookup` mapping, and `lmd_delta_blocks` data in the WAL. Crashes trigger WAL replay for `diskann_store.duckdb`, restoring consistency. The `graph.lmd` file is modified only by a crash-tolerant merge process (e.g., using temporary files and fsync before metadata updates). This two-tiered approach ensures that after recovery, shadow tables and metadata reflect committed but unmerged updates, allowing merge re-initiation or use of shadow data as truth until the next merge.

## Transactional Consistency and MVCC Integration

LM-DiskANN maintains DuckDB’s ACID properties and snapshot isolation by embedding MVCC metadata (like `creation_epoch`) in each `NodeBlock` and coordinating with DuckDB’s transaction manager. This ensures queries see a consistent index state per their transaction's snapshot, preventing phantom reads during concurrent modifications.

***\*Creation\** Epochs and Visibility:** Each `NodeBlock`'s `creation_epoch` (a durable visibility marker from DuckDB) is compared against a query's snapshot epoch. Blocks with `creation_epoch` greater than the query's snapshot are ignored, preventing reads of "future" data. Uncommitted blocks lack a globally visible `creation_epoch` and are invisible to other transactions.

**Transaction ID and Abort Handling:** Only data from committed transactions becomes durable. The flush logic writing to `lmd_delta_blocks` verifies transaction status with DuckDB; aborted transaction data is discarded, preventing "ghost" blocks. Recovery and startup checks reconcile `lmd_lookup` mappings with `NodeBlock` data, removing orphaned mappings from incomplete insertions to ensure consistency.

**Tombstones and Snapshot Isolation:** Logical deletions are transactional. A deleted `node_id` is recorded in `lmd_tombstoned_nodes` with a `deletion_epoch`. Visibility depends on the query's snapshot epoch relative to this `deletion_epoch`: older snapshots see the node as live, newer ones see it as deleted. The `is_tombstone` flag in `NodeBlock`s and the `lmd_tombstoned_nodes` table ensure correct visibility across snapshots. Committed deletions are invisible to new queries.

**Maintaining RowID Consistency:** To handle DuckDB `row_id` reuse, `lmd_lookup` mappings are transactional. Aborted transaction mappings are removed, allowing safe `row_id` reuse. If a reused `row_id` gets a new vector, its old node is already tombstoned; a new `NodeBlock` with a fresh `node_id` is created and mapped. Epoch checks prevent stale data from appearing.

**Overall Index Consistency:** An index-wide epoch/version (e.g., `merge_sequence_number`) in `lmd_metadata` tracks significant committed changes like merges. On load, this is compared with DuckDB's catalog information. Mismatches can trigger validation, refresh, or mark the index unusable (requiring rebuild) to ensure full consistency or prevent use, avoiding erroneous results.
