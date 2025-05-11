# LM-DiskANN Extension: Architecture and Design Evolution

## Background and Evolution from LM-DiskANN Paper to DuckDB Extension

**LM-DiskANN Overview:** *LM-DiskANN* (Low-Memory DiskANN) is a dynamic graph-based Approximate Nearest Neighbor (ANN) index designed to reside primarily on disk while using minimal RAM. It builds on the DiskANN approach, which stores a navigable small-world graph of vectors for high recall and performance, but eliminates DiskANN’s need to keep a compressed copy of all vectors in memory. 

In LM-DiskANN, each graph node’s disk *block* contains *complete routing information* – the node’s vector and a list of neighbors (with compressed neighbor vectors) – enabling searches with only on-demand disk reads and negligible memory overhead. The original paper demonstrated that this design achieves recall-latency performance comparable to in-memory indexes while drastically reducing memory footprint.

**From Paper to DuckDB Integration:**  The extension must handle **transactional updates, concurrency, and recovery** in a DBMS context – areas beyond the paper’s scope. Initial integration attempts (using DuckDB’s in-memory allocators) revealed issues like **memory-pinning and poor eviction**, prompting a redesign to a *“shadow table” architecture* that fully embraces disk storage and DuckDB’s WAL/transaction mechanisms. Over successive design iterations (v1 and v2), the architecture evolved to ensure **ACID properties**, minimal write amplification, and synergy with DuckDB’s optimizer for filtered queries. The final design, described below, marries the LM-DiskANN graph structure with DuckDB’s MVCC model and storage framework, resulting in a robust, modular extension that supports billion-scale vector data with dynamic inserts/deletes.

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
2. **"BufferManager fails to deallocate memory for index or ART data"** A "V1" design's misuse of the database's buffer manager can prevent the deallocation of memory occupied by index structures (such as node data or ART-based lookup structures). This leads to persistent memory pinning, culminating in memory bloat and potential system resource starvation. An example cited was the `FixedSizeAllocator` pinning all index blocks.

####  Related Problems in the V1 design

A rudimentary "V1" index design, particularly one characterized by a monolithic structure and lacking robust mechanisms for the management of auxiliary data, typically encounters the following substantial challenges:

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

Each vector in the index corresponds to a **NodeBlock** in the `graph.lmd` file. All NodeBlocks are equal-sized (e.g. 8192 bytes) and aligned, so the `node_id` serves as an index for direct file offset (`offset = node_id * BLOCK_SIZE`). This fixed size simplifies random access and in-place updates.

**Contents of a NodeBlock:** Each block is essentially a self-contained representation of a graph node and its neighbors. The layout is:

- **Node Header:** Includes administrative fields:
    - `node_id` (uint64) – Unique identifier for the node (also determines its block position). Assigned once and never reused.
    - `origin_txn_id` (int64) – The DuckDB transaction ID that created or last modified this node block.
    - `commit_epoch/versioning info` (int64) – A monotonically increasing commit timestamp or epoch given when the creating transaction committed. Used for MVCC visibility checks.  Or some versioning info?
    - `tombstone` (boolean) – A flag indicating the node is logically deleted. Set to true when a deletion is committed; causes searches to ignore this node.
    - `checksum` (uint64) – Checksum (e.g. xxHash64) of the block’s content for corruption detection.
- **Vector Data:** The full uncompressed feature vector for this node (e.g., an array of `float` values of length = dimension). This is used for accurate distance calculations when needed (e.g., final re-ranking).
- **Neighbor List:** An array of neighbor **node_ids** (e.g., up to `R` neighbors, where `R` is the max degree). These are the outgoing edges in the graph from this node. Neighbors are typically chosen based on nearest-neighbor criteria (the graph is often built using something like the Vamana or NSW algorithm).
- **Compressed Neighbor Vectors:** For each neighbor in the list, a compressed representation of that neighbor’s vector is stored.  Tertiary quantization is used to shrink vectors while preserving distance approximations. Storing these in the NodeBlock allows distance computations to neighbors without loading each neighbor’s full NodeBlock from disk – only the current block is needed.
- Backlink List: …. 2xr
- **Padding/Free Space:** Bytes padding the structure to the fixed block size. This can accommodate slight growth in neighbor list size without relocating the block. Large changes or overflows are handled by writing a new version of the block to the shadow store (copy-on-write).

*Figure: Conceptual NodeBlock Layout — ID & metadata, full vector, neighbor IDs, neighbor compressed vectors, then padding.* Each block has all info to evaluate that node and traverse to neighbors, enabling a **single disk I/O per node** during search. This trades extra storage for far fewer random reads: the search algorithm can fetch a node’s block and immediately have approximate distances to all its neighbors, rather than retrieving each neighbor separately.

**Graph Structure:** The nodes form a **navigable small-world graph** (like DiskANN’s graph or an HNSW graph). Each node links to ~`R` nearest neighbors, forming a highly connected graph that a greedy search can traverse to find close vectors. One node is designated as an entry point (often the one with maximum vector norm, as in DiskANN). The graph can be viewed as dynamic: insertions add a new node that connects to some existing nodes (and may cause some local neighbor list adjustments), while deletions remove a node (and ideally its references in others’ neighbor lists). The graph is kept *approximately* well-connected through local heuristics (e.g., neighbors are selected by a *prune* algorithm to maintain good recall). Periodic maintenance or smarter insertion heuristics can mitigate any degradation in graph quality over many updates.

**Identifier Management:** Node IDs are 64-bit and monotonically increasing as new nodes are added (the index can scale to billions of vectors). 