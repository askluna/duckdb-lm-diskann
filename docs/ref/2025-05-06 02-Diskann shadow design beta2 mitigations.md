# LM-DiskANN Shadow Architecture beta2– Implementation & Mitigation Specification

## Shadow System Design Overview

**Folder-Per-Index Architecture:** Each LM-DiskANN index lives in its own directory with two key components:

- **Primary Graph File (`graph.lmd`)** – An append-only file of fixed-size *node blocks* (e.g., 8KB each) storing vectors and graph links. This is the main on-disk index storing each node (vector) and its neighbors.
- **Index Store Database (`diskann_store.duckdb`)** – A single DuckDB database file acting as the central repository for all other index-related data. It is WAL-backed and contains several key tables:
  - **Shadow Delta Table (`__lmd_blocks`)**: Stores new or updated node blocks transactionally before they are merged into `graph.lmd`. This ensures durability and atomicity for recent updates.
  - **Lookup Mapping Table (`lmd_lookup`)**: Contains the mapping from base table **row_id** to index **node_id**. This indirection layer allows translating ANN search results (internal node_ids) back to user row identifiers and is maintained with full ACID guarantees.
  - **Metadata Table (`index_metadata`)**: Holds index metadata like vector dimensions, block size, node count, free list head, merge sequence number, etc. Storing metadata in a table allows for transactional updates and robust integrity checks.

**Node Block Structure:** Each fixed-size node block in `graph.lmd` is self-contained, storing one vector and its neighbor list (with neighbor IDs and often their compressed vectors). The block includes a header (with the node's unique ID, size, version, and transactional metadata) followed by the vector's data and a list of neighbor entries. By storing neighbors' information within the block, graph traversal during ANN search can occur with minimal random I/O – all data needed to evaluate a node and its immediate neighbors is loaded in one block read. This slightly larger storage footprint is a trade-off to achieve **low RAM usage** and **high locality** in disk reads.

**Transactional Metadata (in Node Blocks):** To integrate with DuckDB's MVCC, each node block (within `graph.lmd`) carries a **commit epoch** or timestamp and an origin transaction ID. This metadata ensures that the index observes the same visibility rules as the main database. If a transaction inserts or updates a vector, its node block is tagged with the creating transaction's ID and remains invisible to other transactions until commit. Upon commit, the block gets a commit epoch (e.g., the global commit sequence number). Readers use this to **ignore uncommitted or rolled-back nodes**: any node with a commit epoch greater than the reader's snapshot (or a special epoch indicating invalidation) is skipped. This prevents **"ghost reads"** where an index search might otherwise return a vector that was added and rolled back, or a stale entry if a row_id was reused after deletion. Aborted nodes (commit epoch not set due to rollback) are later garbage-collected from the `__lmd_blocks` table within `diskann_store.duckdb`.

**Durability and WAL (for Auxiliary Data):** By using a single DuckDB database (`diskann_store.duckdb`) for the shadow delta table, lookup mapping table, and metadata table, the system leverages robust WAL (Write-Ahead Logging) and recovery mechanisms for all these components collectively. Any insertion or update to these tables is first logged in the DuckDB WAL for `diskann_store.duckdb`, which is flushed to disk on transaction commit. This means recent index updates and metadata changes have the **same durability guarantees as standard DuckDB transactions** – if the system crashes, `diskann_store.duckdb` recovers by replaying its WAL, ensuring no committed update or metadata change is lost. The design avoids directly writing to `graph.lmd` except during controlled merge operations, thereby treating the `__lmd_blocks` table as an append-only **update journal** that can be safely replayed or merged.

**Coordination of Components:** On each base table modification, the extension coordinates updates across `graph.lmd` and the tables within `diskann_store.duckdb` to keep them in sync:

- When a new row with a vector is inserted into the base table, the extension allocates a new **node_id** and inserts a mapping (row_id -> node_id) into the `lmd_lookup` table (within `diskann_store.duckdb`) as part of the same user transaction. This ensures that if the transaction rolls back, the mapping is rolled back too (no "dangling" mappings). The actual vector and its neighbors are added as a new NodeBlock in memory and queued for flushing to the `__lmd_blocks` table.
- When an existing row is deleted, the extension marks the corresponding node as deleted (tombstoned) – e.g., setting a tombstone flag or commit epoch indicating deletion – and removes its mapping from the `lmd_lookup` table (within `diskann_store.duckdb`) as part of the delete transaction. Any neighbor links referring to this node are updated (dropped) in memory and flushed to the `__lmd_blocks` table so that searches won't consider the deleted vector.
- **External ANN Index File (`graph.lmd`) Updates:** Direct writes to the main index file happen only during **merge (compaction)** operations, which occur in the background. The merge process consolidates the buffered changes from the `__lmd_blocks` table (in `diskann_store.duckdb`) into the `graph.lmd` file (see the **Flush/Merge Lifecycle** below). This two-phase approach (delta then merge) isolates rapid updates from the heavy I/O of writing the large graph file. It dramatically **reduces write amplification**, since only changed blocks are rewritten to disk instead of the entire index.
- **Metadata Durability:** Changes in high-level metadata (e.g., increase in node_count, updates to free list, or a new merge "epoch") are recorded as transactional updates to the `index_metadata` table within `diskann_store.duckdb`. This leverages DuckDB's atomicity and durability for metadata, eliminating the need for complex file-level atomic operations (like write-temp-rename) that would be required for a separate metadata file. The system maintains a recoverable and consistent state for all metadata.

**Read Path (Lookup & Search):** A read (ANN query) goes through an indirection and caching sequence to ensure it sees the latest committed data:

1. **RowID to NodeID:** If the ANN query originates from a table scan (e.g., a "find nearest neighbors for row X"), the `lmd_lookup` table in `diskann_store.duckdb` is used to translate the row_id to the internal node_id. For KNN searches that return many vectors, after obtaining internal node_ids from the graph traversal, this table (with an index on node_id) translates those back to row_ids for the final output.
2. **Cache → Shadow Table → Base Graph File:** Given a node_id to fetch (either as a search entry point or during graph traversal of neighbors), the system first checks an **in-memory LRU cache** for the node's block. If present and valid, it contains the latest version. If not in cache, the system queries the `__lmd_blocks` table in `diskann_store.duckdb`. Because this shadow table contains the latest version of any recently modified block, a query like "SELECT block_data FROM __lmd_blocks WHERE block_id = ?" will return the newest data if that node was updated but not yet merged. Only if the node is absent in the shadow table does the system fall back to reading from the `graph.lmd` file on SSD. This lookup order (cache → shadow table → base file) guarantees that the **newest committed version** of each node is read. It also means that recent updates (which reside in the shadow table) are accessible with low latency via DuckDB's B-Tree index rather than scanning the large `graph.lmd` file.

**High-Level Write Path:** All index modifications use **copy-on-write** and indirection to preserve consistency. Rather than editing any data in place, inserting or updating a vector creates a new NodeBlock in memory (with a new version), leaving existing data untouched for concurrent readers. These new blocks are staged in the `__lmd_blocks` table (within `diskann_store.duckdb`) before eventually replacing the old ones in `graph.lmd`. This indirection (via the shadow table) allows **concurrent transactions** and readers to proceed without locking the large file; readers either get the old block (if they started before the change committed) or the new block from the shadow table (if they start after). Furthermore, the use of DuckDB's transactional store for deltas ensures that partial failures (crashes in the middle of writes) do not expose torn or half-applied updates – the `__lmd_blocks` table either has the new block or it doesn't, thanks to atomic WAL commits for `diskann_store.duckdb`.

## Issues in Prior Architecture and Shadow Design Mitigations

The shadow architecture was conceived in response to several shortcomings and failure modes identified in earlier designs. Key issues such as durability gaps, stale data reads, and complex index rebuild processes were raised in internal reviews. Below, we summarize each issue and describe how the new shadow system mitigates or resolves it:

### 1. Durability Gaps and Crash Recovery Inconsistencies

**Issue:** In prior approaches, updates to the ANN index were not guaranteed durable at transaction commit, leading to windows where a transaction could commit in the main database but the corresponding index update might be lost on crash. For example, a user inserts a new vector: the main DB commit succeeds (row is persistent, and mapping might be written), but the actual ANN node could be buffered in memory or a non-durable log. If a crash occurs just after commit, the index would start up missing that vector or with a broken mapping reference. This is a serious consistency problem: the base table and index could disagree on whether a vector exists. Another facet was handling **transaction aborts** – previously there was no robust mechanism to ensure that index changes from rolled-back transactions were discarded, potentially leaving *ghost nodes* that never got a matching commit.

**Shadow Mitigation:** The new design implements multiple strategies to eliminate these durability gaps:

- **Immediate Flush on Commit:** To narrow the commit-to-durability window as much as possible, the extension triggers an immediate flush of new node blocks to the `__lmd_blocks` table in `diskann_store.duckdb` at the moment a transaction commits. Using DuckDB's hooks (e.g., a `Transaction::RegisterCommitHook` or similar mechanism), the flush daemon is signaled to write out all dirty blocks from that transaction as soon as the main commit is done. This means that by the time the user sees a commit succeed, the index update is already in the WAL of `diskann_store.duckdb` (or will be within a few milliseconds), dramatically reducing the window where a crash can cause loss. The flush operation batches writes and uses `INSERT OR REPLACE` into the `__lmd_blocks` table to ensure idempotency (only the latest version of each block is kept). The WAL commit on `diskann_store.duckdb` makes the new blocks durable on disk, ensuring that **committed transactions have their index entries safely stored**.
- **Unified Commit Epoch & Visibility Filtering:** Every NodeBlock's commit epoch ties it to a particular commit of the main database. On recovery, this helps reconcile state across components. For instance, if a crash occurred after the main DB commit but before the flush completed, the new row_id->node_id mapping might exist in the `lmd_lookup` table (in `diskann_store.duckdb`) but the NodeBlock wasn't flushed. Upon restart, the commit epoch on that NodeBlock will be missing or set to an "invalid" marker (since it never made it to durable storage). The system can detect this and resolve the inconsistency. A likely strategy is: during index opening, scan the `lmd_lookup` table and ensure that for each mapping, either the node is present in `graph.lmd` or in the `__lmd_blocks` table. If a mapping exists with no corresponding node data, the index can **tombstone or remove that mapping** (effectively dropping the index entry for safety, perhaps with a warning) or attempt a recovery rebuild for that vector. Because the mapping and data are always added together during normal operation, any discrepancy on recovery signals a crash mid-operation, which the system addresses by either re-applying the shadow log or cleaning up the orphan mapping. In practice, the immediate flush on commit and WAL replay for `diskann_store.duckdb` cover most scenarios, making such orphaned mappings unlikely except in a narrow crash window.
- **Flush Barrier for Aborts:** To handle rolled-back transactions (so that no changes from an aborted transaction become durable), the flush daemon implements a **commit check** before writing any NodeBlock to the `__lmd_blocks` table in `diskann_store.duckdb`. Each dirty entry carries the originating transaction's ID, and the flush thread verifies the transaction status via DuckDB's transaction manager API (if available) or an epoch system. Only entries from committed transactions are flushed; aborted ones are skipped. This ensures no "ghost" NodeBlocks from rolled-back transactions ever reach the disk. Additionally, the memory for uncommitted blocks can be allocated in a per-transaction arena that is freed on abort, so any in-memory state is also purged if a transaction fails. This two-layer approach (in-memory segregation and flush-time verification) closes the gap where previously an extension might not know about aborts.
- **Idempotent Merge and Recovery:** The merge (compaction) process – which moves data from the `__lmd_blocks` table in `diskann_store.duckdb` into `graph.lmd` – is designed to be **crash-tolerant and idempotent**. If a crash occurs during merge (e.g., after some blocks have been written to the main file but before the shadow entries are cleared or metadata updated), recovery will simply find that the `__lmd_blocks` table still contains those entries (since the final step of clearing them didn’t commit). The merge can then be safely retried. Writing the same NodeBlock to the base file again is harmless because it’s the same data (or a no-op if it was already written). To support this, the merge procedure likely involves writing all pending blocks to `graph.lmd`, fsyncing, then within a transaction deleting them from the `__lmd_blocks` table and updating the `index_metadata` table (e.g., raising a merge sequence number) in `diskann_store.duckdb` **after** the file is durably updated. Only when the transaction committing these changes in `diskann_store.duckdb` succeeds do we consider the merge complete. If a crash interrupts the process at any point, either the `__lmd_blocks` table still has the blocks (so we can redo the writes), or if the transaction on `diskann_store.duckdb` committed, the metadata and shadow state are consistent. The design relies on the atomicity within `diskann_store.duckdb` to ensure metadata and shadow block states are updated together.
- **Atomic Metadata Updates:** With metadata (node count, merge sequence, etc.) stored in the `index_metadata` table within `diskann_store.duckdb`, its atomicity is guaranteed by DuckDB's transactional mechanisms. This is a significant improvement over managing atomicity for a separate metadata file (e.g., `graph.lmd.meta`) using complex write-temp-and-rename protocols. All critical metadata changes are now part of DuckDB transactions, ensuring that they are either fully committed or fully rolled back, maintaining consistency even in the event of a crash. For any metadata that might still reside outside `diskann_store.duckdb` (e.g., a magic number or version for the `graph.lmd` file itself, if absolutely necessary), standard write-temp-and-rename with fsyncs would be used, but the primary index operational metadata benefits from database-level atomicity.

Together, these measures ensure **strong durability**: once a transaction commits, either its index updates will persist, or the system will detect and handle the discrepancy on recovery, never producing incorrect search results. The WAL for `diskann_store.duckdb` and the commit protocol guarantee that **no partial index state is ever visible** to users: it’s all-or-nothing. This closes the durability gap present in earlier designs.

### 2. Stale Data Reads (Stale Neighbor or Vector Data)

**Issue:** In graph-based ANN, a common challenge for dynamic indexes is **stale neighbor data**. Each NodeBlock stores not just the node’s own vector, but also compressed copies of its neighbors’ vectors for fast distance calculations. If a vector **v** is updated (or a new vector is inserted), the neighbor blocks that contain v’s data may become stale. In the prior architecture (without a robust update strategy), this could mean that search traversals use outdated vector information, degrading accuracy over time. For example, if we change a vector’s embedding, or even insert a new node without updating neighbors, some blocks might still have old data until a full rebuild is done. The earlier design required either costly immediate updates to all affected blocks or risked returning slightly incorrect distances.

**Shadow Mitigation:** The shadow system addresses stale data through **eager update propagation for local neighbors and eventual consistency for wider changes**, combined with versioning:

- **Eager Neighbor Updates on Insert/Delete:** Whenever a new node is inserted, the algorithm selects some existing nodes to connect with (the new node’s neighbors). The design ensures that these neighbor blocks are immediately updated in memory to include the new node (or to remove a node on deletion) and marked dirty. Both the new node’s block and all *affected neighbor blocks* are flushed to the `__lmd_blocks` table in `diskann_store.duckdb` together at commit. This way, any search after the transaction will find that the neighbors’ blocks in the shadow table have been updated – they will include the new neighbor ID and the compressed vector for it. This **copy-on-write propagation** prevents stale reads in the most common case (topology changes due to inserts/deletes). The flush batching logic even deduplicates by block_id, so if the same neighbor is updated multiple times in quick succession, only the latest version is kept. By always writing the latest version per block to the shadow table, we ensure queries never see an outdated neighbor list or missing link that should have been there.
- **Marking Version and Lazy Recompression:** Each NodeBlock carries a version counter or timestamp that is incremented on any change to its content (neighbors or itself). This version is stored alongside the block in the shadow table and base file. If a vector’s own embedding is updated (a less common operation, effectively like a new insert for that row), the system can mark all of its neighbors as needing recomputation. The design could choose **not** to cascade the recomputation of compressed vectors immediately (to avoid excessive write amplification), but instead mark those neighbor blocks in metadata (or simply rely on the version timestamps). At query time, using the commit epoch and version, the search logic can decide to trust the stored neighbor distances or recompute on the fly if it detects a discrepancy. However, a more straightforward approach is taken during **merge**: when the background merge runs, it can **refresh stale neighbor vectors**. Since merge already reads each dirty block from the `__lmd_blocks` table (which includes all recently changed nodes), it has an opportunity to recalc any derived data. For example, if node X’s vector was updated, when writing out node Y (a neighbor of X) the merge process can fetch X’s new vector and recompute the compressed form to store in Y’s block before writing it to `graph.lmd`. This ensures that by the time changes make it into the base file, all neighbor references are consistent and up-to-date. The review notes recommended clarifying this strategy, and the implementation will incorporate an **eventual consistency** approach: neighbor compressed vectors are updated *either at insert time or by the next merge*. There is thus a bounded window during which a neighbor’s cached vector might be slightly stale, but in practice, because we flush neighbors eagerly on structural changes, the only staleness can come from *value* changes of a vector.
- **Controlled Rebuild of Graph Structure:** The design acknowledges that maintaining graph optimality under many updates is complex. It incorporates **graph rebuild or optimization steps** in a controlled way. For instance, after a large number of insertions, a maintenance operation might recompute nearest neighbors for certain nodes (to ensure search quality). The **shadow architecture makes such rebuilds incremental**: one can insert updated neighbor lists into the `__lmd_blocks` table as if they were normal updates, then merge. This avoids needing to rebuild the entire index from scratch. Essentially, even a “graph quality optimization” is just treated as another series of updates to NodeBlocks.
- **Testing for Staleness:** For safety, each neighbor entry in a NodeBlock could store a reference to the neighbor’s version or epoch. If during a search we access neighbor data and find a mismatch (neighbor’s current epoch != epoch of the cached vector), the system can detect it and potentially ignore the stale info. However, given the above approach, this may not be necessary unless live vector value updates are frequent.

**Trade-off:** The chosen mitigation strategy balances complexity and performance: it avoids a full cascade of updates on every vector change (which would be very expensive for high-degree nodes) by sometimes deferring the refresh to the merge or a maintenance phase. This eventual consistency is acceptable because the commit epoch mechanism will *never allow completely wrong results*: an entirely uncommitted or deleted node will be skipped. The only staleness is in distance computations, which might be slightly off until the next recompute. The design team deemed this acceptable given that ANN search is anyway approximate. Still, to minimize impact, critical changes (like neighbor list inclusion/exclusion) are done eagerly, and only the finer detail (vector compression updates) are deferred. Over time, periodic merges and possible **VACUUM/optimize operations** will clean up any remaining staleness.

### 3. Complexity of Index Rebuilds and Merges

**Issue:** Earlier designs for dynamic ANN indexing struggled with expensive index rebuilds and maintenance operations. If updates accumulated, one might have to completely rebuild the graph index (recompute all neighbor links) or write out a new copy of the index file, incurring huge I/O costs. A specific concern was internal fragmentation of the `graph.lmd` file over long-term use – deletions and updates could leave “holes” that either bloated the file or required complex compaction. The prior approach did not clearly define how to *incrementally* maintain the index structure, leading to either potential degradation or very costly periodic rebuilds.

**Shadow Mitigation:** The LM-DiskANN shadow architecture is explicitly designed to simplify maintenance by breaking it into **incremental, well-defined operations** rather than monolithic rebuilds:

- **Incremental Merge (Compaction):** Instead of rebuilding the whole index after a series of inserts/updates, the system continuously merges in changes from the `__lmd_blocks` table in `diskann_store.duckdb`. The merge operation can be tuned to run when the shadow log grows beyond a threshold (e.g., when the `__lmd_blocks` table in `diskann_store.duckdb` has X MB of data or Y% of index nodes are updated). Because each merge only needs to process the blocks that changed since the last merge, its cost is proportional to the number of updates, not the total index size. For example, if only 5% of nodes changed, merge reads those from the shadow table and writes 5% of the blocks in `graph.lmd`. This is far cheaper than rewriting 100% of the index. The design even measured a prototype scenario: ~8GB index merging in 1 million updated blocks in ~38s on NVMe, showing scalability. The merge can also be parallelized or done in batches if needed (processing blocks in chunks, multiple threads writing different ranges, etc.).
- **Free List and In-Place Updates:** The `graph.lmd` file uses fixed-size slots for nodes, allowing **in-place overwrite** when merging. A node’s position in the file is determined by its node_id (e.g., node_id * block_size offset). This means we don’t have to relocate unchanged data during merge – we only overwrite the slots that have new versions. Deleted nodes simply free their slot (not reused immediately, to avoid node_id reuse issues). A free list is kept in the `index_metadata` table (within `diskann_store.duckdb`) for eventually reusing or compacting space. This approach eliminates the need to shift large swaths of data; merging an update is O(1) to seek and write one block. Fragmentation is handled by the free list and a possible **VACUUM INDEX** command that can compact space if needed. The vacuum process could, for instance, move the last block into a hole and truncate the file, updating the lookup table and metadata accordingly. However, to start, the simpler strategy is to only reclaim space at the end (tail trimming) and defer internal hole compaction, because internal moves require updating neighbor references if node_ids change. The design suggests possibly avoiding node_id reuse entirely to simplify this (deleted IDs remain tombstoned), and only vacuum out whole trailing ranges of blocks that are free.
- **Graph Quality Maintenance:** Full index rebuilds are largely avoided. Instead, **partial rebuilds** or **graph refinements** are done in place. For example, the insert algorithm uses a “prune” heuristic to maintain graph quality locally. Over time, if the graph’s quality degrades, one can run an offline process (or background thread) that takes batches of nodes and re-evaluates their nearest neighbors (this is akin to running additional insert operations in a batch). Those changes enter the `__lmd_blocks` table as updates to neighbor lists. Thus, improving the graph is just more delta updates – applied transactionally and merged – not a full rebuild from scratch. The worst-case scenario (if the graph becomes very suboptimal or if major param changes) might involve rebuilding, but that can be done by bulk inserting all nodes into a fresh index directory. The normal path is to **never require a cold rebuild for routine operation**.
- **Mitigating Merge Bottlenecks:** For extremely large indices (billions of vectors), even merge could become heavy if not managed (since it touches many GB of data). The design provides options like **tiered merges** or limiting the scope of merges. For instance, it could merge in waves (merge 10% of the `__lmd_blocks` table at a time) or maintain multiple delta levels (recent updates in memory, mid-term in the `__lmd_blocks` table, older in base) to avoid any single huge compaction. Also, merge can be performed while queries run: since readers always check the shadow table first, they can continue to see updates that haven’t merged yet. During merge, to avoid conflicts, the process might acquire a short exclusive lock per block (to prevent flush thread from writing the same block). But it doesn’t need a full read lock on the whole index – queries can proceed, skipping blocks currently being written (or using the shadow version until the moment of switch). Proper locking (see Concurrency below) ensures no visible inconsistency. The result is that merges can happen **online** with minimal query downtime, unlike a full rebuild which would require reloading the index.
- **Simplified Backup/Restore and Rebuild:** Because each index is self-contained in one directory (containing `graph.lmd` and `diskann_store.duckdb`), operational tasks are simplified. Backing up an index is as easy as copying its directory (ensuring the copy happens after a merge or while flush is quiescent, or by also copying the WAL file for `diskann_store.duckdb`). In worst-case failure (e.g., both `graph.lmd` and `diskann_store.duckdb` get corrupted beyond WAL recovery), one can reconstruct the index by scanning the base table: the extension could rebuild by inserting all vectors anew (effectively reindexing). This is obviously expensive, but it’s a last resort. More commonly, if `diskann_store.duckdb` is corrupted but `graph.lmd` is intact, the `lmd_lookup` and `index_metadata` tables might be partially reconstructible by scanning all NodeBlocks in `graph.lmd` (if each NodeBlock stores its row_id and other necessary info for metadata reconstruction). In summary, the shadow architecture turns rebuild into a manageable maintenance operation rather than a routine requirement.

### 4. Other Notable Issues and Mitigations (Concurrency, Consistency, Scale)

*(In addition to the three primary issues above, internal reviews highlighted concurrency complexity and certain edge-case conditions. We briefly note how the implementation handles these:)*

- **Concurrency & Deadlocks:** With multiple threads (user threads inserting/querying, background flush thread, background merge thread), careful locking is required. The implementation will use **fine-grained locks** on NodeBlocks (or shards of nodes) to avoid global locks. For example, each NodeBlock has a mutex; inserts lock the new node and its neighbors’ blocks while linking, flush locks blocks it’s writing, etc. A global `merge_mutex` might serialize the merge against other writers, but readers mostly use atomic epoch checks instead of heavy locks. The plan is to define a clear lock acquisition order (e.g., always lock lower node_id first or use try-lock and back off to prevent cycles). Additionally, the extension initialization uses a global latch to avoid deadlocks when multiple indexes are attached simultaneously (ensuring one attaches fully before another begins). These measures mitigate the risk of deadlocks and race conditions in the complex multi-threaded environment.
- **Lookup Table Scaling:** The `lmd_lookup` table in `diskann_store.duckdb` could grow very large (billions of rows for 10B-vector index). DuckDB’s indexing (B-tree on the PRIMARY KEY row_id and perhaps a secondary index on node_id) is expected to handle this scale, but performance will be monitored. If necessary, we could partition the mapping or use an alternative key-value store, but for now the simplicity of DuckDB’s single-file DB and its robust indexing is advantageous. Batch operations on the lookup table (like bulk insert during index build) can be accelerated by using DuckDB’s append APIs with larger chunks.
- **Memory and Cache Management:** The shadow architecture offloads persistent storage to disk, but memory efficiency is still critical. The LRU cache for NodeBlocks will be tuned to only occupy a fraction of RAM (maybe caching the hottest 1-10% of vectors, depending on workload). If the cache is full, least-recently-used blocks are evicted – provided they are not dirty. Dirty blocks (unflushed updates) won’t be evicted until flushed. We also rely on the OS page cache for the `graph.lmd` file reads; large sequential reads (e.g., during merge or range scan) will populate the page cache which benefits later random reads. The extension avoids pinning huge amounts of memory via DuckDB’s buffer manager (an issue in a previous approach was that using DuckDB’s `FixedSizeAllocator` pinned all index blocks in memory) – instead, we manage reading/writing ourselves and only cache selectively. This gives us full control to ensure memory usage stays bounded no matter how large the index grows.

With these mitigations in place, the shadow architecture resolves the major pain points of earlier designs. **Durability gaps** are closed by WAL and flush-on-commit; **stale reads** are minimized by eager neighbor updates and MVCC rules; **rebuild complexity** is tamed by incremental merges and an update-friendly file format. We now turn to the detailed implementation plan reflecting this design.

## Implementation Plan and Technical Specification

In this section, we outline the concrete implementation approach for the shadow architecture in C++ (as a DuckDB index extension). This includes the key data structures, lifecycle of operations (insert, update, delete, flush, merge, recovery), integration points with DuckDB’s extension API, file format details, and concurrency control. The focus is on **how to build** the system described above, with guidance suitable for engineers implementing it.

### Data Structures and Components

#### Node Block and Related Structs (C++)

At the heart is the `NodeBlock` structure, representing a vector and its neighbors either in memory or as stored in the index file. Pseudocode for the struct might look like:

```
struct NodeBlock {
    uint64_t node_id;         // Unique identifier (also determines file offset)
    row_t row_id;             // The DuckDB row_id this vector corresponds to (for optional redundancy)
    uint64_t version;         // Version counter for this block (increment on each update)
    transaction_t origin_txn; // ID of creating transaction (for uncommitted blocks)
    uint64_t commit_epoch;    // Commit timestamp/epoch; 0 or MAX indicates not yet committed or deleted
    bool tombstone;           // True if this node is deleted (pending removal)
    VectorData vector;        // The raw vector data (e.g., float array of dimension D)
    std::vector<Neighbor> neighbors; // List of neighbor entries (each has neighbor node_id and compressed vector)
    // ... possibly checksums or padding to ensure fixed size ...
};
```

Each `Neighbor` entry would contain something like:

```
struct Neighbor {
    uint64_t neighbor_id;
    CompressedVector comp_vector; // Compressed representation of neighbor's vector (e.g., PQ or ternary encoding)
};
```

The `CompressedVector` is specific to the compression used (ternary, PQ, etc.) and stores just enough info to compute distances quickly. The total size of a `NodeBlock` is fixed (e.g., 8192 bytes) so that `graph.lmd` can be indexed by block number.

We maintain a few key in-memory maps:

- **Mapping Table Cache:** While the `lmd_lookup` table in `diskann_store.duckdb` persists the row_id -> node_id mapping, the extension may also keep a hash map or ART (adaptive radix tree) in memory for quick lookups of node_id by row_id (to avoid a DB query each time). This in-memory map is populated at index load (scanning the lookup table or reading metadata from `diskann_store.duckdb`). It must be kept in sync with inserts and deletes (updates).

- **Dirty Ring Buffer:** A lock-free (or mutex-protected) ring/circular buffer of dirty block entries to be flushed by the background thread. We define a `DirtyEntry` struct capturing a pointer or reference to a dirty NodeBlock, along with its block_id (node_id) and the transaction that dirtied it:

  ```
  struct DirtyEntry {
      uint64_t block_id;
      NodeBlock *block_ptr;  // pointer to the in-memory NodeBlock (e.g., in cache)
      transaction_t origin_txn;
  };
  ```

  Writers will push `DirtyEntry` into this buffer whenever they modify a block (on commit). The flush thread will pop batches of these for writing to the `__lmd_blocks` table.

- **LRU Cache:** Implemented perhaps with an LRU list + hash map from node_id to NodeBlock*. It stores NodeBlocks that are either clean (same as disk) or dirty (newer than disk, not yet merged). A dirty NodeBlock in cache must also exist in the `__lmd_blocks` table once flushed. We mark dirty vs clean for each.

- **Index Store DB Connection:** We maintain a persistent DuckDB `Connection` to the `diskann_store.duckdb` database file for each index. On index initialization, we `Connection::Open` this file. For performance, we prepare commonly used statements for operations on tables within `diskann_store.duckdb`:

  - `insert_shadow_stmt` for inserting or replacing into `__lmd_blocks`.

  - `select_shadow_stmt` to fetch a block by block_id from `__lmd_blocks`.

  - `insert_lookup_stmt` for `lmd_lookup` table.

  - `delete_lookup_stmt` for `lmd_lookup` table.

  - `select_lookup_stmt` for `lmd_lookup` table.

  - `update_metadata_stmt` for `index_metadata` table.

  - select_metadata_stmt for index_metadata table.

    These can be PreparedStatement objects reused by flush and read operations to avoid SQL parse overhead.

- **Table Schemas in `diskann_store.duckdb`:**

  - **Shadow Delta Table (`__lmd_blocks`):**

    ```
    CREATE TABLE __lmd_blocks (
        block_id BIGINT PRIMARY KEY,
        block_data BLOB,      -- raw bytes of NodeBlock
        version BIGINT,
        commit_epoch BIGINT,  -- Storing commit epoch here can optimize certain recovery/visibility checks
        tombstone BOOLEAN DEFAULT FALSE, -- Explicit tombstone marker
        checksum BIGINT        -- (optional) checksum for integrity
    );
    ```

  - **Lookup Mapping Table (`lmd_lookup`):**

    ```
    CREATE TABLE lmd_lookup (
        row_id BIGINT PRIMARY KEY, -- Assuming row_id from base table is unique
        node_id BIGINT UNIQUE NOT NULL -- node_id in graph.lmd
    );
    CREATE INDEX IF NOT EXISTS lmd_lookup_node_id_idx ON lmd_lookup (node_id);
    ```

  - **Metadata Table (`index_metadata`):**

    ```
    CREATE TABLE index_metadata (
        key VARCHAR PRIMARY KEY,
        value_text VARCHAR,
        value_int BIGINT,
        value_double DOUBLE
    );
    -- Example rows:
    -- ('num_nodes', NULL, 1000000, NULL)
    -- ('block_size', NULL, 8192, NULL)
    -- ('dimension', NULL, 128, NULL)
    -- ('merge_sequence', NULL, 5, NULL)
    -- ('free_list_head', NULL, 0, NULL)
    -- ('metric_type', 'L2', NULL, NULL)
    ```

    This key-value schema for metadata is flexible. Alternatively, a fixed-column schema could be used if metadata fields are stable.

#### Index Class and API Integration

We implement a DuckDB index extension, for example as `class LMDiskANNIndex : public duckdb::Index`. DuckDB’s `BoundIndex` interface requires us to implement methods for index operations. Key methods we’ll implement and their roles:

- **`Initialize(IndexInfo&)` or constructor:** Opens/creates the index directory and files. This will:
  - Create the directory if not exists, using DuckDB’s FileSystem API.
  - Open or create `diskann_store.duckdb` via DuckDB’s API (`DuckDB db(path_to_diskann_store_duckdb)` and `Connection`).
  - If creating anew, execute the `CREATE TABLE` statements for `__lmd_blocks`, `lmd_lookup`, and `index_metadata` within `diskann_store.duckdb`. Initialize default metadata.
  - If opening existing, read metadata from the `index_metadata` table.
  - Load the mapping table cache (or at least initialize our in-memory row_id->node_id map).
  - Start the flush thread (see below).
  - Possibly schedule a merge if the `__lmd_blocks` table is non-empty from last run (crash recovery scenario).
- **`Append(DataChunk &entries, Vector &row_identifiers)`:** Called when new vectors are inserted. For each new vector:
  - Allocate a new node_id (e.g., from `index_metadata.num_nodes`, then increment).
  - Create a NodeBlock in memory. Compute its neighbors.
  - Mark the new NodeBlock and affected neighbor blocks as dirty.
  - Insert the `row_id -> node_id` mapping into the `lmd_lookup` table (within `diskann_store.duckdb`) as part of the user’s transaction (see **Transactional Coupling**).
  - Update `index_metadata` (e.g., `num_nodes`) transactionally.
  - Append `DirtyEntry` for new/modified blocks to the ring buffer, tagged with the current transaction’s ID.
- **`Delete(DataChunk &row_identifiers)`:** Called when rows are deleted. For each row_id:
  - Look up node_id from the `lmd_lookup` table.
  - Mark NodeBlock as tombstone.
  - Remove mapping from `lmd_lookup` table (transactionally).
  - Update neighbors, mark them dirty.
  - Update `index_metadata` (e.g., free list, tombstone count) transactionally.
  - Push dirty entries.
- **Transactional Coupling:** Critical for atomicity.
  - Attach `diskann_store.duckdb` to the main connection: `ATTACH DATABASE 'path/to/diskann_store.duckdb' AS idx_store;`.
  - Index operations (inserts/deletes in `idx_store.lmd_lookup`, updates to `idx_store.index_metadata`) are executed within the user's transaction on this attached database. DuckDB ensures atomicity.
  - Use `Commit` and `Rollback` callbacks provided by DuckDB's index API if available to manage transaction-specific state or trigger actions. The flush barrier (checking transaction status before flushing NodeBlocks to `__lmd_blocks`) remains crucial.
- **`Scan` (or `Search`)**:
  - Performs DiskANN graph traversal, fetching NodeBlocks via cache -> `__lmd_blocks` table -> `graph.lmd`.
  - Translates resulting node_ids to row_ids using the `lmd_lookup` table in `diskann_store.duckdb`.
  - Respects transaction snapshot visibility using commit epochs on NodeBlocks.
- **`Merge` / `Vacuum` (Manual):** Expose `VACUUM INDEX my_index`. Internally calls `DoMerge()`. `Vacuum()` in DuckDB's API can be implemented.
- **`CommitDrop`**: On `DROP INDEX`, clean up: shut down threads, close `diskann_store.duckdb`, delete index directory.

#### Flush Thread (Asynchronous Delta Writing)

The flush daemon writes dirty NodeBlocks to the `__lmd_blocks` table in `diskann_store.duckdb`:

- **Flush Trigger:** Periodic or signaled on transaction commit.

- **Process:** Pops batch from ring buffer, deduplicates.

- **Transaction on `diskann_store.duckdb`:**

  ```
  // Pseudocode for flush thread action
  // cx is the connection to diskann_store.duckdb
  cx.BeginTransaction();
  for (auto &entry : batch) {
      // Verify entry.origin_txn committed successfully before proceeding
      if (!TransactionManager::IsCommitted(entry.origin_txn)) continue;
  
      // Serialize entry.block_ptr to a blob
      Value block_blob = SerializeNodeBlock(entry.block_ptr);
      insert_shadow_stmt.Bind(0, (int64_t)entry.block_id);
      insert_shadow_stmt.Bind(1, block_blob);
      insert_shadow_stmt.Bind(2, (int64_t)entry.block_ptr->version);
      insert_shadow_stmt.Bind(3, (int64_t)entry.block_ptr->commit_epoch); // Store commit epoch
      insert_shadow_stmt.Bind(4, entry.block_ptr->tombstone);      // Store tombstone status
      // ... bind checksum ...
      insert_shadow_stmt.Execute(); // Uses INSERT OR REPLACE
  }
  cx.Commit();
  ```

- The commit to `diskann_store.duckdb` ensures durability via its WAL.

- **Coordination & Thread Safety:** Ensure flush happens after main transaction commits. Use appropriate locking for ring buffer and NodeBlock data access.

#### Merge Process (Compaction to Base File)

Merges data from `__lmd_blocks` into `graph.lmd`:

1. **Prepare:** Acquire `merge_mutex`. Optionally pause/drain flush.
2. **Gather Deltas:** Query `__lmd_blocks` table in `diskann_store.duckdb` for all entries.
3. **Write to `graph.lmd`:** For each entry, `pwrite()` block_data to `graph.lmd` at `block_id * BLOCK_SIZE`. Fsync `graph.lmd`.
4. **Commit Changes in `diskann_store.duckdb`:** Within a single transaction on `diskann_store.duckdb`:
   - `DELETE FROM __lmd_blocks` where blocks have been merged.
   - `UPDATE index_metadata SET merge_sequence = merge_sequence + 1, num_nodes = new_num_nodes, ...;`
   - Commit this transaction. This atomically clears processed shadow entries and updates metadata.
5. **Post-Merge:** Release `merge_mutex`. Mark relevant NodeBlocks in cache as clean.

#### Crash Recovery Sequence

- DuckDB recovers `diskann_store.duckdb` via its WAL. All tables (`__lmd_blocks`, `lmd_lookup`, `index_metadata`) are restored to a consistent state.
- Index initialization:
  - Opens `diskann_store.duckdb`.
  - Reads metadata from `index_metadata` table.
  - If `__lmd_blocks` is non-empty, it means updates are pending merge. The system can continue normal operation (queries will see these deltas) and schedule a merge later, or trigger a catch-up merge.
  - Verify consistency between `lmd_lookup`, `index_metadata`, and `graph.lmd` if necessary (e.g., during a debug/paranoid startup).
- Handle the narrow crash window (mapping committed, NodeBlock not flushed): On recovery, if a mapping in `lmd_lookup` exists for a node_id that has no corresponding data in `__lmd_blocks` or `graph.lmd` (and is beyond the last known `num_nodes` from before the crash), attempt to re-index that row from the base table or log/drop the inconsistent mapping.

#### DuckDB Extension Hooks and Lifecycle

- **Index Registration, Create Index, Serialization, Commit/Rollback Hooks, Drop:** Largely as described previously, but all interactions with auxiliary data (shadow, lookup, metadata) are now through the single `diskann_store.duckdb`.
- **Threading:** Use C++ threads or DuckDB's TaskScheduler.

### Concurrency and Edge Case Handling

(This section remains largely the same conceptually, as the change to `diskann_store.duckdb` primarily affects atomicity and file management, not the fundamental concurrency logic for NodeBlock access or graph traversal. References to specific DB files should be interpreted as tables within `diskann_store.duckdb`.)

#### Concurrent Readers vs. Writers

- Copy-On-Write for NodeBlocks.
- Fine-grained locks.
- Ring buffer coordination.
- Flush vs. Read: Handled by transactional semantics and commit epochs.
- Flush vs. Merge: Coordinated via `merge_mutex` and flush quiescence.

#### Crash During File Operations

- For `graph.lmd` itself, careful fsync ordering.
- `diskann_store.duckdb` relies on DuckDB's WAL for its integrity.
- Idempotent merge logic.

#### Partial Index Rebuild / Recovery

- If `diskann_store.duckdb` is severely corrupted, recovery might involve rebuilding it by scanning `graph.lmd` (if NodeBlocks store row_ids) and the base table.

#### Lookup Inconsistency Handling

- Transactional guarantees from `diskann_store.duckdb` (when ATTACHed) greatly reduce chances of inconsistency between base table and `lmd_lookup` / `index_metadata`.
- Recovery checks for dangling mappings or nodes.

### Example Lifecycle Flows

(Update references to reflect `diskann_store.duckdb` and its tables)

#### Insert Transaction Flow

1. Begin Transaction (DuckDB).
2. User INSERTs row.
3. `Index::Append`:
   - Create NodeBlock, find neighbors, mark blocks dirty.
   - `INSERT INTO idx_store.lmd_lookup (...)` and `UPDATE idx_store.index_metadata (...)` (via attached `diskann_store.duckdb`).
   - Push DirtyEntries to ring buffer.
4. User commit:
   - DuckDB commits main table and `idx_store` changes. Assigns commit epoch E.
   - Commit hook signals flush. Assign commit_epoch E to NodeBlocks.
   - Flush thread: Pops entries. Begins transaction on `diskann_store.duckdb`. `INSERT OR REPLACE` into `__lmd_blocks` for each committed NodeBlock. Commits.
5. Post-commit: Data durable in base table and `diskann_store.duckdb`.

#### Query (Search) Flow

1. User queries.
2. `Scan` called:
   - Traverses graph. `GetNode(node_id)`:
     - Cache miss -> Query `__lmd_blocks` table in `diskann_store.duckdb`. Hit if updated, not merged.
     - If miss in `__lmd_blocks` -> Read from `graph.lmd`.
   - Resulting node_ids -> `SELECT row_id FROM idx_store.lmd_lookup WHERE node_id IN (...)`.
3. Returns row_ids.

#### Delete Transaction Flow

1. User deletes row.
2. `Index::Delete`:
   - Lookup node_id in `idx_store.lmd_lookup`.
   - Mark NodeBlock tombstone. Update neighbors (mark dirty).
   - `DELETE FROM idx_store.lmd_lookup (...)` and `UPDATE idx_store.index_metadata (...)`.
   - Push DirtyEntries.
3. User commit:
   - DuckDB commits changes. Assign commit epoch E.
   - Flush thread: Flushes updated neighbor blocks and tombstoned block to `__lmd_blocks`.
4. Post-commit: Deletion reflected.

#### Index Merge (Compaction) Flow

1. Condition: `__lmd_blocks` table in `diskann_store.duckdb` is large.
2. `DoMerge()`:
3. Acquire `merge_mutex`. Pause/drain flush.
4. Read entries from `__lmd_blocks` table.
5. For each entry, write to `graph.lmd`. Fsync `graph.lmd`.
6. **Atomic update in `diskann_store.duckdb`**:
   - Begin transaction.
   - `DELETE FROM __lmd_blocks` for merged entries.
   - `UPDATE index_metadata SET merge_sequence = ..., num_nodes = ...`.
   - Commit transaction.
7. Release `merge_mutex`. Resume flush.
8. `__lmd_blocks` table (for merged entries) is now empty. Cache updated.

#### Concurrent Queries During Merge

(Logic remains similar, ensuring queries see consistent state via `__lmd_blocks` or `graph.lmd` based on merge progress and cache state.)

### Conclusion

This implementation plan translates the LM-DiskANN shadow architecture into a concrete design, leveraging a consolidated `diskann_store.duckdb` for managing the shadow delta table, lookup table, and crucial index metadata. This approach enhances durability and transactional consistency for these auxiliary components. The solution addresses prior shortcomings by **never updating in place**, carefully coordinating commits across the base table and the index's own data stores, and using standard database techniques (WAL, MVCC, transactional metadata updates) to guarantee correctness.

With this design, we expect to handle indexes on the order of 100M to 10B vectors on SSD with high throughput. Reads are optimized by the block caching and on-disk layout, and writes (updates) are efficient due to batching and the robust transactional framework provided by DuckDB for all non-graph data. The most complex parts – like concurrency and crash recovery – are mitigated by the refined shadow/indirection approach, as detailed in the above specification. Each potential issue raised in design reviews has a corresponding mitigation here, making the architecture resilient and practical for implementation.

**Sources:**

- LM-DiskANN Shadow Table Design Proposal
- Design Review & Risk Analysis
- Mitigation Matrix for Failure Cases