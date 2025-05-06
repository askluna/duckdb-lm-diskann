

# LM-DiskANN with Shadow Lookup Table – Technical Design & Implementation Specification

## Architecture Overview and Goals

LM-DiskANN is a disk-native graph-based ANN (Approximate Nearest Neighbor) index designed for billion-scale vector data with a low memory footprint. The proposed architecture follows a **folder-per-index, shadow lookup table design** that prioritizes incremental updates, transactional correctness, and high I/O performance. Each vector index is stored in its own directory, containing:

* **Primary graph file (`graph.lmd`)** – an append-only file of fixed-size *node blocks* (e.g. 8 KB each) storing vectors and graph links. This is the bulk SSD-resident storage for the index’s nodes and neighbor lists.
* **Shadow delta store (`shadow.duckdb`)** – a small DuckDB database acting as a WAL-backed *delta table* for modified blocks. This “shadow table” buffers changes (insertions or updates to node blocks) transactionally, instead of updating `graph.lmd` in place.
* **Lookup mapping store (`lookup.duckdb`)** – a separate DuckDB database containing a mapping table from base table **row\_id** to index **node\_id** (`shadow_db.lmd_lookup`). This mapping allows translating query results (node IDs) back to row identifiers and is maintained with full ACID properties.
* **Metadata file (`graph.lmd.meta`)** – a small file with header info (e.g. dimensions, block size, node count, free list head, etc.) used for integrity checks and fast index reopening.

**Figure: LM-DiskANN node block layout.** Each fixed-size block stores one vector and its neighbors’ info in a self-contained manner. *Each block contains a node ID and its full vector data, followed by a list of neighbor IDs and their compressed vectors, padded to a constant size.* This design (inspired by LM-DiskANN) trades extra storage for minimal memory: all neighbor information needed for graph traversal is present in the block itself, eliminating the need to keep all vectors in RAM.

### Key Design Goals

* **Dynamic Updates & Low Write Amplification:** Instead of expensive full-file rewrites on each update, use an incremental log-structured approach. Only modified blocks are written to the shadow delta store and periodically merged to the main file, making write cost proportional to changes (\~O(dirty\_blocks)) rather than total index size.
* **Correctness under Concurrency:** Avoid any shared in-place mutations that could break consistency. All updates produce new block copies (copy-on-write) and use versioning and transactional metadata to ensure readers see a consistent snapshot. This prevents issues like partial writes or stale pointers from “merged pages” or shared references.
* **Transactional Integrity (MVCC):** Integrate with DuckDB’s transaction system to handle **row\_id reuse and rollbacks** safely. Each node carries a commit timestamp/epoch to enforce visibility rules similar to MVCC, so an index never returns results from uncommitted or rolled-back transactions. A **commit epoch** and origin transaction ID in each node’s header allow the system to *discard nodes newer than the querying transaction’s snapshot*, preventing “ghost reads” when DuckDB reuses row identifiers after rollbacks. Aborted entries are later garbage-collected.
* **High Read Performance:** Support fast ANN search with minimal memory. Leverage a multi-tier caching and lookup hierarchy for node blocks (in-memory LRU cache, OS page cache, shadow table, then SSD). Each block read yields the vector and compressed neighbors for local distance computations, reducing random I/O during search.
* **Scalability to 100M–10B+ vectors:** The design should efficiently handle today’s target of \~100 million vectors per index, and remain forward-compatible with multi-billion scales. This implies 64-bit identifiers and careful disk space management (free lists, compaction) to avoid overflow or fragmentation. The block ID space is monotonic (IDs 0..N-1) and never reused, enabling neighbor references to fit in 32 bits for large N (up to \~4B nodes) and simplifying pointer updates.
* **Isolation and Maintainability:** By isolating index storage in its own files, the design contains potential failures. The main DuckDB database is unaffected by index file corruption or crashes in the extension. Backup/restore of an index is as simple as copying its folder, and dropping an index just deletes that folder.

## Data Structures and Components

### Node Block Format (`graph.lmd`)

Each node in the graph is stored in a fixed-size **`NodeBlock`** (e.g. 8192 bytes). The block layout (see figure above) includes:

* `uint32_t id` – the node’s logical ID (index into `graph.lmd`). This ID is assigned once and never changes or reuses a prior ID’s slot.
* **Vector data** – the full, uncompressed vector (e.g. float\[dim]) for this node.
* **Neighbor list** – an array of neighbor node IDs, typically 20–100 nearest neighbors. The IDs are stored as 32-bit ints (sufficient up to 4B nodes).
* **Compressed neighbor vectors** – for each neighbor in the list, a compressed representation of that neighbor’s vector (using product quantization or another compression). This allows distance computations to the neighbor without loading the neighbor’s full block. The compression ratio (e.g. 16×) drastically reduces memory needed per neighbor.
* **Padding/free space** – ensures each block is exactly the fixed size, to simplify indexed access by `offset = block_id * BLOCK_SIZE`. Padding can also allow in-place updates if a neighbor list grows slightly (though substantial changes create a new block in shadow storage).

Additionally, each block’s **internal header** contains metadata for transactional consistency and versioning (embedded in either reserved bytes or the vector/neighbor count fields):

* **`txn_id` and `commit_epoch`** – identifiers of the DuckDB transaction that created or last modified this node. For an inserted node, these are set at the time of commit. They are used to decide if a node is visible to a given query (matching DuckDB’s MVCC rules).
* **`node_version`** – a local sequence number incremented on each mutation of this node’s neighbors. This is used to pick the latest update when merging concurrent changes.
* **`checksum`** – a 64-bit checksum (e.g. xxHash64) of the block data. This helps detect partial writes or bit flips, isolating corruption to a single node block (which can then be repaired or rebuilt if needed).

**Free List:** Since block IDs are never reused for new inserts, deletion of a node leaves a “hole” in the file. A free-list of freed block slots is maintained to recycle physical space. Freed blocks can be reallocated for writing updated blocks of other IDs during merges. The `graph.lmd.meta` stores a pointer to the head of a free list (a linked list chain via unused blocks or an internal structure) and the current highest block ID, so space can be reclaimed by a background **vacuum** process.

### Shadow Delta Table (`__lmd_blocks` in `shadow.duckdb`)

All modifications to node blocks (insertions of new nodes or updates to neighbors of existing nodes) are recorded in an internal *shadow table* rather than applied in-place to `graph.lmd`. This table resides in the `shadow.duckdb` database and acts as a durable change buffer. Its schema is:

```sql
CREATE TABLE __lmd_blocks (
    block_id  BIGINT PRIMARY KEY,   -- logical node ID
    data      BLOB NOT NULL,        -- exact BLOCK_SIZE bytes (serialized NodeBlock)
    version   BIGINT NOT NULL,      -- node_version (incremented on each mutation)
    checksum  BIGINT NOT NULL       -- xxHash64 of data for integrity
) WITHOUT ROWID;  -- (so block_id is the index key for O(1) lookups)
```



* **Block ID:** The unique identifier of the node block being modified. This is the stable logical pointer used throughout the system.
* **Data:** The full serialized block content (exactly 8192 bytes, for example) containing the updated node vector/neighbors. Storing the entire block as a BLOB ensures that a reader can reconstruct the node entirely from either the base file or this delta.
* **Version:** A copy of the node’s `node_version`. If two threads attempt to update the same node concurrently, both may be buffered, but only the higher version should win when merging. This field thus allows the merge process to resolve write-write conflicts by picking the latest update.
* **Checksum:** Stored again for safety (it can be verified on merge or recovery to detect torn writes).

DuckDB’s WAL will log any insertion into this table, so writing to `__lmd_blocks` leverages the DB’s atomic commit and recovery capabilities. We create this table with `WITHOUT ROWID` so that `block_id` is the primary index, enabling efficient point queries by ID. Four **prepared statements** are pre-allocated for this table (INSERT, SELECT, range SELECT, DELETE) and kept alive to avoid repeated query planning overhead.

**Change Tracking and Buffering:** In memory, the extension maintains a **lock-free ring buffer** (or queue) of pending block modifications. Each entry in the ring buffer is a tuple `{block_id, data_ptr, checksum}` for a dirty block that needs to be persisted. Worker threads (inserts or updates) push to this ring when they mark blocks dirty. A background **flush thread** (described later) continuously pops from this ring and writes batches into the `__lmd_blocks` table.

By decoupling immediate graph updates (in memory) from on-disk persistence, we avoid holding up transactions on disk I/O. Yet, because the changes go through DuckDB’s WAL, **crash consistency is ensured** – once a transaction commits its changes to the shadow table, an OS crash cannot lose those changes. (If the system crashes *before* the extension flushes a recent commit’s vectors, we discuss recovery below.)

### RowID ↔ NodeID Lookup Table (`lookup.duckdb`)

For each base table row with a vector, we need to map its DuckDB **row identifier** (row\_t) to the internal **node\_id** in the DiskANN graph. This mapping is dynamic: insertion of a new row allocates a new node\_id; deleting a row frees its node and mapping. We maintain this mapping in a separate DuckDB database `lookup.duckdb` (to isolate it from both the main DB and the shadow delta store). The mapping table (e.g. `lmd_lookup`) schema might be:

```sql
CREATE TABLE lmd_lookup (
    row_id  BIGINT PRIMARY KEY, 
    node_id BIGINT 
);
-- with an index on node_id as well (if reverse lookup needed)
```

By keeping this in DuckDB, we leverage its indexing, WAL and recovery for mapping updates. Every insert into the base table will insert a row here (within the same transaction, ensuring no mapping is written if the user transaction rolls back). The **lookup.duckdb** can be attached on index open, and ensures ACID consistency of the pointer from row to node. In case of corruption, the mapping can be rebuilt from the graph file if needed (by scanning all node blocks for their row\_ids, since each node block could also store its originating row\_id if redundancy is desired).

**Note:** DuckDB’s `row_t` (row IDs) can be reused after VACUUM or rollback in the base table, which is a major correctness concern. The design addresses this by **transactionally coupling mapping changes with base table changes** and by the MVCC epoch on nodes (see *Transactional Consistency* below). On rollback, any inserted mapping will be undone by DuckDB, and the node inserted remains invisible (its commit\_epoch will indicate “not committed”).

### In-Memory Cache and Buffer Manager

To accelerate access, the extension uses a two-level cache for node blocks:

* An **LRU Cache** (in-process) stores a limited number of recently-used node blocks in their uncompressed form (8KB each). This avoids repeated parsing or disk reads for hot nodes. The cache key is `block_id` and the value includes the deserialized NodeBlock and metadata (dirty flag, version).
* Rely on the **OS Page Cache** (or DuckDB’s Buffer Manager) for lower-level caching of file data. Because `graph.lmd` may be memory-mapped or read via standard I/O, the OS can cache frequently read disk pages. Optionally, we could integrate with DuckDB’s buffer manager to have more explicit control over caching of the index file; however, the custom LRU at the block level already provides fine-grained eviction control at the index level.

The cache also plays a role in coordinating writes: when a node is updated, its cache entry is marked dirty and scheduled for flush. When merges occur, the cache may be updated or invalidated for those blocks.

For thread safety, each node block can be protected by a **mutex or sharded lock**. Inserts that update different neighbor lists can proceed in parallel (e.g. partition by node\_id modulo N shards to minimize contention). This concurrency design will be refined (e.g., *sharded locks for neighbor updates* is noted as a future optimization).

## Operational Flows and Algorithms

### Transactional Consistency and Visibility

Before detailing operations, it’s crucial to understand how the design ensures **correctness across DuckDB transactions**:

* **Insertions and Updates** occur within a DuckDB transaction (e.g., an `INSERT` into the base table). The LM-DiskANN extension hooks into the insertion logic so that vector index updates happen as part of the user’s SQL transaction. The new node is created in memory and given a provisional `txn_id` (the current transaction) and no commit timestamp yet. It won’t be visible to queries from other transactions until commit.
* On **Commit** of a transaction, a commit epoch (a monotonically increasing counter or the transaction’s commit ID) is assigned to any new nodes or updated blocks. The flush daemon will persist these to `shadow.duckdb` nearly immediately (within milliseconds), but even if there’s a slight delay, other transactions will see the effect logically as soon as committed because queries check the commit epoch.
* On **Rollback**, any new node or update made by that transaction must not persist or must be ignored. If the flush of that change never happened, the in-memory dirty entry can be discarded. If it did get into the shadow table (unlikely if we flush at commit, but possible in asynchronous mode), the `txn_id/commit_epoch` in the node block marks it as uncommitted. Such a node will be filtered out from search results (its commit\_epoch will be missing or marked as aborted). A periodic background job can purge shadow entries or blocks that belong to aborted transactions (since they will never get a valid commit epoch). This functions analogously to vacuuming aborted MVCC tuples in a database.

**Query-time Visibility Check:** Every node block carries `(txn_id, commit_epoch)`. When performing a search or lookup, the index will **ignore any node whose commit\_epoch is greater than the current transaction’s snapshot epoch** (or which has an invalid epoch due to rollback). This way, if DuckDB reuses a row\_id after a rollback, the stale node for the rolled-back insert will still exist in `graph.lmd`, but it has an old `txn_id/epoch` and will not be considered visible, thereby avoiding incorrect query results (no “ghost” vectors). This measure directly addresses the rowid reuse correctness issue by making the index *transaction-aware*.

### 1. Index Construction (Bulk Build)

Building a new index (for example, via `CREATE INDEX ... ON table(column) USING LM_DiskANN`) involves scanning the base table’s vector column and inserting each vector into the graph:

1. **Folder and File Setup:** Create the index directory and files. Initialize an empty `graph.lmd` (with header and metadata), a new DuckDB database for `shadow.duckdb` (set to WAL mode), and another for `lookup.duckdb`. Ensure the directory is created via DuckDB’s `FileSystem` API (to handle various OS/paths) and is empty/writable.
2. **Initial Graph Building:** If vectors are large, use a streaming build to avoid memory blowup. One approach is to sort or shuffle the input and insert vectors one by one, using a simplified heuristic (like first vector as node 0, etc.). Each insert uses the **Insertion** process (described below) – but for bulk load, we might batch inserts without flushing every single one, or disable strict transactional isolation for speed since it’s a fresh index.
3. **Finalize Metadata:** Once all vectors are inserted and flushed to the main file (we can merge in batches during build to keep shadow small), write out the final header metadata: total nodes count, dimension, etc., and ensure `graph.lmd.meta` is written and fsynced. Close the DuckDB connections for shadow and lookup (or leave open if index remains open).

### 2. Search Operation (ANN Query)

The search algorithm implements a graph traversal (e.g. a greedy *Best-First Search* as in DiskANN/Vamana) to find nearest neighbors. Given a query vector `q` and a desired number of neighbors `k`, the process is:

1. **Initialization:** Start with an entry node. Typically, a random node or a small set of randomly picked nodes is chosen as the starting point for search. (Optionally, we could store a few designated entry nodes in the metadata for deterministic behavior).
2. **Candidate List:** Maintain a min-heap or similar structure of candidate nodes to explore (size bounded by parameter L). Also keep a set of visited nodes.
3. **Traversal Loop:** Pop the closest not-yet-visited candidate node `p` from the heap. Compute the distance from `q` to `p` (we have `p`’s full vector either from cache or by reading its block). Mark `p` as visited. Then retrieve all neighbors of `p`:

   * To get neighbor info, load `p`’s NodeBlock. This may be in the LRU cache; if not, check if it’s in the shadow delta (i.e., updated but not merged) via `__lmd_blocks`; if found, use that data. Otherwise, read from `graph.lmd` on disk. These steps are abstracted by a function `ReadNodeBlock(id)` that does:

     * If `id` is in cache, return it.
     * Else `SELECT data FROM __lmd_blocks WHERE block_id = id` in `shadow.duckdb`. If found, deserialize the block data into a NodeBlock and (optionally) cache it.
     * If not in shadow, calculate file offset = `header_size + id * BLOCK_SIZE` and read that 8KB from `graph.lmd` into a buffer (possibly via pread or using the OS page cache). Deserialize to NodeBlock, cache it.
   * Verify the block’s `checksum` if present (especially for shadow data) to ensure integrity.
   * Also check the node’s `commit_epoch`: if the node is uncommitted or too new for the current snapshot, **skip its neighbors** (treat as if no neighbors – effectively the node is invisible).
   * Retrieve `p`’s neighbor list (IDs and compressed vectors). For each neighbor `n` not yet visited, compute an approximate distance using the **compressed neighbor vector** (this is much faster than full precision). Use this to decide if `n` could be a good candidate.
   * Push each new neighbor `n` into the candidate min-heap with the estimated distance. (We may also compute the exact distance for `n` using its full vector *if and when* it is popped for expansion, not for all neighbors preemptively – this defers I/O cost).
4. **Termination:** Continue until the candidate list is empty or the closest candidate in the list is already in the result set (visited). Then the visited set contains `k` approximate nearest neighbors. If needed, compute exact distances for those and select the top `k`.
5. **Result Mapping:** The final neighbor IDs (node\_ids) are looked up in the mapping table to get actual row IDs. This is done with a batched query to `lookup.duckdb` (since it has an index on node\_id or we maintain an array mapping if node\_ids are dense). The results are returned as the row references for the original table, which can then be joined to retrieve additional columns.

Throughout the search, **concurrency control** ensures that if a merge (compaction) is happening in the background, it will not disrupt reading blocks. A merging thread uses fine-grained locks per block when replacing an old block with a new version. If our search tries to read a block currently being written, it will either wait on that block’s lock or read the older version (depending on implementation of generation counters). In this design, we can implement an atomic pointer or generation number for each block in cache: when a block is updated/merged, the pointer is swapped after the write is complete, under a lock, so readers either see the old or new data atomically.

### 3. Insertion Operation (Adding a Vector)

Insertion of a new vector is integrated with DuckDB’s insert into the base table. Pseudocode for inserting a single vector `v` with row\_id `r`:

1. **Preprocessing:** The vector `v` may be normalized if using cosine similarity (the index expects normalized vectors). The insertion routine receives the vector (as a DuckDB DataChunk or similar) from the planner.
2. **Allocate Node ID:** Determine the new node’s ID. This is usually `node_id = current_node_count` if no deletions, or if a free list exists, take the next free slot (if reusing space). However, to maintain monotonic IDs, we do not reuse old IDs for new nodes – instead we append. So typically, `node_id = node_count` and increment node\_count. Reserve an 8KB block space for it in memory.
3. **Create NodeBlock:** Construct a new NodeBlock for `node_id`:

   * Set its ID, store the vector `v`.
   * **Graph linking:** Find neighbors for this new node. We perform a *neighbor search (robust prune)*: run a mini-ANN search for `v` in the current index (perhaps with a reduced parameter since index is slightly stale) to get approx `M` nearest existing nodes. This uses the same search logic as above, but limited to a subset or using some insertion heuristic (like DiskANN’s robust pruning).
   * Take the resulting neighbor list and assign it as `node_id`’s neighbors. For each neighbor found, also consider adding `node_id` into that neighbor’s neighbor list (mutual linking). This may require evicting the farthest neighbor in those lists if they are at capacity.
   * All affected neighbors (the ones that gain this new node as a neighbor) are marked **dirty** as well, since their neighbor lists changed.
   * Initialize `node_block.version = 1` (first version) and set `txn_id = current_txn, commit_epoch = 0` (to be filled on commit).
4. **Update In-Memory Structures:** Insert the new NodeBlock into the LRU cache (it’s now the freshest block). Also create an entry in the mapping table (in DuckDB’s transaction context, `INSERT INTO lookup VALUES (r, node_id)`).
5. **Schedule Persistence:** For each dirty block (the new node and any neighbors updated):

   * Compute its checksum.
   * Push `{block_id, data_ptr, checksum}` into the global ring buffer (lock-free queue) for the flush daemon. If the ring buffer length grows beyond a threshold (e.g. 4096 entries), signal the flush thread to wake up immediately.
   * Mark the cache entry as dirty (if not already).
6. **Commit Handling:** If the user transaction commits, the flush daemon will handle the dirty blocks (see below). On commit, also mark the new node’s `commit_epoch` (e.g. set it to a global increasing counter or the DuckDB commit id for that txn). The mapping insertion in `lookup.duckdb` will commit as part of DuckDB’s commit, making the row->node mapping durable. If the transaction aborts, we remove the new node from cache and ignore its ring buffer entry (the flush thread can detect that `txn_id` has aborted by checking DuckDB transaction state or simply by the absence of a commit epoch – such entries will be purged in vacuum).
7. The base table insertion returns, and from the user’s perspective, the insert is done. The index updates happen asynchronously but are buffered by the WAL for safety.

**Threading:** Note that neighbor computations (robust prune search) could be CPU intensive. In a batch insert scenario, we could parallelize vector insertions or do them one by one. Each insertion locks the neighbors it touches. The design can use fine-grained locks (e.g., lock each neighbor list when updating it) to allow concurrent inserts that affect different areas of the graph.

### 4. Deletion Operation (Removing a Vector)

Deletion is more complex in a graph-based index, but supported in a dynamic index. When a row is deleted from the base table (or the user issues a DELETE), the index must remove the corresponding node and update neighbors:

1. **Find Node:** Look up the row’s `node_id` via the lookup table. If not found, the index might already be out-of-sync (or the row had no index entry).
2. **Mark Node Removed:** Invalidate the node’s block. The node can be marked with a tombstone flag in its block header (or set its neighbor list length to 0 and commit a special “deleted” epoch). We still keep the block around until a merge cleanup.
3. **Update Neighbors:** For each neighbor `n` in the node’s list, remove the deleted node from `n`’s neighbor list. That means those neighbor blocks become dirty and will be written to shadow. (If a neighbor’s list has a vacancy, one could optionally attempt to fill it by searching for an alternative neighbor, but a full graph rebuild is not necessary – many ANN graphs handle deletions by lazy cleanup.)
4. **Free Resource:** Add the deleted node’s block ID to the free list (so its space can eventually be reused). Do **not** reuse the ID for new inserts, to avoid confusion; the ID is just retired.
5. **Persist Changes:** Like insertion, push dirty updates for each affected neighbor (and the deleted node’s block, if we choose to write a tombstone) into the flush ring buffer. Also remove the mapping from `lookup.duckdb` (this happens as part of the SQL DELETE transaction on the base table).
6. **Merge Cleanup:** During the next merge (compaction), the deleted node’s block won’t be copied to the new file (or can be skipped), effectively removing it. The free list entry is updated so that block space can be reclaimed by a file **VACUUM** operation. If the index supports a `VACUUM INDEX` command, it would move blocks to fill holes and shrink `graph.lmd` accordingly.

Throughout deletion, the commit/rollback logic should be similar: if the transaction aborts, the node remains as it was.

### 5. Flush Daemon (Writing Deltas to Shadow Table)

A dedicated **flush thread** runs for each open index, responsible for draining the ring buffer of dirty blocks and inserting them into the shadow delta table. The flush thread uses DuckDB’s **TaskScheduler** to schedule a recurring task (e.g. every 250 ms) that wakes up and performs flushes. Pseudocode for the flush daemon:

```cpp
while (!shutdown) {
    sleep_until_next_interval_or_condition(250ms);
    std::vector<DirtyEntry> batch = ring_buffer.pop_up_to(2048);
    if (batch.empty()) continue;
    // Deduplicate by block_id (keep only last update per block in batch)
    sort(batch.begin(), batch.end(), by_block_id);
    batch.erase(unique_by(block_id, keep_last), batch.end());
    
    // Prepare bulk insert into __lmd_blocks
    Connection& cx = shadow_db_connection;
    cx.BeginTransaction();
    for (auto& entry : batch) {
       // Use prepared insert: (block_id, blob_data, version, checksum)
       insert_stmt.Bind(..., entry.block_id, Value::BLOB(entry.data_ptr, BLOCK_SIZE), 
                        entry.data_ptr->version, entry.checksum);
       insert_stmt.Execute();  // or add to a vector of parameters for chunk insert
    }
    cx.Commit();
}
```

In practice, we use `INSERT OR REPLACE` into `__lmd_blocks` for each dirty block, so that the shadow table always contains the latest version per block (older versions are overwritten or superseded). We batch up to e.g. 2048 blocks per transaction to amortize overhead. Using DuckDB’s **PreparedStatement::ExecuteChunked()** allows inserting many rows with minimal overhead. The commit of this transaction flushes a single WAL page (\~128KB) to disk to cover all 2048 inserts, which is extremely fast. This way, the flush throughput is very high – measured at \~45k blocks/sec (≈350 MB/s) on a 4-core machine.

After flushing, these blocks are now safely stored in `shadow.duckdb` (and WAL-fsynced). The in-memory cache still holds them as dirty (we might mark them as clean since they’re persisted, but they are still “newer” than main file). Readers will find them via the shadow table on cache miss. The flush thread repeats, keeping the shadow table’s size in check.

**Trigger conditions:** The flush thread wakes periodically, but also flushes sooner if the ring buffer is filling up (to avoid excessive memory usage). Additionally, on **transaction commit**, we can trigger an immediate flush for that transaction’s blocks to narrow the window of lost updates in a crash. In DuckDB, if available, a `Transaction::RegisterCommitHook` could be used to signal the flush thread to run right after commit. This ensures durability as soon as possible after commit.

### 6. Merge (Compaction) of Delta into Base File

Over time, the shadow table accumulates delta entries representing updated blocks. To prevent unbounded growth and to persist changes to the main index file, a **merge (compaction)** operation runs occasionally. This can be triggered by either: (a) the shadow table size exceeding a threshold (e.g. 256 MB of deltas), or (b) during a DuckDB checkpoint event (to ensure a consistent on-disk state). The merge process does the following:

1. **Locking and Snapshot:** The merge (one per index) acquires necessary locks to prevent conflicting flushes or merges. It may still allow concurrent queries – to do so, it can lock one block at a time as it merges, rather than locking the whole index. We obtain a snapshot of all entries in `__lmd_blocks` at this time (e.g. using `SELECT * FROM __lmd_blocks ORDER BY block_id` to iterate in ID order).
2. **Iterate Deltas:** For each delta row `(block_id, data, version)`:

   * Compute the target offset in `graph.lmd`: `offset = header_size + block_id * BLOCK_SIZE`. If this block\_id is beyond the current end-of-file (i.e., a brand new node), we will **append** it to the file (growing the file). If the block\_id is within the current file range, we have two cases:

     * If the block’s space is currently free (node was deleted or moved), we can reuse that space (“space recycle”). Otherwise, we will overwrite in place. In this design, since we never reuse block IDs, an existing ID corresponds to a valid block slot – so we usually write in place to update that node’s contents. (We ensure the file was pre-sized to accommodate the highest ID, or we extend it as needed.)
   * Prepare a 8KB buffer for the write (the `data` blob itself is exactly 8KB after serialization). We might use a contiguous 4MB I/O buffer to batch writes: if many block updates are contiguous in ID, their offsets are contiguous in file, which we can coalesce into one large write. In practice, sorting by block\_id ensures we write sequentially which improves throughput by using large consolidated writes.
   * Write the block data to `graph.lmd` at the target offset. If using direct pwrite, this may bypass the OS cache or use it – we rely on OS for caching writes as well.
   * If a block\_id was beyond previous `node_count`, update `node_count` (and consider any gap filled).
   * Maintain or update the free list: if we overwrote a free slot, remove it from free list. If we appended new blocks, they’re not on free list. If we overwrote in-place an existing block, it wasn’t on free list anyway. (Deletion handling: any block that was a tombstone might simply be skipped – we’d not find it in shadow unless a tombstone update was stored, in which case we might write something to indicate emptiness. Alternatively, deletion could be handled by writing an updated neighbor list for others and not writing the deleted node at all, effectively dropping it.)
3. **Finalize Merge:** After writing all blocks from the shadow table:

   * Flush and **fsync** `graph.lmd` to ensure all writes are durable.
   * Wipe the delta table: execute `DELETE FROM __lmd_blocks;` to clear all entries. DuckDB’s WAL and storage will truncate this table efficiently (using its truncate optimization to avoid per-row cost).
   * Update the meta information: store the new `node_count`, update `free_head` (if any changes), maybe update a generation number for the index state.
   * Release locks and resume normal operation.

Because merges are incremental, the I/O volume is proportional to the number of dirty blocks. For example, writing 0.5% of nodes in a 400GB index is \~2GB of writes (which can complete in tens of seconds on SSD, instead of hours for rewriting the entire file). An empirical test showed merging \~1 million blocks (\~8GB of data) took \~38s on NVMe with <1GB RAM used for the operation. During merging, queries can continue using old blocks until a block is replaced, at which point the query will detect the update (our design can incorporate a generation counter that indicates a block was updated, so if a query encounters that block, it refetches it).

**Crash Safety during Merge:** We consider three crash scenarios:

* **Crash after some flushes but before merge starts:** No issue; the deltas remain in shadow\.duckdb. On restart, we see non-empty `__lmd_blocks` and can re-run the merge.
* **Crash in the middle of writing to `graph.lmd` (before fsync):** On recovery, `shadow.duckdb` still has all the delta entries (since we haven’t deleted them yet). We can safely retry the merge. Partially written blocks in `graph.lmd` are detected via checksum mismatches, but since the shadow still has the authoritative copy, we will overwrite those anyway.
* **Crash after `graph.lmd` fsync but before clearing shadow table:** In this case, the main file has been updated fully, but the shadow table still has duplicates of those updates. On restart, we will attempt the merge again. To avoid re-applying the same changes twice, we compare the `version` and checksum of each shadow entry with the `graph.lmd`. Since the versions match, we can skip or detect no changes needed. Thus, duplicates are harmless – we essentially perform an idempotent merge.

These measures ensure the index remains consistent and durable across crashes, with at most a small window of un-merged but committed data (which would still reside safely in the WAL of shadow\.duckdb).

### 7. Querying and Index Maintenance in Edge Cases

* **Index Opening:** On database startup or when attaching an index, the extension opens `graph.lmd` (memory-mapping it or opening a file handle), opens `shadow.duckdb` and `lookup.duckdb` connections. It verifies the integrity (e.g., checks a checksum in `graph.lmd.meta` to ensure the files match). If `shadow.duckdb` is non-empty (crash left un-merged deltas), the extension immediately runs a merge to apply those changes before servicing queries. If any required file is missing or corrupted, the index is marked unusable (queries will fall back to scanning the base table, and user can choose to rebuild the index).
* **Index Closing:** Ensure the flush thread is stopped. Any remaining dirty entries in the ring buffer are flushed (or we commit a final merge) so that no changes are lost. Close file handles and DuckDB connections. On `DROP INDEX`, delete the entire index directory recursively (with care on Windows if files are locked, possibly renaming for later deletion).
* **Background Vacuum:** Over time, if many deletions occurred, the `graph.lmd` file may have free blocks scattered and possibly a lot of free space at the end. A manual or automatic `VACUUM INDEX` operation can compact the file by relocating the highest blocks into free holes and truncating the file end. This process would iterate free list blocks, read the last block of the file into those freed slots, update neighbor references if needed (since block IDs remain the same, ideally we don’t move IDs around – instead we truly only reclaim tail free space after last used ID). Our design can track a `free_head` and possibly a sorted free list; a tiered free-list structure might be employed to group contiguous free runs for efficient reuse (not yet implemented, but noted as a future improvement). If any error occurs during compaction (I/O error), we abort and leave the file as-is (no data loss, just space not reclaimed).
* **Remote Storage:** Though not immediate, the design can extend to store the index folder on a remote filesystem (S3, etc.). By using DuckDB’s `FileSystem` interface for all file operations (which supports plugging in S3, etc.), and possibly using DuckDB’s buffer manager to handle read-ahead, the index could operate on cloud storage (noting higher latencies). This is mentioned as a future to-do (e.g. adding S3 support).

## C++ Component Interfaces

Below we outline the core classes/structs for implementation. (We use `struct` for plain data holders and `class` for components with logic).

```cpp
// Represents a fixed-size vector index node block in memory.
#pragma pack(push, 1)
struct NodeBlock {
    uint32_t    node_id;
    uint32_t    version;       // node_version (local epoch for neighbor updates)
    uint64_t    txn_id;        // creating transaction ID
    uint64_t    commit_epoch;  // commit timestamp/epoch
    uint32_t    num_neighbors;
    // ... possibly other flags (e.g., deleted flag)
    float       vector[DIM];   // uncompressed vector (DIM known at index build time)
    NeighborRef neighbors[MAX_DEGREE]; // neighbor list (IDs + compressed bytes)
    uint8_t     pad[...]       // padding to reach BLOCK_SIZE
    uint64_t    checksum;
};
#pragma pack(pop)

// An entry in the ring buffer for a dirty block.
struct DirtyBlockEntry {
    uint64_t    block_id;
    NodeBlock*  data_ptr;   // pointer to 8KB data in memory (cache)
    uint64_t    checksum;
};
```

**NeighborRef** could be a struct holding a neighbor ID and its compressed vector bytes. The `vector` field in NodeBlock is of fixed dimension (for flexibility, we might allocate vector+neighbors dynamically, but fixed block simplifies).

**Index class:** The main class `LMDiskANNIndex` (or similar) encapsulates the whole index and its operations:

```cpp
class LMDiskANNIndex {
  public:
    // Configuration
    const std::string index_path;
    const idx_t dim;
    const idx_t block_size;
    const std::string metric;

    // Constructors, open/close
    LMDiskANNIndex(std::string path, idx_t dim, idx_t block_size, MetricType metric);
    void Open();    // opens files and starts flush thread
    void Close();   // flushes pending, stops threads, closes files

    // Index operations
    void Insert(const Vector &vec, row_t row_id);
    void Delete(row_t row_id);
    std::vector<row_t> Search(const Vector &query, size_t k);

    // Maintenance
    void FlushDeltas();    // force flush daemon to run
    void Merge();          // force compaction
    void Vacuum();         // reclaim space

  private:
    // File handles
    FileHandle graph_file;
    DuckDB    *shadow_db;
    DuckDB    *lookup_db;
    Connection shadow_conn;
    Connection lookup_conn;
    PreparedStatement ps_insert, ps_select, ps_delete;  // for shadow table

    // Metadata
    std::atomic<uint64_t> node_count;
    std::atomic<uint64_t> next_commit_epoch;
    FreeList free_list;
    // Cache and buffers
    LRUCache<uint64_t, NodeBlock*> cache;
    LockFreeRingBuffer<DirtyBlockEntry> dirty_ring;
    std::thread flush_thread;
    std::mutex merge_mutex;
    std::vector<std::mutex> node_locks;  // e.g., one per N shards or per block if fine-grained

    // Internal helpers
    NodeBlock* readNodeBlock(uint64_t id, Transaction *txn);
    void writeDirtyBlock(NodeBlock *block);
    void flushDaemonLoop();
    void mergeAllDeltas();
    // ... etc.
};
```

*(This is a high-level sketch; in practice we integrate with DuckDB’s Extension API rather than designing a completely separate class.)*

Important methods and their function:

* `Open()`: uses DuckDB’s `db.OpenExternal(path)` for `shadow.duckdb` and `lookup.duckdb`, prepares statements (`PREPARE insert INTO __lmd_blocks VALUES (?, ?, ?, ?)` etc.), maps or opens `graph.lmd`. It also registers a **DatabaseEvent::Listener** for pre-checkpoint events to trigger a merge before DuckDB checkpoints. Then starts `flush_thread` running `flushDaemonLoop()`.
* `Close()`: signals flush\_thread to stop (possibly via an atomic boolean). If needed, performs a final `Merge()` to flush all changes. Closes DB connections and file handles.
* `Insert(vec, row_id)`: as described in insertion steps – normalizes vector, allocates node\_id (`node_count++`), locks neighbor selection, updates neighbors, pushes dirty entries, and inserts mapping into `lookup_conn`.
* `Delete(row_id)`: looks up node\_id from `lookup_conn` (`SELECT node_id FROM lmd_lookup WHERE row_id=?`), then proceeds with deletion logic (mark node, update neighbors etc.) and enqueue dirty blocks. Also `DELETE FROM lmd_lookup WHERE row_id=?` in the mapping table.
* `Search(query, k)`: as described, performing a graph traversal. Likely uses a `Transaction` handle or at least reads the current transaction ID/epoch to pass to `readNodeBlock()` for visibility checks. It might leverage multiple threads for search if desired (e.g., multiple entry points), but initially single-thread BFS is fine.
* `flushDaemonLoop()`: implements the ring buffer consumption. It uses the `ps_insert` prepared statement to batch write to `__lmd_blocks`. It also checks if the shadow table size exceeds threshold to invoke `mergeAllDeltas()` (compaction) or if signaled by checkpoint.
* `mergeAllDeltas()`: acquires `merge_mutex`, runs the SELECT query on `__lmd_blocks`, iterates as described, writing to `graph_file`. It uses `FileSystem` I/O APIs which might allow direct scatter/gather writes or at least sequential writes.
* `readNodeBlock(id, txn)`: encapsulates the logic of retrieving a NodeBlock: check cache, check shadow (using `ps_select` prepared statement on shadow table for that id), then file. It also enforces the visibility rule by comparing `block.commit_epoch` to `txn.start_epoch` (DuckDB provides the current transaction’s start time/ID).
* `writeDirtyBlock(block)`: computes checksum, and pushes to ring. Might also mark a global `delta_bytes` counter to decide on merge triggers.

We will reuse DuckDB subsystems wherever practical:

* The **DuckDB storage and WAL** in `shadow.duckdb` and `lookup.duckdb` saves us from writing custom persistence code for deltas and mappings. DuckDB ensures atomicity (WAL frames are atomic at filesystem level) and recovery (replaying the WAL on crash) for those components.
* The **DuckDB TaskScheduler** and event listener interface help us hook into checkpoint events and schedule background tasks, rather than managing our own threads entirely.
* The **FileSystem API** from DuckDB abstracts file I/O, so our code can use `FileSystem::Read`, `FileSystem::Write`, `CreateDirectory`, etc., which handle platform differences and allow easy future integration with S3 or other storage backends.
* We could consider using DuckDB’s **BufferManager** for caching `graph.lmd` pages. However, since we want a custom LRU at the granularity of logical blocks (which contain semantic info like neighbor lists), and because `graph.lmd` is not a DuckDB table, it’s simpler to manage caching ourselves. We do ensure, though, that large reads/writes (e.g. the 4MB bounce buffer) align with page boundaries to play nicely with OS caching.
* For neighbor search computations, we rely on our own implementation (DiskANN algorithm), but all data is readily accessible via our NodeBlock structure without additional translation.

## Node Lifecycle and State Transitions

Each node (vector) in the index goes through a clear lifecycle in terms of memory and storage:

1. **Created (in-memory, uncommitted):** A new NodeBlock is allocated when a vector is inserted. It exists only in memory (cache) initially, marked with the inserting transaction’s ID and no commit epoch. Neighbor links are established in memory.
2. **Flushed to Shadow (committed, not merged):** Once the transaction commits, the flush thread writes the node’s block (and any affected neighbors) to the `__lmd_blocks` table in `shadow.duckdb`. At this point, the node is durable (survives crashes) and visible to new transactions. The authoritative copy of the node is the one in the shadow table (since `graph.lmd` doesn’t have it yet). If the database restarts now, the index will load this node from shadow.
3. **Merged to Base (persistent):** On the next compaction, the node’s block is written into the `graph.lmd` file at position `node_id * BLOCK_SIZE`. After a successful fsync, the node is permanently part of the base index file. The shadow entry is then removed. Now the authoritative copy is in `graph.lmd`. The node remains in cache if frequently used.
4. **Updated (dirty in cache):** If the node’s neighbors are later updated (e.g., through insertion of another node or graph optimization), a new version of its block is created in memory (copy-on-write). Its `version++` and potentially new `txn_id` if this update is part of another transaction. This dirty version is flushed to shadow (step 2 again) and eventually merged (step 3), overwriting the old data in `graph.lmd`. Readers between flush and merge will get the latest from shadow.
5. **Deleted (tombstoned):** If the node is removed, its block may be marked as deleted. That information is flushed to shadow as well (or at least the neighbors are updated to drop it). After merge, the node’s slot in `graph.lmd` is free. The node effectively no longer exists, though its ID is retired.
6. **Vacuumed (reclaimed):** In a vacuum/compaction, if the node was last in the file or we consolidate free space, the file may shrink and free that space. The node’s ID remains unused for future (unless we implement ID reuse, which we avoid for now).

We can illustrate the life cycle in terms of which storage holds the latest copy and the visibility:

* **During a transaction insert:** Node is only in memory (not visible outside txn).
* **After commit, before merge:** Latest copy in shadow table (shadow\.duckdb); base file either has nothing (for new node) or an old version. Queries see the shadow version via our lookup.
* **After merge:** Latest copy in base file; shadow has none. (Cache likely has it too.)
* **If updated again:** Dirty in memory (cache), then new version in shadow, then merged to base (overwriting previous base version).

This cycle repeats. At all times, one of {cache, shadow, base file} has the newest committed version of the node. The lookup order (cache → shadow → base) ensures we fetch the correct version.

## Performance and Trade-off Analysis

**Write Amplification:** This design **dramatically reduces write amplification** compared to a naive approach of rewriting the entire index file on each update. Only changed blocks are written to the WAL (shadow) and later to the index file. In the best case, each update is written twice (once to shadow WAL, once to main file) plus a tiny WAL metadata overhead – a huge improvement over writing gigabytes for every small change. The trade-off is that we store changes temporarily in two places, but the merge process is efficient and only writes what’s necessary. Write amplification is further mitigated by batching: multiple updates to the same block before a merge result in only one final write (we replace older versions in shadow). The cost of maintaining checksums is negligible compared to the safety it provides.

**Read Amplification:** A query may need to read from multiple sources (cache, then possibly shadow table, then disk) in the worst case, but practically this is minor. A cold query might check the shadow table for every block it accesses; however, the shadow table lookup is an indexed DuckDB query (O(log n) or O(1) with our PK) and typically much faster than a disk read (\~20µs vs 100µs). Once warmed up, frequently accessed blocks will reside in the in-memory LRU or OS cache, giving near-RAM speed (25ns LRU hit, as noted). The *neighbor compression* scheme slightly increases CPU time (to decompress vectors on the fly) but **avoids extra disk reads** – each block read yields neighbor info that would otherwise require separate random reads. Thus, search can often proceed without fetching every neighbor’s full vector block unless necessary, keeping I/O low. Redundant storage of neighbor vectors *increases storage* but not the number of I/O operations needed.

**Memory Overhead:** The memory footprint is kept minimal. We do not store all vectors in memory (unlike original DiskANN) – only a small LRU cache (configurable size, say a few hundred MB for millions of vectors) and the overhead of running DuckDB for the two small databases. The shadow and lookup DuckDB instances will consume some memory for buffer pool and WAL, but given their size (lookup table scales with number of items, shadow table is mostly empty except during bursts), this is manageable. The compression of neighbor vectors means we do a bit more CPU work per read, but drastically cut down memory if we ever decided to cache some neighbor info. Overall, the design meets the *low-memory* goal by storing the heavy data on disk and using block-level caching.

**Concurrency and Latency:** Insertions are asynchronous relative to queries. A query might not immediately see a just-inserted vector until commit and flush (or an explicit sync) – this is acceptable under our eventual consistency model (similar to how indexes might be slightly stale between flushes). We mitigate staleness by flushing frequently (and at commits). The flush daemon ensures that even if many inserts come, they’ll be persisted quickly in the background. For read latency, one concern is that if a needed neighbor resides only in the shadow table (not merged yet), we incur a slight overhead to fetch it from `shadow.duckdb`. But since that fetch is from a local database (likely memory or OS-cached) and returns a ready-to-use blob, it’s still quite fast (microseconds).

**Scalability:** The design should handle tens of billions of vectors with some adjustments:

* The 32-bit neighbor ID limit (4 billion) may become a problem beyond that scale. We may need to allow 64-bit IDs or implement some sharding (multiple index files) at extremely large scales. For now, 32-bit IDs cover up to 4B nodes which may suffice for many use cases; forward-compatibility can be kept by reserving 64-bit fields if needed.
* The size of the DuckDB mapping table (lookup) will grow with number of items; DuckDB can handle billions of rows, but performance should be monitored. Alternatively, an external ART or B+Tree on disk could be used if needed, but reusing DuckDB’s indexing is preferable.
* The flush and merge process is scalable because it’s proportional to changes. Even at billion-scale, writing, say, 1% of nodes (10 million nodes) in a merge is feasible on fast SSD (tens of GB of writes). We can tune the merge trigger threshold (maybe use a tiered approach if the shadow grows very large, merging in chunks).
* Multi-threading the search over large graphs (for latency) and parallel merges (to use more cores) would be areas to optimize as the data scales up.

**Correctness Trade-offs:** The **consistency model** is slightly weaker than a fully synchronous index – there can be a gap between a transaction commit and the index reflecting it (until flush). In extremely unlucky crash scenarios, a few last transactions could be missing in the index if the system goes down before flush. We provide tools to detect and handle this (e.g., comparing base table count vs index count on startup as a heuristic). In practice, frequent flush on commit and WAL durability make the window very small. Users are made aware that recently committed data might not be immediately queryable via the ANN index until after a flush, which is akin to eventual consistency. This is a conscious design decision to favor performance.

The benefits of the architecture outweigh these minor trade-offs: we achieve orders-of-magnitude faster update throughput and maintain correctness via MVCC principles, making the index usable in transactional environments where previous designs failed.

## Implementation Plan (Step-by-Step)

To implement LM-DiskANN with this architecture, we propose the following plan:

1. **Skeleton Extension Setup:** Start by creating a DuckDB extension project for the LM-DiskANN index. Define the catalog structures to register a new index type and the `vector_top_k` table function for querying. Ensure the extension can parse `CREATE INDEX ... USING LMDiskANN` syntax and create the folder structure (using `FileSystem::CreateDirectory`).
2. **Folder and File Management:** Implement functions to create and open an index directory:

   * Creation: make directory, open/initialize DuckDB databases for `shadow` and `lookup`. Use DuckDB API to execute PRAGMA `journal_mode=WAL` on these to ensure WAL durability. Initialize `graph.lmd` and `graph.lmd.meta` (write placeholder header with dimension, etc.).
   * Open: locate the directory from the main DB catalog (the catalog can store the path and config in JSON). Open the DuckDB instances (using `duckdb_open_ext` or similar). Verify and perform immediate merge if needed.
   * Close/Drop: implement proper cleanup as described (stop threads, remove files).
3. **NodeBlock and Graph File I/O:** Define the `NodeBlock` struct layout according to dimension and neighbor list length. Implement low-level I/O: reading a block by ID from `graph.lmd` (e.g., via `pread` into a 8KB buffer) and writing a block to `graph.lmd`. Use DuckDB’s `FileHandle` and `FileSystem` for these to be cross-platform. Also implement computeChecksum() for a NodeBlock for integrity.
4. **Shadow Table Access:** Using the `shadow_conn`, create the `__lmd_blocks` table on index creation. Prepare the four statements (or use the C++ API directly):

   * `ps_insert = shadow_conn.Prepare("INSERT OR REPLACE INTO __lmd_blocks VALUES (?, ?, ?, ?)");`
   * `ps_select = shadow_conn.Prepare("SELECT data, version FROM __lmd_blocks WHERE block_id=?");`
   * `ps_range = shadow_conn.Prepare("SELECT block_id, data, version FROM __lmd_blocks ORDER BY block_id");` (for merge)
   * `ps_delete = shadow_conn.Prepare("DELETE FROM __lmd_blocks WHERE TRUE");` (to clear all, or maybe use `DELETE WHERE block_id <= X` if needed).
     Ensure these are cached for reuse. Test basic insert and select into the table.
5. **Lookup Table Integration:** On index create, in `lookup_conn`, create the `lmd_lookup` table (columns row\_id, node\_id). Create an index on `row_id` (it’s PK) and possibly on `node_id`. For DuckDB integration, also consider adding a foreign key in main DB if needed (though not strictly necessary). Implement lookups via this table for search results and deletions.
6. **LRU Cache and Locking:** Implement a simple LRU cache for NodeBlock pointers. This can use an `std::list` and hashmap for O(1) operations, or leverage an existing cache library. Integrate a mutex for thread safety (or more granular locks per entry as needed). Also set up an array of spinlocks or mutexes for node update synchronization. Initially, a global lock for any insertion might be fine; optimize later with sharded locks.
7. **Search Functionality:** Implement the `Search(query, k)` method. This involves writing the graph traversal logic, which can reuse a priority queue from C++ STL. Pay attention to using our `readNodeBlock(id)` to fetch nodes, which in turn checks shadow and base. Manage a transaction or snapshot such that `readNodeBlock` knows what commit epoch to compare against (DuckDB can provide the current transaction’s ID/epoch if the search is executed as part of a SQL query). Test search on a small static graph first (without dynamic updates).
8. **Insertion Logic:** Implement `Insert(vec, row_id)`. This includes:

   * Normalizing the vector if needed.
   * Allocating new node\_id = node\_count.fetch\_add(1). If a free list is used for physical space, handle that, but still assign a fresh logical ID.
   * Running a search (as in step 7) to find neighbors for this new vector. Possibly implement the **robust prune** algorithm: do an ANN search with a relatively high `L` to get a candidate neighbor set, then prune that set to the best M neighbors according to some heuristic (like ensuring diversity).
   * Lock the neighbor nodes (to prevent concurrent modifications) and update their neighbor lists to include the new node (if appropriate). Mark those neighbors dirty.
   * Create the NodeBlock for the new node, fill neighbors, etc., mark it dirty.
   * Use `lookup_conn` to insert the mapping (this is within the user transaction).
   * Push all dirty blocks to the ring buffer. If ring size > threshold, notify flush thread (perhaps via condition\_variable).
   * (We will refine this with proper error handling: if insertion fails mid-way, roll back changes in memory, etc.)
   * Ensure that if this Insert is called multiple times in one transaction (bulk insert), we handle accumulating multiple new nodes properly.
9. **Flush Daemon:** Implement the background thread routine `flushDaemonLoop` that waits on a condition variable or timeout. When triggered, it pops entries from `dirty_ring`. Deduplicate by block\_id (we can keep only the last one for each). Then bind parameters in chunks to `ps_insert` and execute. Use a transaction on `shadow_conn` around the batched inserts for efficiency. After commit, update a `delta_bytes` counter (sum of BLOB sizes inserted). Clear the condition or signal.

   * Also, inside this loop, check if `delta_bytes > MERGE_THRESHOLD (e.g. 256MB)` and if so, call `mergeAllDeltas()`. After merge, reset `delta_bytes=0`.
   * Use atomic flags or an event to handle graceful shutdown of the thread.
10. **Merge (Compaction):** Implement `mergeAllDeltas()`. Acquire `merge_mutex` to block concurrent merges/flush (the flush thread should not insert during merge – or we can pause flush during merge by design). Use `ps_range` to query all delta rows in sorted order. For each row, do the file writes as described. One challenge: performing millions of small writes – solve by collecting consecutive block writes: e.g., buffer 4MB at a time (512 blocks of 8KB) and write with one system call. Our design mention a *4MB bounce buffer*. So allocate one such buffer, and as we iterate deltas, copy each 8KB into the buffer. When buffer fills or a gap in offsets is detected, flush it to disk. This optimizes disk throughput.

    * After writing, fsync the file handle.
    * Then execute `ps_delete` to clear the shadow table. DuckDB will truncate the table efficiently.
    * If this merge was triggered by a checkpoint event, ensure the extension’s metadata (node\_count etc.) is updated in `graph.lmd.meta` so that the checkpoint on main DB records up-to-date info (the extension’s `Serialize` method should write the meta and possibly store some info in main DB catalog JSON).
11. **Testing Crash Recovery:** Simulate or test the scenarios:

    * Flush half-written: corrupt the WAL and ensure recovery still uses older data (we can simulate by killing process at points).
    * Merge crash after fsync: ensure duplicate entries are handled (this could be tested by injecting the same entry twice and verifying merge skip logic by comparing `version`).
    * Rollback and reuse: Insert and rollback some vectors, then insert new ones and ensure the old rolled-back ones are not visible.
    * Power loss test: after many ops, kill process, reopen index via `Open()` and ensure data is consistent (perhaps using the “crash gap detector” of comparing counts).
12. **Deletion & Update:** Implement `Delete(row_id)` fully and test it. Also consider a function to *update* an existing vector’s embedding (this could be done as a deletion + insertion, or if supported, directly altering the vector in its NodeBlock and marking dirty).
13. **Performance Tuning:** Profile the insertion throughput and search latency:

    * If insertion is CPU-bound on neighbor search, consider multi-threaded neighbor search or using a heuristic initial graph (like placing new node close to a randomly picked existing node then refining).
    * Tune flush frequency and batch sizes for WAL flush (maybe increase WAL page size if needed).
    * Ensure the lookup table access is not a bottleneck (maybe cache recently used rowid->nodeid mappings in memory).
    * Try with 100M vectors (if possible, simulate by smaller dimension to fit test environment) to see if any bottlenecks in memory or merging arise.
14. **Integration with DuckDB Query Planner:** Make sure that when a vector index exists, queries like `SELECT ... FROM table WHERE vector_column ANN ...` use the `vector_top_k` function to perform the search and then join by rowid. Implement the `vector_top_k` table function to call our `Search` and produce a table of rowid, distance.
15. **Documentation and Edge Cases:** Document the consistency model (that after a crash, some recent inserts might not be indexed but will be detected and can be rebuilt). Provide user guidance on when to run `CHECKPOINT` or `VACUUM INDEX`. Address any edge case like attaching multiple indexes (the design review mentioned potential “attach races” if many indexes open simultaneously – ensure each index’s lookup.duckdb attach is thread-safe, perhaps using a global mutex when attaching multiple indexes).

Each step should be accompanied by unit tests (where possible) or integration tests on sample data. We will also incorporate feedback and iterate on parts like neighbor locking granularity and potential use of more DuckDB internals if beneficial.

By following this plan, we will realize a robust, high-performance LM-DiskANN index integrated with DuckDB. This design directly addresses prior shortcomings: no more full file rewrites, and no more correctness issues from in-place mutations or stale references – all changes go through a WAL-backed shadow table and an epoch-based visibility filter. The result will be a production-ready, scalable vector index that can handle dynamic workloads on SSD while maintaining transactional consistency with the database.

**Sources:**

* Yu Pan et al. *“LM-DiskANN: Low Memory Footprint in Disk-Native Dynamic Graph-Based ANN Indexing.”* IEEE BigData 2023.  (architecture inspiration and block design)
* DuckDB LM-DiskANN design review (2024) – feedback incorporated on shadow tables, WAL, and rowID epochs.
* libSQL (TursoTech) blog on integrating DiskANN – explanation of LM-DiskANN block format and trade-offs.
