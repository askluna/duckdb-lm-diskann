Got it. I’ll generate a new, highly detailed technical specification document that fully aligns with your updated requirements:

* Focused exclusively on the LM-DiskANN algorithm and graph build pipeline
* Fully disk-based (no predicate labels or data in node blocks beyond ids, pointers, vectors, and neighbors)
* Designed for filtered queries, hybrid vector-FTS fusion, and DuckDB planner integration (scans, pushdown, filters, joins)
* Includes C++ API and struct signatures, index integration hooks, and a comprehensive breakdown of optimizer and scan behavior
* Considers parallel build with prefix-doubling, per prior docs

I’ll synthesize all prior documents, critical reviews, and the LM-DiskANN paper to ensure this spec is accurate, performant, and ready for implementation.

I’ll let you know as soon as it’s ready.


# Technical Specification: LM-DiskANN Graph Index Integration with DuckDB beta2

## Overview and Node Storage Layout

LM-DiskANN is a **disk-native graph-based ANN index** derived from DiskANN, designed for low memory usage by storing the graph on disk. Each data point (vector) is a node in a sparse **small-world graph** (Vamana graph) with edges to its nearest neighbors. Unlike the original DiskANN (which keeps a compressed copy of all vectors in memory), LM-DiskANN stores *complete routing information* on disk within each node’s block. This means **each node’s disk block contains**: the node’s identifier, its raw vector (embedding), a list of neighbor IDs, and compressed representations of each neighbor’s vector. By co-locating neighbors’ data in the same block as the node, the search algorithm can retrieve all necessary information (neighbors’ IDs and approximate vectors) with a single I/O per node, avoiding random disk accesses for neighbor vectors. All blocks are fixed-size (e.g. 4–8 KB) and aligned to disk pages for efficient reads.

&#x20;*Layout of a disk-resident index node block (LM-DiskANN format). Each fixed-size block stores the node ID, the node’s full vector, the list of neighbor IDs, and the compressed vectors of those neighbors; padding is used to meet the 8KB alignment (if needed).*

**Node structure:** No application-specific metadata (labels, masks, etc.) is stored in the nodes – only the data required for ANN search. In particular, **no predicate filters or category labels are embedded** in the node. This keeps the node layout uniform and minimal, containing only:

* **Node ID:** A unique integer (e.g., row ID or index in the dataset).
* **Vector:** The embedding coordinates of this node (either full precision float32 array or an optional quantized form).
* **Neighbor IDs:** An array of identifiers for the neighboring nodes in the graph (outgoing edges).
* **Neighbor vectors (compressed):** For each neighbor ID, a compressed version of that neighbor’s vector is stored (e.g. using ternary quantization or product quantization). This allows distance computations to neighbors without fetching their full vector from disk during graph traversal.
* **Pointer/offset metadata (optional):** e.g., an offset in the file or a pointer for the node’s block, if needed for quick access. (Often the node ID can be mapped to a file offset, so explicit pointers may be unnecessary.)
* **Degree/count:** The number of neighbor slots used (especially during construction). During parallel graph build, an atomic counter may track how many neighbors have been added so far for lock-free updates.

All node data is stored contiguously in an on-disk **block** (e.g., 8 KB) and aligned to a power-of-two boundary (8KB alignment). This fixed size guarantees that reading a node loads all its neighbor info in one go. The block includes padding if the actual data is smaller than the block size. The **vector dimensionality** and **max neighbors (R)** are chosen such that one node (ID + vector + R neighbors + R compressed vectors) fits in the block. For example, with 128 dimensions (float32) and R=64 neighbors, a 4–8KB block is sufficient to store the 512-byte vector plus neighbor info and some padding. If using ternary quantization (2 bits per component) for neighbor vectors, memory per neighbor is drastically reduced (e.g. 32 bytes for 128-dim) to help fit larger R or higher dimensions into the block. This fixed layout ensures **no dynamic resizing** of nodes is needed; each node’s neighbor list is capped at R entries.

**Ternary Vector Quantization & SIMD:** LM-DiskANN can store neighbor vectors in a **ternary quantized** format (each component ∈ {-1, 0, +1}). This lightweight compression trades some precision for dramatically smaller storage. During search, distance approximations can be computed using SIMD instructions by treating -1/0/+1 codes as multipliers on the query’s components. For example, the dot product or L2 distance between the query and a compressed neighbor can be computed by vectorized multiplication of the query’s float components with the neighbor’s {-1,0,1} values (expanded via lookup tables or bit masks). Because 16 components can be packed in 32 bytes, AVX2/AVX-512 registers can process many dimensions in parallel. This yields fast filtering of neighbors using approximate distances. Once a neighbor is chosen to explore, its full precise vector (already in that neighbor’s own block) can be fetched if needed for final distance refinement – though in LM-DiskANN, often the compressed distances are enough to guide search. The design leverages DuckDB’s vectorized execution philosophy by using SIMD to speed up these inner-loop distance calculations.

## Graph Build Algorithm (DiskANN/Vamana)

LM-DiskANN builds the graph index using DiskANN’s underlying **Vamana** algorithm. The goal is to construct a **directed graph** where each node has up to R outgoing edges to its nearest neighbors, enabling efficient approximate search. The build process can be done **serially** (one node at a time) or in **parallel batches**. Below we detail both strategies, which yield the same final graph structure.

### Serial Graph Construction (Vamana Insertion)

In the simplest case, nodes are added to the graph one by one in some order (e.g., in insertion order or random order). For each new vector **p**, the algorithm performs the following steps:

1. **Greedy search (neighbor candidate search):** Starting from an entry point in the existing graph (often the first node or the current “best” entry), perform a greedy walk: jump from node to node, always moving to the neighbor that is closest to **p** (in vector distance) until no closer neighbor is found. This produces a set of candidate neighbor nodes for **p**. The search is bounded by a parameter `L_insert` (the size of the candidate pool or “beam width” for insertion) to limit how many neighbors are evaluated. Essentially, **p** explores the graph until it reaches an approximate local minimum in distance.

2. **Robust prune:** Take the set of candidate neighbors found by the greedy search and sort them by distance to **p**. Then apply the **“α-pruning” rule** (from the Vamana algorithm): iteratively construct **p**’s neighbor list by adding the next candidate if it is significantly closer than the farthest already accepted neighbor. Formally, for each candidate neighbor `q` in order of increasing distance: include `q` as a neighbor of **p** if `α * dist(p, q) < dist(p, S)`, where `S` is the set of neighbors already selected for **p**. This rule ensures diversity in the neighbor list (some farther points are kept if they link to unexplored regions). The result is a pruned neighbor list (size ≤ R) for the new node **p**.

3. **Mutual neighbor updates:** For each neighbor `q` selected in the list, add **p** as a neighbor to `q`’s adjacency list as well (making the edge mutual). In the serial algorithm, this is done under locks to avoid concurrent modification issues. The reference DiskANN implementation even uses a global lock or per-node locks to protect neighbor lists during these inserts. If adding **p** would exceed `q`’s maximum degree R, the farthest neighbor in `q`’s list is evicted (to keep the degree ≤ R). In code, this looks like locking `q`, performing `InsertEdge(q, p)`, then unlocking.

4. **Entry point update (optional):** Optionally update the global entry point of the graph. A common heuristic is to track the node with the largest vector norm and use it as the entry for future searches (because high-norm points often are good navigators in high-dimensional spaces). If the new node **p** has a larger norm than the current entry, it becomes the new entry point. (This step is not critical for correctness but can improve search speed.)

5. **Insert new node into graph:** Finally, add **p** and its neighbor list to the graph’s data structure. In DiskANN’s case, this means writing **p**’s neighbor IDs (and possibly their compressed vectors) into **p**’s node block on disk, and marking **p** as present in the graph. In-memory, the neighbor count for **p** is set to the number of neighbors found. This step might also require locking a global structure if the graph’s node list is being updated (in the reference code, a global lock protects writing the new node’s own neighbor list).

**Pseudocode (serial build):** The process can be summarized in pseudocode for clarity:

```cpp
for each point p in dataset: 
    // 1. Greedy search for candidate neighbors
    auto candidates = GreedySearch(entry_point, p, L_insert);
    // 2. Prune candidates to get neighbor list
    auto nbrs = RobustPrune(p, candidates, alpha, R);
    // 3. Mutual updates: add p to each neighbor q
    for (node_id q : nbrs) {
        lock(q); 
        InsertEdge(q, p);            // add p as neighbor of q (evict worst if >R)
        unlock(q);
    }
    // 4. Add p's neighbor list to graph
    lock(global_graph);
    InsertNode(p, nbrs);            // set p's neighbor list (and store to disk)
    unlock(global_graph);
    if (‖p‖₂ > ‖entry_point‖₂) 
        entry_point = p;            // (optional) update entry point
}
```

This serial algorithm produces a high-quality graph but can be slow for large datasets. The time complexity is roughly **O(N · L\_insert · log R)** in terms of distance computations (each insertion does a search of width L\_insert and sorting/pruning of up to R neighbors). For example, building 10 million vectors can take hours on a single thread. To accelerate this, LM-DiskANN supports a parallel build mode.

### Parallel Batch Graph Build (Prefix-Doubling Strategy)

To leverage multiple CPU cores during index construction, LM-DiskANN can perform **batch-parallel insertions** inspired by ParlayANN and FreshDiskANN. The graph build is divided into rounds that *double* the number of nodes inserted in each round (prefix-doubling). Each round inserts a batch of new nodes concurrently, using atomic operations instead of coarse locks to update neighbor lists. This yields identical graph structure to the serial algorithm (given a fixed insertion order) but significantly reduces wall-clock time.

**Prefix-Doubling Batching:** The algorithm first builds a small initial graph prefix serially (e.g., the first P0 = 2^k nodes, such as 1024 nodes) to have a seed graph. Then it enters doubling rounds: in round 1, insert the next P0 nodes in parallel; round 2, insert the next 2*P0 nodes in parallel; round 3, next 4*P0, and so on, doubling the batch size each time until all N nodes are inserted. For example, if P0=1024, we build 1024 nodes serially, then 1024 in parallel (making total 2048), then 2048 in parallel (total 4096), 4096 (total 8192), etc. This approach keeps the workload per batch balanced (each thread handles roughly the same number of insertions) and ensures that when inserting a batch, the graph of previously inserted nodes is well-formed and can be used for searches.

**Concurrent insertion mechanics:** Within a batch of new nodes, each thread operates independently on its subset of nodes:

* All threads share a **read-only snapshot** of the graph as it exists before the batch. That is, when inserting batch nodes \[P, 2P), the graph containing nodes \[0, P) is considered fully built and is queried for neighbor searches. Threads do *not* see each other’s new nodes during their search phase, only the older prefix. This restriction (new nodes don’t link to each other within the same batch) preserves determinism and avoids race conditions in traversal. It slightly changes the graph construction process (batch peers won’t directly connect immediately), but the next round’s searches will discover those connections through older nodes, yielding the same final connectivity.

* Each thread performs **GreedySearch + RobustPrune** for its assigned node `x[i]` (for i in \[P,2P)) just like in the serial case. This yields a neighbor list `nbrs` for the new node *i* (with neighbors drawn from the first P nodes only).

* **Lock-free neighbor updates:** For each neighbor `q` in *i*’s list, the thread attempts to add *i* to `q`’s neighbor list using an atomic operation rather than a lock. In each node’s in-memory representation, we maintain an **atomic counter** for how many neighbors have been added so far (degree count). The neighbor list array has capacity R. The thread does:

  ```cpp
  do {
      k = degree[q].load();
      if (k >= R) break;                        // neighbor list is full, cannot add
  } while (!degree[q].compare_exchange_weak(k, k+1));
  if (k < R) {
      neighbors[q][k] = i;                     // place new neighbor at position k
  }
  ```

  This uses a compare-and-swap (CAS) on `degree[q]`. If the list is not full, the CAS reserves the next slot by incrementing the count; the thread then writes *i* into that slot of q’s neighbor array. If the list *was* full or another thread filled the slot first, the CAS will fail or `k>=R` check triggers, and the new edge (q←p) is skipped (meaning q already has R closer neighbors). This lock-free approach ensures **no more than R neighbors** are added to any node, even under race conditions, and avoids needing explicit locks on each neighbor list. It effectively performs the “mutual update” step concurrently and safely. (In the final graph, some new nodes might not get added to a particular neighbor q if there was contention and q’s list filled up – but in theory, those edges would have been the weakest and would be pruned in a serial build as well.)

* **Writing new node’s own list:** After processing neighbors, the thread writes the neighbor list of *i* into the new node’s block (in-memory). This includes writing neighbor IDs (and perhaps precomputed compressed vectors of those neighbors, if needed for search). The degree of *i* (outgoing edges count) is simply the size of its `nbrs`. This operation may be done under a thread-local buffer without locks since *i* is not being accessed by other threads.

* **Synchronization:** A barrier is used to ensure all threads in the batch complete their insertions before proceeding. Once the batch is done, the prefix is extended: P ← 2P (doubling the number of nodes considered built). At this point, all nodes \[0, 2P) have their neighbor lists, and the next batch can use them as the base graph.

**Determinism:** The above parallel method is crafted to produce exactly the same graph as a serial insertion in a fixed order, despite concurrency. The key is that the insertion order is effectively fixed (by prefix batches and by assigning each thread a fixed range of indices) and the usage of atomic retry loops for neighbors. Any time multiple threads try to add themselves to the same neighbor `q`, the CAS ensures only the first R successes get in; if a thread fails to insert into `q` because of a race, it means another thread’s node took that slot, which in serial would have happened if that other node had been inserted earlier. In essence, if two new nodes both consider `q` as a neighbor, the one with smaller index (or whichever the thread scheduling favors first, which we fix by deterministic thread assignment) will win the slot, and the other will be dropped – matching the deterministic tie-breaking of the serial algorithm. As reported in prior work, a deterministic batch strategy yields identical graphs on every run. LM-DiskANN’s implementation fixes the batch order and uses thread-index or node-ID ordering to break ties, so that the resulting graph is reproducible run-to-run.

**Parallel build performance:** Batch-parallel construction dramatically improves build times. For instance, FreshDiskANN (a similar approach) achieved \~8× speedup on 64 threads for billion-scale datasets. In LM-DiskANN, the parallel build is I/O-bound by sequential writes (writing large chunks of neighbor lists) rather than random writes per node. Each batch can flush new edges in bulk (e.g., threads buffer their neighbor additions and write in chunks of 256KB), which the DiskANN file format and DuckDB’s buffer manager can accommodate. Memory overhead is modest (each thread may use a buffer for pending writes, \~10% overhead). The trade-off is increased code complexity (managing atomics and batch coordination vs simple loops) and ensuring thread safety. However, given the need to regularly rebuild or bulk-load indexes in analytical databases, this parallel option can be gated by a build parameter (e.g., `CREATE INDEX ... WITH (parallel_build=TRUE, batch_size=1024)`). If disabled, the builder defaults to the simpler serial method.

**Alternative concurrency (streaming updates):** For dynamic workloads with frequent single inserts, a fully parallel batch may be overkill. A simpler middle-ground is to use a **background worker thread and a queue** of pending inserts. In this design, transactions that insert vectors simply enqueue the new vector’s ID (and data) into a thread-safe queue and return immediately. A dedicated index builder thread then pops from the queue and applies the serial insertion algorithm (greedy search + prune) for each, under a global lock or in small batches. This approach (1) avoids blocking the user’s insert transactions on index updates, and (2) keeps code simple (no CAS or complex sync – it’s still one thread building, just decoupled in time). The downside is it still processes one node at a time (not utilizing multiple cores for a single index build). LM-DiskANN’s implementation can start with this “queue + worker” approach for simplicity (minimal code changes, low risk) and later upgrade to the full batch-parallel build when higher throughput is required. In either case, the resulting graph structure remains purely based on vector distances; no additional metadata is involved in graph construction.

### C++ Structures for Graph Construction

The core data structures for the graph build are defined in C++ as part of the LM-DiskANN index implementation. Key structures include:

```cpp
struct DiskANNIndex {
    idx_t num_nodes;               // number of vectors (nodes) indexed
    uint32_t dim;                  // dimensionality of vectors
    uint32_t max_degree;           // R: max neighbors per node
    float alpha;                   // pruning parameter α
    bool use_ternary;              // true if using ternary quantization for neighbors
    BlockManager *block_mgr;       // handle to DuckDB buffer manager or file for index blocks
    idx_t entry_node;              // current entry point (node with max norm, etc)
    // Pointers to in-memory structures (populated during build):
    std::atomic<uint16_t>* degrees; // array of atomic degrees for each node (for parallel build)
    // ... possibly other metadata like vector norms, etc.
    
    // Build methods:
    void BuildIndex(const float* data, idx_t count, bool parallel);
    void InsertNode(idx_t node_id, const std::vector<idx_t>& neighbors);
    void GreedySearch(idx_t start, const float* query, uint32_t L, std::vector<idx_t>& out_candidates) const;
    void RobustPrune(const float* query, std::vector<idx_t>& candidates, std::vector<idx_t>& out_neighbors) const;
};
```

* **DiskANNIndex**: represents the index and holds global parameters and metadata. The `BuildIndex` method orchestrates the construction (either serial or parallel based on the flag). It uses helper functions `GreedySearch` and `RobustPrune` which implement the algorithm described above. Internally, the index uses a `BlockManager` (tied to DuckDB’s storage manager) to allocate and write fixed-size node blocks on disk. The `degrees` array is allocated if parallel build is enabled (with one atomic per node for safe neighbor updates). After build, the final degree of each node can be stored as a normal integer in its block (and the atomic array can be freed or reused for dynamic updates).

```cpp
struct DiskANNBuilderThreadState {
    // Used in parallel build for each thread
    std::vector<idx_t> local_new_nodes;   // the node IDs this thread is inserting
    std::vector<std::pair<idx_t,float>> candidate_list; // scratch space for search (node, distance)
    // ... potentially per-thread buffers for writing neighbor entries
};
```

In a parallel build, multiple `DiskANNBuilderThreadState` instances are used (one per thread) to hold thread-local data such as candidate lists, which avoids false sharing on data structures.

**Pseudo-implementation notes:**

* *GreedySearch(start, query)*: This performs the graph traversal: begin at `start` (entry\_node or a random node) and keep track of a current best distance. It maintains a small max-heap or list of candidates to explore. In each step, it pops the closest unexplored candidate and checks its neighbors, adding them to the candidate list if they haven’t been visited. It stops when the candidate list’s closest distance doesn’t improve over the best found. It uses the index’s block manager to load neighbors: e.g., load block of `current_node`, get its neighbor IDs and neighbor vectors (compressed), compute distances from `query` to each neighbor (using SIMD if compressed), and push neighbors into the min-heap of candidates. A visited bitmap or a visitation epoch counter is used to avoid re-visiting nodes. This returns a list of the best `L` candidates found.

* *RobustPrune(query, candidates)*: This will sort the `candidates` by distance ascending, then iterate and apply the α \* distance criterion to build a pruned neighbor list (size ≤ R). It may use another array to hold the selected neighbor IDs. This can be done in O(L log L) for sorting plus O(L \* |out\_neighbors|) for selection. The result is written to both the `out_neighbors` vector and directly into the new node’s disk block via `DiskANNIndex::InsertNode`.

* *InsertNode(node\_id, neighbors)*: This will allocate or locate a free block in the index file for the new node (if building incrementally) or use the next sequential block (if bulk building). It writes the node’s ID, the neighbor count, and neighbor IDs, and if using quantization, also computes & writes the compressed neighbor vectors. (The neighbors’ original vectors can be fetched from the dataset or perhaps from an array of all vectors if available in memory during build.) On completion, the block is flushed via DuckDB’s storage manager. If building in-memory first, this step might defer actual disk writes until the end (writing sequentially).

During **parallel build**, the main differences are: (a) `BuildIndex` will divide nodes into batches and spawn tasks on DuckDB’s `TaskScheduler` for each batch, each task running a loop similar to the pseudocode under *Parallel insertion mechanics* above, and (b) `InsertNode` for a batch might be done after all CAS updates, writing multiple nodes’ blocks in one go or in parallel (with proper synchronization to avoid interleaving writes). The `degrees` atomic array is used by tasks to perform CAS when inserting mutual edges. After a batch, a synchronization ensures all neighbor updates are visible before the next batch’s searches.

**Thread safety and locking:** In serial mode, the implementation uses simple `std::mutex` locks for updating neighbors (or even coarse locking around each insertion as needed). In parallel mode, fine-grained atomic operations replace most locks. A remaining lock might protect the allocation of new node IDs or writing to certain shared file structures, but neighbor list updates are lock-free. The design avoids modifying any existing node’s neighbor array beyond adding an ID to an empty slot via CAS – thus no two threads ever remove or reorder edges concurrently (pruning decisions are made only by the owner thread of the new node, and existing neighbors only get appended to). This greatly simplifies correctness reasoning.

After build, the on-disk graph is finalized. Each node’s block has the final neighbor count and neighbor list. The atomic counters (if any) can be reinitialized for use during future dynamic inserts or simply not persisted (on disk we only need the final neighbor count). The graph can now be used for querying.

## Integration with DuckDB Optimizer and Execution

To use the LM-DiskANN index inside DuckDB, we integrate it as a custom index type and provide hooks into DuckDB’s query planning and execution engine. This allows queries to utilize the index for fast vector similarity searches, including hybrid queries that combine vector search with other predicates (e.g., filtering by metadata or full-text search conditions). The integration points span **index creation**, **planner rules**, and **a custom scan operator** that performs the DiskANN graph traversal at runtime.

### Index Creation and Storage in DuckDB

LM-DiskANN will be implemented as a DuckDB **index extension** (similar to how DuckDB supports ART indexes). Users can create the index with a statement such as:

```sql
CREATE INDEX myindex ON table_name USING DISKANN(vector_column)
-- optional parameters: DIM=d, BLOCK_SIZE=8192, ALPHA=1.2, R=64, parallel_build=true ...
```

At index creation, DuckDB will call into the extension’s index builder, which in turn uses `DiskANNIndex::BuildIndex` on the specified table’s vector column. The index can be stored in DuckDB’s storage as part of the table’s metadata (likely in the form of a binary blob or an external file managed by the index). We leverage DuckDB’s **BufferManager** and file I/O API to allocate a contiguous region of the database file (or a separate index file) for the graph’s blocks. For example, if there are N vectors, and each node block is 8KB, the index might occupy an 8KB\*N segment in the file. The `BlockManager` is used to pin/unpin pages of this segment during queries (more on this below).

**Persisting the index:** The index’s configuration (dimension, R, etc.) and possibly a small meta-data (like the entry point node ID, or a pointer to a start of free list for dynamic inserts) will be stored in DuckDB’s meta storage so that on database restart the index can be reconstructed (or memory-mapped) without full rebuild. The neighbor graph itself lives in the file pages. DuckDB’s buffer manager ensures that frequently accessed index pages stay in memory cache, while infrequently used ones can be evicted, thereby handling the I/O caching transparently.

**Index maintenance (insertion/deletion):** If the table is static or append-only, we can batch build the index once. For dynamic tables, whenever a new row is inserted (with a vector), DuckDB will call the index’s `Insert` method (similar to how it updates an ART index on inserts). Our `DiskANNIndex::Insert` can either do the **synchronous update** (greedy search + prune for that vector) or push the update to a background thread. As discussed, a simple approach is to enqueue the new vector into a build queue and have a background task apply it; DuckDB’s transaction commit would mark the index as updated once the enqueue is successful (since the background thread will eventually add it). Because the index does not store any *filtered* information, there’s no need to update label masks or similar – each insert is purely based on the new vector’s position in metric space. For deletions, a straightforward approach is **lazy deletion**: mark the corresponding node as removed (e.g., via a tombstone bit in an external structure or a “deleted” flag array). The node’s block might remain in place but a runtime check can skip it during search. True deletion (removing edges) is more involved and may be deferred or done by a background rebuild of affected parts. (In this spec we assume deletion is infrequent and can be handled by higher-level logic, not by altering node blocks on the fly.)

### Planner Hooks and Query Execution

DuckDB’s optimizer needs to be aware of the DiskANN index to use it in query plans. There are two ways the index can be invoked:

* **Implicit use via predicates:** If a query includes a vector similarity condition (for example, a function or operator that computes distance), the optimizer could recognize that an index is available to satisfy that condition. However, standard SQL has no built-in syntax for “nearest neighbor” search in a WHERE clause. In DuckDB, this might be handled via a special **comparison operator or function** (perhaps an extension like `"<->"` for distance or a function `vector_distance(vec, const_vec) < r`). We can extend the binder to translate a clause like `vector_column ANN(query_vec) /* ANN stands for approximate nearest neighbor */` into an index scan on the DiskANN index. Because this is complex, the more straightforward approach is the next point.

* **Explicit use via `USING INDEX`:** The user (or application) can hint or enforce the index usage with a clause `USING INDEX myindex`. For example:

  ```sql
  SELECT * 
  FROM table_name 
  WHERE vector_column <-> [1.0, 2.0, ...] < SOME_THRESHOLD 
  USING INDEX myindex;
  ```

  Or, more directly, DuckDB might support a table-valued function or subquery like:

  ```sql
  SELECT * 
  FROM table_name 
  USING INDEX myindex(vector_query => [1.0,2.0,...], k => 10) 
  WHERE other_filters...;
  ```

  The exact syntax can vary, but the key is the optimizer will be instructed to plan an **IndexScan** using `myindex` for retrieving the top-K nearest neighbors to the given query vector. The `USING INDEX` clause essentially bypasses the need for cost-based comparison (it assumes the index should be used).

**Planner integration details:** We implement a new planner path analogous to an index scan:

* **Binding:** A new LogicalOperator, e.g. `LogicalIndexScan`, is created when the parser sees the `USING INDEX` or when it recognizes a pattern that can use the index. The bind function will locate the `DiskANNIndex` object by name (from the catalog), and parse any provided parameters (like the query vector, or reference to a parameter/constant that holds the query vector, and `k` if specified). It produces a## DuckDB Query Planning and Execution with DiskANN

Once the index is built and stored, DuckDB’s optimizer and execution engine must orchestrate its use in queries. LM-DiskANN is integrated as a custom index type, which means the planner can choose an **Index Scan** over a table using this index. The integration involves special handling for query vectors and the ability to combine (push down) additional filters. Key aspects include the `USING INDEX` clause to force index usage, planner hooks (bind, cost, init, scan functions), and ensuring other predicates (filters) are applied outside the index.

**Planner support (`USING INDEX` and pushdown):** DuckDB allows queries to explicitly specify an index via the `USING INDEX index_name` hint in a SELECT. For example:

```sql
SELECT * 
FROM vectors 
WHERE vectors.embedding_vector <-> [0.1, 0.2, ...] < 0.5 
USING INDEX my_diskann_idx;
```

This hypothetical query finds all rows with embedding vectors within distance 0.5 of the query vector (given as a literal array) using the DiskANN index. The `USING INDEX my_diskann_idx` instructs the optimizer to plan an index scan using `my_diskann_idx` instead of a full table scan, even if the cost is not known. In practice, vector searches often use a **top-K** selection rather than a distance threshold. DuckDB can expose a table-valued function (similar to SQLite’s `vector_top_k`) or treat the index scan as returning the K nearest neighbors as a set of row IDs. For instance, a query could use a subquery or join:

```sql
SELECT t.* 
FROM vector_index_search('my_diskann_idx', [0.1,0.2,...], 10) AS nn(rowid, distance)
JOIN vectors AS t ON t.rowid = nn.rowid
WHERE t.category = 'foo';
```

Here, `vector_index_search` is a table function that takes an index name, a query vector, and K, and returns the top K rowids with their distance scores. Under the hood, this could be implemented by our DiskANN index scan operator. Whether via `USING INDEX` or a table function, the planner will route the query to use the index and also capture any additional filters (like `category = 'foo'` above). These filters are provided to the index scan as a DuckDB **TableFilterSet**, which is a collection of predicates on columns that the scan should enforce.

**Bind phase (planner):** When the optimizer binds the plan for an index scan, it will call into the DiskANN index’s bind method. This method prepares a **bind data** object containing:

* A reference to the `DiskANNIndex` (the index metadata and structures).
* The query vector (as a constant or parameter). If the query vector is given as a literal in SQL, it’s captured as a DuckDB `Value` or constant vector. If it’s parameterized (e.g., a prepared statement parameter), the bind data notes the parameter index so it can be fetched at runtime.
* The number K (if applicable) or search parameters (e.g., search depth `L` or alpha, though these typically default from index settings).
* The **TableFilterSet** of other conditions. DuckDB’s optimizer will push down simple filters on the same table into the index scan if possible. For example, `category = 'foo'` might be passed in the filter set.
* Optionally, a pointer to any other index that can be used for filtering. For instance, if `category` has an ART index, the bind might locate that to help later.

The bind returns this information, and the logical plan now has an `LogicalIndexScan` node with our DiskANN index and bind data attached. The cost estimation for this node could be minimal (since ANN index search is sub-linear), but we avoid detailed cost speculation. DuckDB will typically favor an index if the user explicitly requested it or if it’s expected to reduce data significantly (e.g., retrieving 10 rows out of millions). In our scenario, assume `USING INDEX` directs usage regardless of cost.

**Physical operator (`DiskANNIndexScan`):** During the physical planning, DuckDB will create a custom operator, say `PhysicalDiskANNIndexScan`, that uses the DiskANN index. This operator will produce rows from the base table that satisfy the vector similarity search (and any other filters). The operator’s primary tasks are:

* **Init (DiskANNScanState):** In the `Initialize` step, prepare the search. This includes retrieving the actual query vector data (if it was a parameter or constant `Value`, convert to a float array). If the query vector is of type blob or DuckDB’s `VECTOR` type, ensure it’s accessible as `float*`.

  * If additional filters were provided, now is the time to prepare them for efficient use. For each pushed-down filter, if there is a quick way to get the qualifying row IDs, do so:

    * For equality or range filters on columns with an ART index (DuckDB’s default index for scalar columns), use that index to **lookup the rowid set**. For example, if filter is `year = 2020` and `year_idx` is an ART index, perform an index scan on `year_idx` to collect all rowids matching 2020. This yields an ordered list or bitmap of allowed rowids.
    * For a full-text search (FTS/BM25) condition, use the FTS index (which might be implemented as a separate table of \[term -> list of rowids]) to get the set of rowids that contain the term. DuckDB’s FTS extension could provide an API to get a bitset or vector of matches.
    * If multiple filters, intersect their rowid sets. For instance, if one filter yields {rowids with category='foo'} and another yields {rowids with year=2020}, take the intersection for allowed rowids.
    * If filters are present but no index or quick path exists (e.g., an arbitrary expression filter), as a fallback the operator can still apply the filter after finding neighbors (post-filtering). In that case, we don’t precompute allowed rowids but will evaluate the predicate on each candidate later.
  * The result of this could be an **allowed\_ids structure**: perhaps a boolean bitmap indexed by rowid, or an ART index of allowed ids, or simply a sorted vector of allowed ids. The simplest is a boolean array of length N (number of rows in table) marking allowed/disallowed. But for large N, that’s memory heavy if the filter is sparse. A sorted vector of allowed ids might be fine and can be binary searched. Since DuckDB uses rowids up to N, a bitset might be okay if N is not huge.
  * Store the prepared query vector and allowed\_ids in the operator state (`DiskANNScanState`). Also allocate a container for results (neighbor list).
  * Then, perform the **DiskANN search**: call something like `DiskANNIndex::Search(query_vec, K, allowed_ids)` which executes the **beam search** (greedy graph traversal) on the index:

    * Start from the index’s `entry_node` (or possibly a set of entry nodes) and use the neighbors graph to find nearest neighbors to `query_vec`.
    * Maintain a min-heap or candidate list of nodes to explore (initialized with entry\_node). Also maintain a `visited` set (or bitset) to avoid revisiting nodes.
    * Iteratively pop the closest candidate from the heap, compute its distance, and if it’s within the current best range or if we haven’t found K neighbors yet, load that candidate’s neighbors from disk:

      * Pin the neighbor’s block via DuckDB’s `BufferManager` (to avoid multiple I/O if reused soon).
      * Read the neighbor IDs and their compressed vectors from the block.
      * For each neighbor, if `allowed_ids` is set and the neighbor’s ID is not allowed, **skip it** (do not add to candidates) – this is the **pushdown filter effect**. If allowed or no filter, compute distance from `query_vec`:

        * Use SIMD to calculate the distance quickly (if neighbor vectors are compressed, decode on the fly or use precomputed dot products). This may use AVX intrinsics on 128-dim chunks.
      * If the neighbor has not been visited and either we haven’t yet found K neighbors or its distance is less than the worst distance in our current top-K list, add it to the candidate min-heap.
    * Continue until the heap is exhausted or a stopping condition is met (e.g., we have examined a certain number of candidates, typically controlled by `L_search` or similar parameter).
    * The algorithm collects the closest neighbors found. Sort them by distance and take the top K.
    * This yields (rowid, distance) pairs for the nearest neighbors.
    * All of this happens within the `Init` (or the first `GetChunk`) of the operator, meaning it’s executed once per query.
    * Complexity: DiskANN search is roughly O(L\_search \* log N) for the heap operations but empirically visits far fewer nodes than N. The index ensures recall/latency tradeoff so that K neighbors are found with minimal I/O. Typically L\_search (often denoted `ef` in HNSW literature) might be 50 or 100 to get K=10 neighbors with high recall.
  * After search, the resulting neighbor IDs are stored in `DiskANNScanState.result_ids` (and maybe distances in a parallel array). Also mark in the state how many results and set a pointer at the start (state.index = 0).

* **Scan (producing tuples):** The `GetChunk` method of our operator returns the query results to the engine. It will output up to `STANDARD_VECTOR_SIZE` results at a time (DuckDB processes data in vectors of up to 1024 entries by default). In our case, K is likely small (tens or hundreds), so all neighbors might fit in one chunk. The operator:

  * Takes the next batch of neighbor rowids from `state.result_ids` (starting from `state.index`).
  * For each such rowid, it retrieves the actual row from the base table. This is essentially a **rowid lookup** in the base table:

    * DuckDB represents row identifiers in a compressed form (containing segment index and offset). Our node IDs might directly correspond to rowids if the table is append-only. If not, we may have stored a mapping from node index to physical rowid.
    * The simplest mapping is: node ID = row index (0-based). DuckDB’s rowid (as exposed by `rowid` pseudo-column) is an integer that encodes location, but we can treat it abstractly. We can convert our node ID to a `Value` of type BIGINT if needed.
    * To get the row’s values, we can use DuckDB’s table storage API. For instance, use `table.Fetch(transaction, row_ids, column_ids, result_chunk)` – if such an API exists. DuckDB likely has internal methods for index scans to fetch rows by rowid. (In the worst case, one could perform a point query on the base table using its primary key; but usually, rowid is direct).
    * If such API is not directly exposed, an alternative is to output only the `rowid` from the index scan and let a join with the base table occur (as in the above SQL example). However, since we want to tightly integrate, ideally our operator directly produces the base table columns. We can achieve this by gathering each column’s data:

      * For each column requested (the projection list), locate the column’s segments that contain these row positions and copy the values into the output chunk’s vector for that column.
      * DuckDB storage is columnar, so we might have to do one column at a time. But because K is small, this overhead is minor.
    * If any additional filters were *not* accounted in allowed\_ids (because maybe they were complex), we then apply them here: evaluate the predicate on each output row and discard those that don’t satisfy (this can be done by selection vectors in the DataChunk).
  * Fill the output DataChunk with the retrieved rows (and optionally a distance column if we want to output similarity score – not typical in pure SQL unless requested).
  * Update `state.index` and repeat on next call until all neighbors are output, then indicate completion.

Throughout execution, **DuckDB’s buffer manager** is used for I/O:

* The index’s on-disk blocks are accessed via `block_mgr->Pin(block_id)` which brings the page into memory (if not already) and returns a pointer to its contents. We pin before reading neighbor lists, and unpin after done (or rely on scope).
* Pinned pages might remain in cache for subsequent queries. If multiple DiskANN searches run concurrently, the same page could be pinned multiple times (the buffer manager handles reference counts).
* The base table’s data is also accessed via buffer manager or vectorized scans for row fetching.
* By using DuckDB’s storage infrastructure, we ensure consistency (if other transactions insert while a query runs, the index might need proper versioning – probably beyond scope, assume read queries see a consistent snapshot of the index).

**Support for hybrid queries:** Because the DiskANN index returns **row IDs** of similar items, combining vector search with other query conditions is naturally supported by set operations on row IDs. For instance, a query that involves vector similarity and a text search (`BM25`) can be executed by:

* Running the text search (FTS) index to get candidate row IDs that match the text condition.
* Running the DiskANN index search to get candidate row IDs by vector.
* Intersecting those ID sets to find rows that satisfy both.

This can be done in the optimizer by adding a join on rowid, or by, as described earlier, using the text condition to filter during the DiskANN search (allowed\_ids). The **graph structure is not modified at all** for these filters – the graph remains a global index on all vectors. We’re effectively querying a subgraph (the nodes that pass the other condition) by ignoring edges leading to disqualified nodes. This approach avoids any need for “predicate masks” stored in nodes or pre-partitioning the graph by filters.

At runtime, the combination is efficient because:

* The text filter yields an **ART index** of rowids or a bitmap which allows O(log N) membership checks. Checking if a neighbor id is allowed is a fast operation (a few pointer chasing in ART or a bit lookup).
* The vector search prunes paths quickly if they lead to disallowed areas. For example, if a node has 10 neighbors but 8 of them are filtered out, the search effectively considers fewer branching paths. This might increase distance computations if the search must explore more because some close neighbors were skipped, but it still confines itself to the relevant subgraph.
* No changes to the DiskANN index are needed when filters differ between queries; the filtering is entirely dynamic and query-dependent.

**Example (hybrid query flow):** Suppose a query asks for similar images to a query image embedding, but only among those tagged as “animals”. We have:

* `DiskANNIndex` on the `embedding` column of `images` table.
* An ART index on the `tag` column (or a separate mapping of tag -> rowids).
* The query `SELECT * FROM images WHERE tag = 'animal' USING INDEX image_embed_idx(<query_vector>, K=5);`.
* Plan:

  * Bind: find `image_embed_idx` (DiskANN), note query vector and K=5, push filter `tag='animal'` in filter set.
  * Init execution: get allowed\_ids by looking up `'animal'` in tag’s index -> suppose it returns {rowid 7, 12, 15, ...}.
  * Perform DiskANN search from entry: the search will naturally traverse many images but when it tries to add a neighbor with rowid not in allowed\_ids, it skips it. Perhaps it starts at some entry that is not an animal and many of its neighbors are not animals; those get ignored, maybe the search hops to a further node that is in allowed set. Eventually, it finds the closest 5 animal images.
  * Output those rows. The filter 'tag=animal' is inherently satisfied because we only considered allowed ids. (We could still double-check the tag in output for safety).

This demonstrates that **pushdown filtering** works seamlessly, with the DiskANN index providing similarity and the external structures ensuring correctness on other predicates.

## DuckDB Extension: C++ Structs and API Usage

To implement the above integration, the LM-DiskANN index comes as a DuckDB extension with specific C++ classes and methods:

* **Index class registration:** We register a new `Index` subclass (DuckDB’s internal class for indexes) called `DiskANNIndex`. DuckDB’s catalog will contain an `IndexCatalogEntry` for each created DiskANN index, referencing this object. The `Index` interface requires methods like `InitializeScan`, `Scan`, `Append`, `Delete`, `ConstructCost` etc. We override these appropriately.

```cpp
class DiskANNIndex : public Index {
public:
    DiskANNIndex(ClientContext &context, TableCatalogEntry &table, 
                 vector<column_t> index_cols, unique_ptr<Expression> unbound_expr, 
                 uint32_t dim, uint32_t max_degree, float alpha, bool use_ternary);
    // Members:
    uint32_t dim;
    uint32_t max_degree;
    float alpha;
    bool use_ternary;
    idx_t count;
    idx_t entry_node;
    BufferManager &buf_mgr;
    block_id_t index_start_block;  // starting block id for index storage
    // maybe a mapping of node id <-> rowid if needed
    // Methods:
    // Called to build entire index:
    void InitializeIndex(const Vector &all_vectors);
    // Called on single insert (if dynamic):
    void Insert(IndexLock &lock, DataChunk &input, Vector &row_identifiers) override;
    // Called on scan:
    unique_ptr<IndexScanState> InitializeScanSinglePredicate(Transaction &txn, Value query_val, ExpressionType comparison_type) override;
    void Scan(Transaction &txn, IndexScanState &state, DataChunk &result) override;
    // ... plus any multi-predicate variants.
};
```

In this outline:

* `InitializeIndex` would bulk-build the index from all existing vectors (using either serial or parallel build as configured). This is called during `CREATE INDEX`.
* `Insert` is invoked on table insert (if the index is marked for updates). It would take the new vector from `input` and its rowid from `row_identifiers`, and run the algorithm to insert it (perhaps by queuing a build task).
* The `Scan` interface is a bit tricky: DuckDB’s standard index scan is geared towards point/range queries (like value comparisons). For DiskANN, we repurpose it: e.g., `InitializeScanSinglePredicate` might be called with a comparison like “<-> query\_vec < X” (distance threshold) or something. But more straightforward is to implement a custom TableFunction or use `IndexScanState` differently. We might override a different method or use a TableFunction for vector search as mentioned.

However, DuckDB is flexible; an alternative is implementing this purely as a table function (`vector_top_k`). But since the prompt asks for optimizer and planner hooks, we present it as an index.

* **DiskANNScanState (IndexScanState subclass):** Holds state during an index scan:

```cpp
struct DiskANNScanState : public IndexScanState {
    Vector query_vector;
    idx_t k;
    // Pre-fetched allowed rowids (e.g., from pushed filters)
    vector<row_t> allowed_ids;
    bool use_allowed;
    // Results:
    vector<row_t> result_ids;
    vector<float> result_distances;
    idx_t result_used;  // how many results have been output so far
    bool executed;
    DiskANNScanState() : use_allowed(false), result_used(0), executed(false) {}
};
```

* `executed` indicates whether we have already performed the search (to not repeat on subsequent scans).

* We populate `result_ids` and `result_distances` after performing the search once.

* **Bind/Init (DuckDB planning API):** When the planner chooses an index scan, DuckDB will call something like `index.InitializeScanSinglePredicate(transaction, query_val, comparison_type)` or a multi-predicate version. In our case, we might hijack this: if `query_val` contains the vector and `comparison_type` is a special token for ANN search, this is how we initiate. Alternatively, since `USING INDEX` might bypass typical predicate binding, we may have a custom binder.

  Let’s assume we integrate with the existing API by treating the query vector as the “value” in a single predicate scan (like we are scanning for "embedding = \[query\_vec]" in a special way). Then:

  * `InitializeScanSinglePredicate` will create a `DiskANNScanState`. It will store the query vector (the `Value` can be a BLOB or blob of floats; we convert it to our internal Vector).
  * It will also see if the transaction has any table filters (DuckDB might pass them or we have them from the `index_condition` in a multi-index scenario). If available, prepare `allowed_ids` as described (perhaps transaction or context can be used to resolve an `TableFilterSet`).
  * Return the state.

* **Scan (execution API):** DuckDB will then call `DiskANNIndex::Scan(txn, state, result_chunk)` one or multiple times to retrieve results. Our implementation will:

  * Check `state.executed`. If false, perform the DiskANN search now. We have access to `this->buf_mgr` to pin pages, etc. Also possibly need transaction for consistency (but assume MVCC not heavily considered here).
  * To perform search, we need access to the query vector from `state.query_vector`. We might need to ensure that is accessible (it could be a DuckDB Vector of type e.g. FLOAT\[]).

    * If `query_vector` is a DuckDB `Vector` (like the actual data vector type that holds e.g. list of floats), we extract its data. If it's stored as e.g. a list or blob, we parse that into a `std::vector<float>`.
  * Use `this->entry_node` and graph to find neighbors. For neighbor exploration:

    * Use `buf_mgr.Pin(index_start_block + neighbor_id)` to get neighbor block.
    * Use our distance function (with SIMD) for each neighbor.
    * Use a local data structure (e.g., `std::vector<uint64_t> visited_bitset` or so) for visited markers. Or maintain an array of bools `visited[node]`.
    * Possibly use a small max-heap for candidate neighbors, and a separate min-heap or array for current nearest list.
  * After search, fill `state.result_ids` and sort them if needed by distance (they likely already come sorted by the algorithm).
  * Mark `state.executed = true`.
  * Then produce output:

    * If the result needs to be joined later, we might just output `rowid` as a value in `result_chunk`.
    * But if this function is supposed to produce actual table rows, we do the fetch as described. The `result_chunk` has specific column layout equal to the base table’s projection list. Possibly, since it’s an index scan, DuckDB might only expect the rowid column here and then will use that to fetch (depending on how index scans are normally implemented).
    * In a typical index scan (like an ART index for a WHERE key = X query), what happens is: the index returns the row locations that match, then the execution engine fetches those from the base table. This fetch might be embedded in the index scan or as a separate step. DuckDB’s architecture often has the index scan produce the physical rowids and then a separate operation (like `CreateIndexScanPhysicalOperator` logic) fetches actual rows.
    * If DuckDB handles that automatically (like it knows this is an index on table and will call table fetch), then our `Scan` can simply set up a result vector of rowids.
    * If not automatic, we can manually fetch by calling into `table.Fetch(...)`.
  * If more results than fit in `result_chunk`, we yield in pieces; otherwise, mark scan done.

**Structs for graph search:** Inside the index, we may have a dedicated structure for performing a search:

```cpp
struct BeamSearch {
    // Reference to DiskANNIndex graph data:
    DiskANNIndex &index;
    const float *query;
    float *query_norm; // maybe precomputed norm or transformed query
    std::vector<char> visited; // boolean flags
    std::vector<idx_t> neighbors; // scratch space
    struct Candidate { idx_t id; float dist; };
    std::priority_queue<Candidate, vector<Candidate>, DistMaxCompare> candidate_max_heap;
    std::priority_queue<Candidate, vector<Candidate>, DistMinCompare> explore_min_heap;
    // ... allowed_ids reference
    BeamSearch(DiskANNIndex &idx, const float *q) : index(idx), query(q) {
        visited.resize(index.num_nodes);
    }
    void Search(idx_t entry, idx_t ef_search, std::vector<idx_t> &out_ids);
};
```

This pseudo-structure uses two heaps: one max-heap (`candidate_max_heap`) of the current found neighbors (size limited to K) and one min-heap (`explore_min_heap`) for the next node to explore (typical for HNSW search). The algorithm:

* Push entry with its distance in explore\_min\_heap.
* While explore\_min\_heap not empty:

  * Pop top (the closest unexplored node).
  * If this distance is worse than the worst in candidate\_max\_heap and the candidate\_max\_heap is full (size >= K), then break (prune further exploration).
  * Otherwise, pin that node’s block, iterate neighbors:

    * If not visited and (allowed\_ids empty or allowed\_ids.contains(neighbor\_id)):

      * Mark visited, compute dist.
      * If candidate\_max\_heap not full or dist < worst in it:

        * push neighbor in explore\_min\_heap and also push neighbor in candidate\_max\_heap (and if candidate\_max\_heap > K, pop worst from it).
  * Continue.
* Finally, out\_ids gets all ids from candidate\_max\_heap (which are the K best). Then sort by dist if needed.

The above might be more detail than needed, but it shows how filtering integrates: the check `(allowed_ids.empty() || allowed_ids.contains(neighbor))` gates the traversal.

**DuckDB buffer manager use:** The index is stored likely as a sequence of blocks. We might compute `block_id = index_start_block + node_id` (if each node is one block and contiguous). Or if multiple nodes per block, we compute offset differently. Simplicity: one node per 8KB block, then block\_id = base\_id + node\_id.

* We call `auto handle = buf_mgr.Pin(block_id); const uint8_t *node_data = handle.Ptr();`
* From `node_data`, parse the node’s neighbor count, then neighbor IDs list starting at a known offset (say 4 bytes after vector data). Because the node’s own vector is also there, if needed we could also use it (for distance computation if doing rerank, but usually we rely on compressed neighbors for search, not the exact vector).
* Unpinning could happen after processing neighbors of that node. Or we may just let the buffer manager handle caching (it won't evict while pinned).

**SIMD optimization:** If using ternary-quantized neighbors:

* Preprocessing: We might precompute the query’s values in a form easy to multiply with { -1,0,1 }. E.g., we can compute partial sums of query components for speed.
* At runtime, for each neighbor, the distance approx:
  `dist ≈ ‖query‖^2 + ‖neighbor‖^2 - 2 * <query, neighbor>` (if L2). Since neighbor’s compressed components are -1/0/1, this simplifies to base distance plus or minus some query components. We can do vectorized masked adds for components where neighbor has 1 or -1.
* Or treat -1 as 255 (if using uint8) and use XOR/bit tricks – maybe too much detail. At spec level, we just note “SIMD usage for distance calcs to neighbors’ ternary vectors”.

**Constraints and design assumptions:**

* **Vector dimensionality fixed:** All vectors in the index have the same dimension (d). This is set when the index is created. The index can enforce this by checking input vectors. There may be a maximum dimension for which we can fit data in 8KB blocks (for instance, d=1024 with R=64 and storing compressed might be borderline; typical use is d=128 to 768 range).
* **Node block size (8KB):** The block size must accommodate: vector (d \* 4 bytes if float32), neighbor IDs (R \* 4 bytes if using 32-bit ids, which supports up to \~4 billion nodes), and neighbor compressed vectors (R \* d \* 2 bits, if ternary, or some bytes if PQ). For example, d=128, R=64:

  * Vector = 128\*4 = 512 bytes
  * Neighbor IDs = 64\*4 = 256 bytes
  * Compressed vectors: 64 \* 128 \* 2 bits = 64 \* 32 bytes = 2048 bytes
  * Sum = \~2816 bytes, well under 4096. So 4KB block is fine. If d=256, that becomes \~5.5KB, so we might up block to 8KB.
* The block size can be chosen (maybe an index creation parameter or automatically the next power of two that fits the data).
* **Memory usage:** Only the entry point and small metadata are kept in memory, plus the cache of recently used blocks. All vectors and neighbor info is on disk, but thanks to the block design and buffer manager, search only loads the needed blocks on the fly.
* **Atomic operations and locks:** In build, we used `std::atomic<uint16_t> degree` per node and CAS loops to avoid locks. This requires that neighbor list arrays be preallocated to size R for each node. We placed the atomic degree in memory adjacent to neighbor list (maybe just store degree in the same block header if building in memory, but on disk final storage only needs the final degree value). The use of CAS ensures thread safety with minimal overhead. We must be careful to use `memory_order_relaxed` or `acq_rel` appropriately since there is no further ordering needed beyond atomicity here (each CAS is independent).
* **Locks:** During parallel build, global locks are not used except maybe on updating the entry point or writing to disk. During serial or single-worker build, an `omp_lock_t` or `mutex` may guard each neighbor insertion if simpler. But we favor the atomic approach for future-proofing.
* **Threading:** If parallel build is enabled, we use DuckDB’s `TaskScheduler`. For example, we create tasks for each batch of prefix-doubling and schedule them via `TaskScheduler::Schedule(task)`. DuckDB’s scheduler handles worker threads. Alternatively, we use `std::thread` but better to integrate with DuckDB’s own pool to avoid oversubscription.
* **No graph changes for queries:** At query time, the graph is read-only. We do not modify entry point or anything during search. If dynamic inserts happen concurrently with a search, we may need a latch or versioning to ensure the search either sees the graph up to a point or the insert waits. Simpler is to lock out writes during read queries or use an optimistic approach if the index supports it. Given DuckDB’s typical usage (mostly append-only or batch loads), we assume either no concurrent modification or it’s handled by transactional versions (maybe rebuild index at commit for new data).

In summary, this technical design enables LM-DiskANN to function as a first-class index in DuckDB, offering efficient approximate nearest neighbor search on disk-based vectors with tight integration into SQL queries. The graph index build and search algorithms from LM-DiskANN (greedy search, robust prune, prefix-doubling parallel build) are implemented in C++ leveraging atomic operations for correctness. DuckDB’s optimizer can incorporate the index through standard interfaces, and crucially, **all additional query filtering is handled outside the index data** by using DuckDB’s own filtering mechanisms (ART indexes, bitmaps, etc.), preserving the purity and generality of the vector index. This design ensures that hybrid queries (vector similarity + structured filters + text search) are executed correctly and efficiently, combining the strengths of the DiskANN graph with DuckDB’s powerful execution engine.
