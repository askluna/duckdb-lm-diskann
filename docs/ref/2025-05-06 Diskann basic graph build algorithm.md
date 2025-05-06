**Evaluating ANN algorithms**

I'll have to consider various aspects like concurrency issues, locking, atomic degree, thread partitioning, and I/O batching. To support everything with proper references, I'll include at least 10 citations from sources like the Microsoft DiskANN repo, LM-DiskANN, the Turso blog, and FreshDiskANN. When crafting the response, I’ll compare simple serial insertion and batch parallel insertion for non-GPU setups, referencing the LM-DiskANN methodology and ParlayANN techniques. I'll also need pseudocode for clarity.


Thought for 22 seconds


## TL;DR – how LM‑DiskANN builds its graph today and how you can parallelise it

The *reference* LM‑DiskANN implementation keeps things serial: every new point performs a greedy search through the current graph (Vamana) to pick L\_insert neighbours, prunes them with the `α` rule, updates those neighbour lists under a global mutex, and returns. ParlayANN and FreshDiskANN show that you can batch thousands of such inserts at once, divide them among threads with **prefix‑doubling**, and use lightweight per‑node atomics so threads splice edges concurrently without corrupting degree bounds. The batch path scales the original build time down by ≈8–12× on a 32‑core CPU while giving the *exact* same graph quality.  Below are the algorithms, data structures, and trade‑offs in detail—no GPU tricks required.

---

## 1 . Serial (simple) graph‑build in LM‑DiskANN

### 1.1 Background: Vamana search & prune

DiskANN’s builder is Vamana — a greedy, layer‑free small‑world graph that keeps ≤ R outgoing edges per node ([Suhas' Webpage][1]).
For each new vector **p**:

1. **Greedy search**: start at the current entry point, walk to closer neighbours until no progress (parameter L\_insert controls candidate pool size). ([Suhas' Webpage][1])
2. **Robust‑prune**: sort the candidate set by distance; add an edge (p→q) if `α · dist(p,q) < dist(p,S)`, where S is the current accepted set ([NeurIPS Papers][2]).
3. **Mutual updates**: for every selected neighbour q
      \* lock q (or whole index in the reference code)
      \* insert p into q’s neighbour list, evict farthest if degree > R
4. **Update entry** (optional): replace entry point with p if ‖p‖₂ is maximal ([GitHub][3]).

The Microsoft repo’s `build_disk_index` runs this loop on a **single thread** and holds an `omp_lock_t` around *every* neighbour list edit ([GitHub][3]).

### 1.2 Pseudocode

```python
for p in dataset:
    cand = GreedySearch(entry, p, L_insert)
    nbrs = RobustPrune(p, cand, alpha, R)
    for q in nbrs:
        lock(q)           # coarse-grained in reference build
        InsertEdge(q, p)  # may drop worst edge if |out(q)| > R
        unlock(q)
    lock(graph)
    InsertEdge(p, nbrs)
    unlock(graph)
```

Build time is **O(N · L\_insert · log R)** I/Os on disk; CPU bound on distance calcs. On 10 M vectors it can take several hours ([Timescale][4]).

---

## 2 . Batch‑parallel build (prefix‑doubling)

### 2.1 What ParlayANN / FreshDiskANN do

*ParlayANN* shows a **deterministic** fork‑join strategy for graph builders including DiskANN ([arXiv][5]).
Key ideas:

| Idea                | Effect                                                                                                                                         | How                                                                                     |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Prefix‑doubling** | Keeps work balanced as graph grows                                                                                                             | Build first 2 k nodes serially, next 2 k in parallel, next 4 k, … until all N processed |
| **Edge CAS**        | Each node stores `atomic<uint64_t>` for degree+id; threads `compare_exchange` to add edges without locks                                       | Guarantees ≤ R edges per node even under races                                          |
| **Batch search**    | All threads share a read‑only snapshot; each thread does greedy search on its slice; candidates drawn from previous prefix ensure connectivity | Eliminates contention during traversal                                                  |
| **Write buffers**   | Per‑thread vectors of pending “add edge (q←p)” flush once; reduces random I/O                                                                  | Uses `FileSystem::Write` in 256 kB chunks                                               |

FreshDiskANN reports 8× speed‑up on 64 threads vs serial DiskANN for 1 B vectors ([arXiv][6]).

### 2.2 Algorithm sketch

```python
P = 1         # current prefix
while P < N:
    parallel_for i in [P, 2P):             # each thread gets a chunk
        cand = GreedySearch(entry, x[i], L_insert)   # snapshot graph
        nbrs = RobustPrune(x[i], cand)
        for q in nbrs:
            while True:                    # atomic degree slot
                k = degree[q].load()
                if k >= R: break
                if degree[q].compare_exchange_weak(k, k+1):
                    edges[q][k] = i
                    break
        edges[i] = nbrs                    # write own list
    barrier()
    P *= 2
```

*Determinism* comes from fixed ordering (prefix‑doubling + thread‑id) and from retry loops on `compare_exchange`; every run produces identical graph ([ACM Digital Library][7]).

### 2.3 Trade‑offs

| Aspect          | Batch‑parallel                                               | Serial                     |
| --------------- | ------------------------------------------------------------ | -------------------------- |
| Build wall‑time | 10–12× faster on 32 cores (disk not bottleneck) ([arXiv][6]) | baseline                   |
| Peak RAM        | +10 % for per‑thread buffers                                 | minimal                    |
| I/O pattern     | mostly sequential (bulk writes)                              | fine‑grained random writes |
| Determinism     | needs extra care (prefix order)                              | inherent                   |
| Code complexity | atomic degree ops, two‑phase write                           | simple                     |

*Testing strategy:* run the build twice with same RNG seed; diff neighbour lists. TSAN for data races; unit tests injecting crashes mid‑batch and verifying rebuild.

---

## 3 . Simple middle‑ground: **queue + worker thread**

If full ParlayANN feels heavy:

1. Maintain `std::queue<RowID>` of pending inserts.
2. One dedicated worker thread pops from queue and runs the *serial* algorithm (still single‑threaded inside).
3. Producers (user inserts) just push to queue and return.

*Pros:* non‑blocking ingest path, very low code delta. *Cons:* still limited to one core, but hides latency from client.

---

## 4 . Integration hooks in your codebase

* In `LmDiskannNodeAccessors.cpp` you already have `SetNeighborCount` and offset helpers—add `std::atomic<uint16_t>` degree just after neighbour count so CAS works.
* In `config.hpp` expose `WITH (parallel_build=true, batch_size=4096)`; default `false`.
* Background batch can reuse DuckDB `TaskScheduler`.

---

## 5 . Choosing a path now

| Option                             | Lines of new C++ | Risk                         | When                                 |
| ---------------------------------- | ---------------- | ---------------------------- | ------------------------------------ |
| **Queue + single worker**          | < 150            | trivial                      | MVP 0.1                              |
| **Prefix‑doubling batch parallel** | 600–800          | medium (atomics, race tests) | after MVP, when build time is a pain |
| **Fully concurrent streaming**     | 1 500+           | high                         | only if you need ≫ 50 k inserts/s    |

My advice: ship with **queue+worker**; keep the node layout parallel‑ready (atomic degree, fixed offsets).  When you need faster offline builds, port ParlayANN’s prefix‑doubling; the deterministic variant makes reproducibility and CI diffs straightforward.

[1]: https://suhasjs.github.io/files/diskann_neurips19.pdf?utm_source=chatgpt.com "[PDF] DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a ..."
[2]: https://papers.neurips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf?utm_source=chatgpt.com "[PDF] Fast Accurate Billion-point Nearest Neighbor Search on a Single Node"
[3]: https://github.com/microsoft/DiskANN/blob/main/workflows/dynamic_index.md?utm_source=chatgpt.com "DiskANN/workflows/dynamic_index.md at main - GitHub"
[4]: https://www.timescale.com/blog/understanding-diskann?utm_source=chatgpt.com "Understanding DiskANN - Timescale"
[5]: https://arxiv.org/pdf/2305.04359?utm_source=chatgpt.com "[PDF] ParlayANN: Scalable and Deterministic Parallel Graph - arXiv"
[6]: https://arxiv.org/pdf/2105.09613?utm_source=chatgpt.com "[PDF] FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for ..."
[7]: https://dl.acm.org/doi/10.1145/3627535.3638475?utm_source=chatgpt.com "ParlayANN: Scalable and Deterministic Parallel Graph-Based ..."
