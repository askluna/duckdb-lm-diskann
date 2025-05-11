# **Understanding and Implementing LM-DiskANN for Billion-Scale Vector Search**

## **1. Introduction: The Challenge of Billion-Scale ANN Search**

### **1.3. DiskANN: Bridging the Gap**

To overcome the memory limitations of purely RAM-based ANN solutions, Microsoft Research developed DiskANN.1 DiskANN is a graph-based indexing and search system specifically designed to leverage Solid-State Drives (SSDs) to handle datasets far exceeding available RAM capacity. The core idea is to store the bulk of the index structure (the Vamana graph) and the full-precision vectors on SSD, while strategically using a limited amount of RAM to guide the search and minimize costly disk I/O operations.9

DiskANN demonstrated that, contrary to prevailing wisdom at the time, SSD-based indices could achieve the three critical desiderata for large-scale ANN search: high recall, low query latency, and high density (points indexed per node).9 On billion-point datasets like SIFT1B, DiskANN was shown to serve thousands of queries per second with mean latencies under 5 milliseconds and achieving over 95% 1-recall@1, using only 64GB of RAM and a commodity SSD.9 This represented a 5-10x increase in index density compared to state-of-the-art in-memory graph methods like HNSW and NSG for similar performance targets.9

### **1.4. Introducing LM-DiskANN**

While DiskANN significantly reduces RAM requirements compared to fully in-memory solutions, it still necessitates keeping compressed representations of the *entire* dataset in RAM to guide the search effectively.6 For extremely large datasets or environments with severe memory constraints (e.g., edge devices, mobile applications, multi-tenant systems with many small databases), even this compressed representation can pose a challenge.6

This motivates the development of LM-DiskANN (Low Memory DiskANN).6 LM-DiskANN is a variant of DiskANN specifically engineered to operate with a minimal memory footprint. Its core innovation lies in modifying the on-disk data layout to store necessary routing information (compressed neighbor vectors) directly within each node's block on disk, thereby eliminating the need for the large global cache of compressed vectors in RAM.6 This report provides a comprehensive technical deep dive into the principles, architecture, and implementation considerations of DiskANN and, specifically, LM-DiskANN, aimed at enabling a Staff Engineer to understand and implement this low-memory variant.

## **2. Understanding DiskANN Fundamentals**

DiskANN builds upon a sophisticated graph-based indexing algorithm called Vamana and adapts it for efficient operation in a disk-based environment. Understanding these core components is crucial before delving into the specifics of LM-DiskANN.

### **2.1. The Vamana Graph Algorithm**

Vamana serves as the foundational graph construction algorithm for DiskANN.1 Unlike HNSW, which builds a hierarchical graph starting sparsely and adding edges/layers 13, Vamana constructs a "flat" (single-layer) graph using an iterative refinement process that starts with a dense random graph and strategically prunes edges.29 This flat structure proves advantageous for disk-based storage, as neighbor information for a node can be laid out more predictably for efficient retrieval.29 The construction relies on two key procedures: Greedy Search and Robust Pruning.1

**Vamana Construction - Step-by-Step:**

The Vamana indexing algorithm (Algorithm 3 in the original DiskANN paper 9) builds a directed graph G where each vertex corresponds to a data point and has an out-degree of at most R (a configurable parameter).

1. **Initialization:** The process begins by creating a random directed graph where each node initially has approximately R outgoing edges, connecting it to randomly chosen points in the dataset.9 A central point, the medoid s, is identified to serve as the starting point for searches during construction.9
2. **Iteration:** The algorithm iterates through the data points p in a random order (permutation σ).9 For each point p=σ(i):

- **Greedy Search (Conceptual Algorithm 1):** A beam search (or greedy search) is initiated from the medoid s to find candidate neighbors for the current point p.1 This search maintains a list (or priority queue) of candidate nodes, ordered by their distance to p. It iteratively explores the most promising unvisited nodes (those closest to p), adding their neighbors to the candidate list while keeping the list size bounded (by parameter L, the search list size).1 The set V of all nodes visited during this search forms the pool of potential neighbors for p.
- **Robust Pruning (Algorithm 2):** The RobustPrune(p, V, \alpha, R) procedure refines the neighbor set for p.9 It selects up to R out-neighbors from the candidate set V (which includes p's existing neighbors). The selection prioritizes nodes closest to p. Crucially, it employs a pruning strategy based on the parameter α≥1. A point p′ is pruned from V if a selected closer neighbor p∗ exists such that α×d(p∗,p′)≤d(p,p′).9 This step ensures that the chosen neighbors are not only close but also diverse, preventing the graph from becoming overly clustered with only very short edges. By relaxing the pruning condition (α>1), it helps retain some longer-range edges, which are vital for efficient navigation across the graph and reducing the graph diameter.1
- **Edge Addition & Maintenance:** New directed edges are added from p to its newly selected R neighbors. To improve connectivity and search robustness, *backward* edges are also added from these neighbors back to p.9 Adding a backward edge might cause a neighbor j's out-degree to exceed R. If this happens, RobustPrune is called again on node j (considering p as a candidate neighbor) to trim its neighbor list back down to R, ensuring the degree constraint is maintained throughout the process.9

1. **Multiple Passes:** The entire iteration process (Greedy Search, Robust Pruning, Edge Addition) is typically performed twice over the dataset. The first pass uses α=1 for more aggressive initial pruning, while the second pass uses the user-specified α>1 to refine the graph structure and better preserve long-range connections.9

**Vamana Graph Properties:**

The resulting Vamana graph is a single-layer (flat) structure.29 Its construction aims to achieve a low graph diameter (meaning fewer hops are typically needed to reach any node from any other node) and a controlled maximum out-degree R for each node.9 The robust pruning mechanism helps ensure a balance between local connectivity (short edges to nearby points) and global reachability (longer-range edges), which is essential for efficient greedy search performance.13 The flat structure is inherently more amenable to disk storage compared to hierarchical structures like HNSW. In a flat graph, the neighbors for any node are predetermined and can be stored contiguously, potentially allowing their retrieval with a single disk read. Hierarchical searches might necessitate accessing different parts of the index corresponding to different layers, potentially leading to more random disk accesses.29

### **2.2. DiskANN Architecture: Adapting Vamana for Disk**

The primary challenge in adapting a graph-based index like Vamana to disk is mitigating the performance impact of slow, random disk I/O operations.9 Even fast SSDs have access latencies orders of magnitude higher than RAM, and random reads are particularly detrimental to throughput.9 DiskANN tackles this through a hybrid storage architecture and optimized disk layout.9

**Hybrid Storage Model:**

DiskANN employs a two-tiered storage strategy:

- **On Disk (SSD):** The main Vamana graph structure (specifically, the list of neighbor IDs for each node) and the full-precision, original data vectors are stored persistently on an SSD.9 This allows the system to handle datasets much larger than available RAM.
- **In Memory (RAM):** Crucially, DiskANN keeps *compressed* versions of *all* dataset vectors in RAM.6 Product Quantization (PQ) is commonly used for this compression.9 Additionally, a cache is typically maintained in RAM to hold recently accessed graph nodes (neighbor lists) and their full-precision vectors, reducing redundant disk reads.19

**Disk Layout Optimization:**

To minimize the number of I/O operations during search, DiskANN optimizes the layout of data on the SSD. For each node (vector) in the dataset, its list of neighbor IDs and its corresponding full-precision vector are stored *contiguously* on the disk.9 This collocation is critical: when the search algorithm decides to "visit" a node (based on approximate distance calculations using the in-memory compressed data), it can fetch both the node's neighbors (needed to continue the graph traversal) and its full-precision vector (potentially needed for later re-ranking or exact distance checks) with a single, sequential disk read operation, or at least minimize the I/O cost.9

This hybrid storage model combined with the optimized disk layout is DiskANN's key architectural innovation. The small, fast in-memory compressed vectors act as a guide for navigating the large on-disk graph. They allow for rapid, approximate distance calculations to determine the most promising paths to explore during the search. Disk I/O is deferred until a node is deemed highly relevant based on these approximate calculations. This strategy drastically reduces the number of random disk reads required compared to a naive approach of storing the entire graph on disk without an in-memory guide, enabling low-latency search on billion-scale datasets.9

### **2.3. The DiskANN Search Process (Beam Search)**

DiskANN employs a beam search algorithm, a variant of the greedy search used during Vamana construction, to find the approximate nearest neighbors for a query vector xq.9 The process leverages the hybrid storage model:

**Initialization:**

1. The search begins at one or more predefined entry points in the graph, typically the medoid(s) identified during index construction.2
2. Two main data structures are maintained in memory:

- A priority queue L (the candidate list) storing pairs of (distance, node_id), ordered by distance to the query xq. Its size is bounded by a search list parameter (often also denoted L).
- A set V (the visited set) storing the IDs of nodes whose neighbors have already been explored.

**Iteration:**

The search proceeds iteratively:

1. **Candidate Selection:** Identify the top W (beam width) candidate nodes from the priority queue L that have not yet been visited (i.e., their IDs are not in V). The selection is based on the *estimated distances* calculated using the **in-memory compressed vectors**.9 This step is fast as it only involves RAM access and approximate computations.
2. **Mark Visited:** Add the IDs of these W selected nodes to the visited set V.
3. **Disk Fetch:** Issue disk I/O requests to retrieve the data blocks for these W nodes from the SSD. Due to the optimized layout, each read fetches both the **full neighbor list** and the **full-precision vector** for the corresponding node.9 These fetched nodes might be placed in the RAM cache.
4. **Neighbor Exploration:** For each of the W nodes fetched, iterate through its retrieved neighbors:

- If a neighbor's ID is not in the visited set V:

- Calculate the *estimated distance* between the query xq and the neighbor using the **in-memory compressed vectors**.9
- Insert the pair (estimated_distance, neighbor_id) into the priority queue L.
- If L exceeds its size limit, prune the farthest candidate.

1. **Termination:** Repeat the iteration until a stopping condition is met. This could be when the search exhausts all promising candidates in L, a fixed number of iterations is reached, or a distance threshold is met.

**Re-ranking (Optional but Recommended):**

Once the beam search terminates, the priority queue L contains the top candidates found based on approximate distances. To enhance accuracy, a re-ranking step is often performed.9 The full-precision vectors for the top-k candidates in L (which were retrieved from disk during the search and are likely in the cache) are used to compute their *exact* distances to the query xq. The final result set is then ordered based on these exact distances. This mitigates the precision loss introduced by using compressed vectors during the main search phase.9

The beam width W is a crucial parameter influencing the latency-recall trade-off. A larger W allows the search to explore more paths simultaneously at each step, potentially improving the chances of finding the true nearest neighbors (higher recall). However, it also increases the number of disk I/Os performed per iteration, thereby increasing query latency. Conversely, a smaller W reduces disk I/O but increases the risk of the search getting trapped in a suboptimal region of the graph (local minimum), potentially lowering recall.9 Tuning W alongside the Vamana graph parameters (R, L, α) is essential for achieving the desired performance balance.

## **3. Deep Dive into LM-DiskANN**

While standard DiskANN offers a significant improvement over purely in-memory methods for large datasets, its requirement to store the entire compressed dataset in RAM can still be a bottleneck. LM-DiskANN addresses this specific challenge.

### **3.1. Motivation: Addressing DiskANN's Memory Footprint**

Standard DiskANN's architecture mandates that compressed representations (e.g., PQ codes) of *all* vectors in the indexed dataset reside in main memory.6 This in-memory structure serves as the crucial guide for the beam search, enabling fast approximate distance calculations to navigate the graph efficiently before accessing the disk. However, for billion-scale datasets, even these compressed representations can consume tens or hundreds of gigabytes of RAM.9 This memory footprint can be prohibitive for:

- Deployments on commodity hardware with limited RAM.
- Memory-constrained environments like mobile or edge devices.25
- Multi-tenant systems aiming to host numerous independent indices cost-effectively.26

LM-DiskANN was specifically designed to overcome this limitation by eliminating the need for this large, global in-memory compressed dataset.6

### **3.2. Core Concept: Storing Compressed Neighbor Information On-Disk**

The fundamental innovation of LM-DiskANN is a change in the on-disk data layout.6 Instead of relying on a global in-memory cache of compressed vectors, LM-DiskANN stores the necessary compressed information for routing *locally* within each node's data block on the disk. Specifically, for each node p, its corresponding block on disk contains not only its own full vector and the IDs of its neighbors, but also the **Product Quantization (PQ) compressed representations of those neighbors' vectors**.6

The "LM" in LM-DiskANN stands for **Low Memory Footprint** 6, directly reflecting its primary design goal. It is important to note that LM-DiskANN, as described in the source paper 6 and implemented in systems like Turso's libSQL 25, does not typically involve an additional "learned model" for routing beyond the standard training required for the Product Quantization compression itself. The routing decisions during search are still based on distance calculations, albeit using the locally stored compressed neighbor data.

### **3.3. LM-DiskANN Node Structure and Disk Layout**

The on-disk structure for a single node in an LM-DiskANN index is designed to be self-contained for routing purposes. A typical layout for a node's data block might be 6:

1. **Node ID:** Unique identifier for the data point.
2. **Full-Precision Vector:** The original, uncompressed vector for this node.
3. **Neighbor IDs:** A list containing the unique IDs of the node's neighbors in the Vamana graph (up to the maximum degree R).
4. **Compressed Neighbor Vectors:** A list containing the PQ-compressed representations of the vectors corresponding to the Neighbor IDs listed above.
5. **(Optional) Padding:** Data might be padded to align the block to a specific size (e.g., 4KB, as suggested in the Turso implementation 25), which can optimize disk I/O.

This contrasts sharply with the standard DiskANN node layout, which typically only stores the Node ID, Full Vector, and Neighbor IDs on disk, relying on the separate in-memory cache for compressed vectors.

This self-contained node structure is the key to LM-DiskANN's low memory operation. When a node's block is read from disk into memory during a search traversal, the system immediately possesses all the information needed to evaluate the next steps: the IDs of the neighbors *and* their compressed vector representations. Approximate distances to all neighbors can be calculated using only the data within this single loaded block. There is no need to perform lookups into a large, global in-memory table of compressed vectors. This "local decision" capability allows the search to proceed efficiently with a minimal RAM footprint, primarily needing memory only for the active search state (candidate list, visited set) and potentially a small cache of recently loaded node blocks.6

### **3.4. Role of Product Quantization (PQ) in LM-DiskANN**

Product Quantization (PQ) is the chosen vector compression technique used within the LM-DiskANN node structure to store neighbor information compactly.6 PQ is well-suited for this task due to its ability to achieve high compression ratios while enabling fast, approximate distance calculations.32

Briefly, PQ operates as follows 32:

1. **Subvector Division:** A d-dimensional vector x is divided into m disjoint subvectors, each of dimension d/m.
2. **Codebook Learning:** For each of the m subvector spaces, a separate codebook (dictionary) of k∗ "centroid" vectors is learned using k-means clustering on a representative training dataset.32 Typically, k∗ is set to 256, allowing each centroid ID to be represented by 8 bits (code_size=8).
3. **Encoding:** To compress a vector x, each of its m subvectors is replaced by the ID (index) of its nearest centroid within the corresponding codebook.32
4. **Compressed Code:** The resulting compressed representation is a sequence of m IDs, requiring significantly less storage than the original vector (e.g., m bytes if code_size=8).32

In the context of LM-DiskANN, PQ enables efficient **Asymmetric Distance Computation (ADC)**.33 During the search, the distance between the full-precision query vector q and the *compressed* representation of a neighbor vector (stored within the current node's block) can be estimated quickly. This typically involves pre-calculating distances between the query's subvectors and the relevant centroids in the PQ codebooks, then summing these pre-calculated partial distances based on the neighbor's compressed code. This ADC is much faster than computing the distance between two full-precision vectors, making the graph traversal efficient even though neighbor vectors are only available in compressed form initially.5

A critical aspect is that PQ is a data-dependent compression method. The quality of the codebooks, and thus the accuracy of the compressed representation and distance estimations, depends heavily on the training data used.35 Training must be performed on a dataset that accurately reflects the distribution of the vectors being indexed.

## **4. Comparative Analysis: LM-DiskANN vs. Standard DiskANN**

Understanding the specific differences, advantages, disadvantages, and trade-offs between LM-DiskANN and standard DiskANN is crucial for making informed implementation decisions.

### **4.1. Key Architectural and Algorithmic Differences**

The primary distinction lies in how and where compressed vector information is stored and accessed:

- **Memory Usage:** This is the most significant difference. LM-DiskANN achieves a drastically lower RAM footprint because it does not require the entire dataset's compressed vectors to be held in memory. Its memory usage is primarily determined by the search state (candidate list, visited set, beam width) and any optional caching of disk blocks.6 Standard DiskANN's RAM usage scales with the dataset size, as it needs memory proportional to N×(compressed vector size) plus cache.9
- **Disk Usage:** LM-DiskANN increases the amount of disk space required *per node*. Each node block must store not only the node's own vector and neighbor IDs but also the compressed representations of all its neighbors.25 This introduces redundancy, as the compressed vector for a given point p will be stored in the blocks of all nodes that list p as a neighbor. Standard DiskANN has lower disk usage per node, storing only the graph structure and full vectors on disk, with the compressed vectors residing solely in RAM.9
- **Node Structure:** As detailed in Section 3.3, the on-disk node layout differs fundamentally. LM-DiskANN includes compressed neighbor vectors locally 6, while standard DiskANN does not.
- **Search Algorithm:** The core beam search logic operating on the Vamana graph remains similar. However, the source of the compressed vectors used for distance estimation during traversal changes. LM-DiskANN retrieves them from the *currently loaded node block* on disk 6, enabling local routing decisions. Standard DiskANN retrieves them from the *global in-memory cache* of compressed vectors.9
- **I/O Pattern:** Both algorithms strive to minimize random disk I/O by leveraging the Vamana graph's properties and contiguous node layouts. LM-DiskANN reads slightly larger blocks per node visit due to the inclusion of compressed neighbor data. Standard DiskANN reads smaller node blocks but relies entirely on the fast RAM access for compressed vector lookups needed during navigation.

### **4.2. Pros and Cons Evaluation**

Evaluating the strengths and weaknesses of each approach:

**LM-DiskANN:**

- **Pros:**

- **Significantly Reduced RAM Footprint:** Enables indexing and searching vastly larger datasets on hardware with limited memory, or supports higher density of smaller indices in multi-tenant scenarios.6 Makes disk-based ANN feasible on edge/mobile devices.25
- **Potentially Simpler System:** Eliminates the need to manage and maintain the large, global in-memory cache of compressed vectors present in standard DiskANN.

- **Cons:**

- **Increased Disk Storage:** The redundancy of storing compressed neighbor vectors increases the overall disk space requirement for the index.25
- **Higher I/O Bandwidth per Node:** Reading larger node blocks might consume more disk bandwidth per visited node, potentially impacting latency if bandwidth is the bottleneck.
- **Update Complexity:** Modifying graph edges (due to insertions or deletions) requires rewriting the larger LM-DiskANN node blocks. Deleting a node necessitates finding all nodes that point to it and rewriting their blocks to remove the reference and the associated compressed vector. While updates are challenging for any disk-based graph index 21, the larger block size in LM-DiskANN could exacerbate the write amplification issue. Strategies like those in FreshDiskANN 17 might be adaptable but need to account for the node structure.
- **Dependence on Local PQ Quality:** The accuracy of routing decisions relies entirely on the quality of the PQ compression stored within the node block. Errors in local distance estimation cannot be easily compensated by other information during that step.

**Standard DiskANN (e.g., Microsoft Implementation** 38**):**

- **Pros:**

- **Lower Disk Usage per Node:** Stores less data per node on disk, leading to a smaller overall index size on persistent storage.
- **Potentially Faster Node Reads:** Reading smaller blocks from disk might be slightly faster if disk latency per block is dominant.
- **Mature and Optimized:** Benefits from extensive development, optimization, and community support via established repositories.38

- **Cons:**

- **Significant RAM Requirement:** The need to store all compressed vectors in RAM fundamentally limits scalability based on available memory, making it expensive or infeasible for the largest datasets on single nodes.9

### **4.3. Performance Trade-offs Analysis**

- **Latency vs. Recall:** Both LM-DiskANN and standard DiskANN operate on the fundamental ANN trade-off curve, adjustable via parameters like beam width (W or search list size L), graph degree (R), and Vamana's α.9 Increasing parameters like L or W generally improves recall but increases latency (more computations, more I/O). The LM-DiskANN paper asserts that it can achieve similar recall-latency performance to standard DiskANN while using significantly less memory 6, suggesting the architectural change doesn't inherently degrade core search efficiency if tuned properly.
- **Build Time:** Constructing Vamana graphs is computationally intensive, involving multiple passes, numerous distance calculations, and graph manipulations.13 Both methods face significant build times for large datasets (potentially days for billion-scale indices 39). LM-DiskANN adds the overhead of PQ-encoding neighbor vectors during the build or update phase. Standard DiskANN implementations often employ partitioning and merging strategies to build indices for datasets too large to fit in RAM even during construction.1
- **Throughput (Queries Per Second - QPS):** QPS is inversely related to average query latency and also depends on hardware capabilities (CPU cores for parallel processing, SSD IOPS and bandwidth) and the ability to batch queries.9 LM-DiskANN's primary impact on QPS stems from its potential latency profile (aiming to be similar to DiskANN) and its lower RAM usage, which might allow deploying more instances or handling larger datasets per server, thus potentially influencing overall system throughput positively in resource-constrained scenarios.
- **Memory vs. Disk:** This is the defining trade-off. LM-DiskANN explicitly prioritizes minimizing RAM at the cost of increased disk space usage per node.25 Standard DiskANN prioritizes minimizing disk usage per node at the cost of significant RAM consumption for the global compressed vector cache.

### **4.4. Table: Feature/Metric Comparison (DiskANN vs. LM-DiskANN)**

This table summarizes the key distinctions for implementation planning:

| **Feature/Metric**     | **Standard DiskANN**                         | **LM-DiskANN**                                               | **Rationale / Notes**                                        |
| ---------------------- | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Primary Goal**       | Scalable ANN on Disk (vs. RAM-only)          | Minimize RAM Footprint (vs. Standard DiskANN)                | Both target disk-based search; LM-DiskANN optimizes further for memory. |
| **RAM Usage**          | High (Stores all compressed vectors + cache) | Low (Stores only working set/cache, no full compressed dataset) | LM-DiskANN's main advantage. 6                               |
| **Disk Usage/Node**    | Moderate (Full Vector + Neighbor IDs)        | High (Full Vector + Neighbor IDs + Compressed Neighbors)     | LM-DiskANN stores redundant compressed data. 25              |
| **On-Disk Node Data**  | Full Vector, Neighbor IDs                    | Full Vector, Neighbor IDs, Compressed Neighbor Vectors       | Defines the core architectural difference. 6                 |
| **Search Guidance**    | Global In-Memory Compressed Vectors          | Local On-Disk Compressed Neighbor Vectors (read with node)   | How approximate distances are calculated during traversal. 6 |
| **I/O per Node Visit** | Reads Node Block (Vector + Neighbor IDs)     | Reads Node Block (Vector + Neighbors + Compressed Neighbors) | LM-DiskANN reads potentially larger blocks.                  |
| **Update Complexity**  | High (Graph updates on disk)                 | Potentially Higher (Larger nodes to rewrite, more complex updates) | Both benefit from strategies like FreshDiskANN 17, but node size matters. |
| **PQ Requirement**     | Yes (For in-memory compressed dataset)       | Yes (For on-disk compressed neighbors within nodes)          | Both rely on PQ for compression and fast distance estimation. |
| **Key Parameters**     | R, L, alpha, W, PQ params                    | R, L, alpha, W, PQ params                                    | Similar tuning parameters related to Vamana, Search, and PQ. 25 |

### **4.5. Implementation Gotchas and Tuning Considerations**

Implementing and tuning LM-DiskANN requires careful attention to several factors:

- **Hyperparameter Sensitivity:** Performance metrics (recall, latency, build time, QPS) are highly dependent on the chosen hyperparameters. These include Vamana graph parameters (Graph Degree R, Search List Size L during build, Robust Pruning factor α), Search parameters (Beam Width W, Search List Size L during query), and Product Quantization parameters (number of subvectors m, bits per subvector code_size).25 Extensive empirical tuning on representative workloads is almost always necessary. Start with common values (e.g., R=32−128, L=50−200, α=1.2−1.4, W=4−16, PQ code_size=8) and iterate.
- **PQ Training Data:** The effectiveness of PQ, and therefore the accuracy of LM-DiskANN's routing decisions, hinges on the quality of the learned codebooks. These must be trained on a dataset that accurately reflects the distribution of the vectors being indexed.35 Using unrepresentative training data will lead to poor compression, inaccurate distance estimations, and ultimately lower search recall. The size of the training set is also important; recommendations often involve a multiple of the number of centroids, e.g., max(1000*nlist, 2^code_size * 1000) for IVF or 2^code_size * 1000 for graph-based indices like Vamana/HNSW when using PQ.35
- **PQ Parameter Choice (m, code_size):** The number of subvectors, m, must be a divisor of the vector dimension d. The code_size (typically 8 bits, yielding 256 centroids per subquantizer) determines the precision of each subvector's representation. Increasing m or code_size generally improves compression accuracy (better recall) but increases the size of the compressed codes.32 In LM-DiskANN, this directly impacts the size of each node block on disk, potentially increasing I/O time and disk usage. A common strategy is to fix code_size at 8 and tune m to balance compression ratio (memory/disk usage) and recall.35
- **Disk I/O Bottlenecks:** Ultimately, the performance of any disk-based index is constrained by the underlying storage hardware's IOPS, latency, and bandwidth.9 Efficient I/O handling in the implementation is paramount. This includes appropriate buffering strategies, potentially aligning data structures and I/O requests to the disk's physical sector or page size (e.g., 4KB), and considering asynchronous I/O patterns to overlap computation with disk access.38
- **Data Distribution Drift (OOD Queries):** Like most data-dependent ANN indices, Vamana (and thus DiskANN/LM-DiskANN) can suffer performance degradation (lower recall for a given latency) if the distribution of query vectors differs significantly from the distribution of the indexed vectors.5 This "Out-Of-Distribution" (OOD) problem is common in multi-modal search (e.g., text query on image index). While LM-DiskANN doesn't specifically address this, awareness is crucial. Specialized variants like OOD-DiskANN exist that use sample OOD queries during index construction to improve performance for such scenarios.5
- **Update Handling:** Implementing efficient dynamic updates (insertions and deletions) in a disk-based graph index is notoriously complex.21 Naive approaches can lead to excessive I/O overhead and degraded graph quality. The LM-DiskANN paper mentions support for dynamic insertions and deletions 6, likely involving incremental Vamana-style updates. However, the larger node size could make updates more costly than in standard DiskANN. If frequent updates are required, careful consideration of update strategies, potentially drawing inspiration from FreshDiskANN (which uses a hybrid SSD/in-memory approach for updates 12), is necessary. Static indices or bulk updates are significantly simpler to implement.
- **Cold Start Performance:** When the system starts, caches (both OS page cache and any application-level node cache) will be cold. The initial queries might experience higher latency as required node blocks are read from disk for the first time. LM-DiskANN's advantage here is that it avoids the potentially lengthy initial load time required by standard DiskANN to populate its large in-memory compressed vector cache.

## **5. Implementing LM-DiskANN in C++**

This section outlines key considerations and components for implementing LM-DiskANN using C++, targeting a Staff Engineer level understanding.

### **5.1. Essential Data Structures**

Careful data structure design is crucial for performance and correctness.

- **In-Memory Structures:**

- QueryVector: Class or struct holding the query vector data (e.g., std::vector<float>) and potentially its PQ-quantized subvectors for efficient distance calculation.
- CandidateNode: Simple struct/class to store (float distance, uint32_t node_id) pairs, used within the priority queue. Needs appropriate comparison operators.
- CandidateQueue: A priority queue (e.g., std::priority_queue wrapping a std::vector<CandidateNode>) to manage the search candidate list L, ordered by distance (min-heap).
- VisitedSet: An efficient mechanism to track visited node IDs during search. std::unordered_set<uint32_t> is viable for moderate scale, but for very large searches or tight memory constraints, a Bloom filter could be considered to trade accuracy for space.
- NodeCache (Optional): An LRU (Least Recently Used) cache (e.g., using std::list and std::unordered_map) to store recently loaded DiskNode objects in RAM, reducing redundant disk reads for frequently accessed nodes. Cache size is a tunable parameter.
- PQCodebook: Structure to load and hold the learned PQ centroids (e.g., std::vector<std::vector<std::vector<float>>> for m codebooks, each with k* centroids of dimension d/m). Needs methods for ADC.

- **On-Disk Structure (Conceptual Layout):**

- DiskNode: This structure needs careful design for serialization and disk layout. It should contain:

- uint32_t node_id;
- uint32_t num_neighbors;
- float vector[DIM]; // Full-precision vector
- uint32_t neighbor_ids; // Neighbor IDs
- uint8_t compressed_neighbors; // Compressed neighbor vectors (assuming m bytes per code)
- char padding; // Optional padding for alignment

- Consider using fixed-size blocks (e.g., 4096 bytes) for simplified I/O management.25 Data should be tightly packed. Serialization/deserialization routines are needed to convert between the in-memory representation (if different) and the packed disk format. Pay attention to endianness if the index file might be used across different architectures.

- **Graph Representation:** The Vamana graph is implicitly represented by the neighbor_ids stored within each DiskNode on disk. LM-DiskANN typically avoids maintaining a full adjacency list in memory to conserve RAM.

### **5.2. Index Building Algorithm Walkthrough (Conceptual C++)**

Building the index involves Vamana graph construction adapted for disk, incorporating PQ encoding.

1. **Initialization:**

- Open the index file for writing.
- Train PQ codebooks: Select a representative sample of the dataset, run k-means for each subvector space, and store the resulting PQCodebook (can be done offline or as a first step).
- Select initial entry point(s) (e.g., compute the dataset medoid). Write the DiskNode for the entry point(s) to the file.

1. **Incremental Build (Point-by-Point):** As described in the LM-DiskANN paper 6, points can be added incrementally. For each new point p with vector vec_p:

- **Find Neighbors:** Perform a Vamana-style Greedy Search starting from the entry point(s). This involves:

- Reading DiskNode blocks from the index file.
- Using the compressed_neighbors within each loaded block and the PQCodebook to estimate distances for routing.
- Maintaining the candidate queue L and visited set V.

- **Prune Neighbors:** Apply Robust Pruning (RobustPrune logic) to the visited set V from the search to select the best R neighbors for p.
- **Write New Node:**

- Create the DiskNode for p.
- Store p's ID, its full vector vec_p.
- Store the IDs of the selected R neighbors.
- For each neighbor n, retrieve its full vector (if not already available from the search), PQ-encode it using encode_pq(vec_n, pq_codebook), and store the compressed code in p's DiskNode.
- Serialize and write p's DiskNode to the index file (appending or writing to a specific offset).

- **Update Neighbors (Reciprocal Edges):** This is the most I/O-intensive part. For each of the R neighbors n selected for p:

- Read the DiskNode for neighbor n from disk.
- Add p's ID to n's neighbor_ids.
- PQ-encode vec_p and add the compressed code to n's compressed_neighbors.
- Increment n's num_neighbors.
- If n's num_neighbors now exceeds R, apply Robust Pruning to n's neighbor list (including p). This might involve reading vectors for n's other neighbors if not cached.
- Serialize and write the updated DiskNode for n back to its original location in the index file.

1. **Concurrency Control:** If multiple threads are building the index or if updates occur concurrently with searches, robust locking mechanisms are essential. This could involve file-level locks, block-level locks (e.g., mutexes associated with node ID ranges or specific file offsets), or finer-grained locking on in-memory structures managing updates.17

### **5.3. Search Algorithm Walkthrough (Conceptual C++)**

The search algorithm implements the beam search described earlier.

C++

// Simplified Conceptual Search Function
std::vector<uint32_t> search_lm_diskann(
  const QueryVector& query,
  uint32_t start_node_id,
  int k, // Number of neighbors to return
  int L_search, // Search list size
  int W_beam, // Beam width
  const PQCodebook& pq_codebook,
  IndexFile& index_file, // Handles disk I/O
  NodeCache& cache // Optional cache
) {
  CandidateQueue candidates; // Priority queue (min-heap by distance)
  VisitedSet visited;

  // Initialize with start node
  float start_dist = /* compute exact distance to start node (needs read) */;
  DiskNode start_node = index_file.read_node(start_node_id); // Initial read
  cache.put(start_node_id, start_node); // Cache it
  // Estimate distance using PQ for consistency in the queue? Or use exact?
  // Let's assume PQ distance for queue management.
  float estimated_start_dist = compute_pq_distance(query.vector,
                         encode_pq(start_node.vector, pq_codebook), // Need encoding step here
                         pq_codebook);
  candidates.push({estimated_start_dist, start_node_id});

  std::vector<CandidateNode> final_candidates; // Store candidates for re-ranking

  while (candidates.has_unvisited(visited)) {
    // 1. Select top W unvisited candidates
    std::vector<uint32_t> beam = candidates.pop_top_W_unvisited(W_beam, visited);

​    if (beam.empty()) break; // Termination condition

​    // Add to visited set
​    visited.insert(beam.begin(), beam.end());

​    // Store candidates for potential re-ranking later
​    // (Need distance associated with these beam nodes)

​    // 3. Fetch nodes and explore neighbors
​    std::vector<uint32_t> neighbor_ids_to_add;
​    std::vector<float> neighbor_dists_to_add;

​    for (uint32_t node_id : beam) {
​      DiskNode current_node;
​      if (!cache.get(node_id, current_node)) {
​        current_node = index_file.read_node(node_id);
​        cache.put(node_id, current_node);
​      }

​      // Add node itself to final candidates if needed for re-ranking
​      // (Requires storing its distance when popped from queue)

​      // 4. Explore neighbors
​      for (uint32_t i = 0; i < current_node.num_neighbors; ++i) {
​        uint32_t neighbor_id = current_node.neighbor_ids[i];
​        if (!visited.contains(neighbor_id)) {
​          // Calculate estimated distance using PQ codes stored in current_node
​          const uint8_t* compressed_neighbor = current_node.compressed_neighbors[i];
​          float estimated_dist = compute_pq_distance(query.vector, compressed_neighbor, pq_codebook);

​          // Check if candidate should be added (based on distance comparison with farthest in queue)
​          if (candidates.size() < L_search |
| estimated_dist < candidates.top().distance) {
​             neighbor_ids_to_add.push_back(neighbor_id);
​             neighbor_dists_to_add.push_back(estimated_dist);
​          }
​        }
​      }
​    }
​    // Add collected neighbors to the queue, maintaining size L_search
​    candidates.push_batch(neighbor_ids_to_add, neighbor_dists_to_add);
​    candidates.prune(L_search);

​    // Check termination condition (e.g., beam nodes are farther than farthest in queue)
  }

  // 5. Re-ranking (Optional but recommended)
  // Get top-k candidates from the search (need to track nodes explored)
  // Fetch their full vectors (from cache or disk)
  // Compute exact distances
  // Sort and return top-k IDs

  // Placeholder: return candidates from queue for simplicity
  return candidates.get_top_k_ids(k);
}



### **5.4. Product Quantization Integration**

- **Training:** Requires implementing or linking k-means. Faiss provides robust k-means implementations. The training function takes a matrix of training vectors (e.g., N_train x d), splits them into m subvectors, runs k-means m times (once per subspace) to find k* centroids for each, and stores these centroids in the PQCodebook structure.
- **Encoding:** encode_pq(vector, codebook) function: Splits the input vector into m subvectors. For each subvector, finds the nearest centroid in the corresponding codebook subspace (using L2 distance) and records the centroid's index (0 to k∗−1). Returns the array of m indices (e.g., std::vector<uint8_t>).
- **Distance Function (ADC):** compute_pq_distance(query_vector, compressed_code, codebook):

1. Precomputation (can be done once per query): Split the query_vector into m subvectors. For each subvector j, compute and store the L22 distances between the query subvector and all k∗ centroids in the j-th codebook subspace. This results in an m×k∗ distance table.
2. Lookup and Summation: For the given compressed_code (which contains m centroid indices i1,i2,…,im), look up the precomputed distances in the table: $distance^2 \approx \sum_{j=1}^{m} \text{distance_table}[j][i_j]$.
3. Return distance2 (or just use squared distances if only relative order matters). Faiss offers highly optimized implementations of these PQ operations, potentially using SIMD instructions.44 Leveraging Faiss for PQ components could save significant implementation and optimization effort.

### **5.5. Managing Disk I/O Efficiently**

This is critical for LM-DiskANN performance.

- **Buffering:** Use standard library buffered streams (std::fstream) or implement custom buffering on top of lower-level file operations (fopen/fread/fwrite or OS-specific APIs) to aggregate small reads/writes into larger, more efficient disk operations.
- **Alignment:** Ensure the DiskNode structure size and file offsets used for reading/writing are aligned to the underlying filesystem block size or SSD page size (often 4KB or a multiple). This can avoid read-modify-write cycles and improve performance. Use alignas or manual padding in C++ structs, and aligned memory allocation (aligned_alloc or platform equivalents) for buffers. Memory-mapped files inherently work with page granularity.38
- **Asynchronous I/O:** To overlap disk latency with computation (e.g., distance calculations, queue management), use asynchronous I/O mechanisms.

- **Linux:** io_uring is the modern, high-performance interface. POSIX AIO (aio_read, aio_write) is an older alternative.
- **Windows:** Input/Output Completion Ports (IOCP) provide a scalable asynchronous model.
- **Cross-Platform (via libraries):** Libraries like Boost.Asio abstract some platform differences.
- **Thread Pool:** A simpler approach is to use a thread pool. Worker threads perform blocking I/O operations, freeing the main search thread(s) to continue processing other candidates or queries.

### **5.6. Relevant Libraries and Tools**

- **Linear Algebra:** For exact distance calculations (re-ranking) or potentially k-means implementation:

- **BLAS/LAPACK:** OpenBLAS, Intel MKL, Apple Accelerate. Provide optimized matrix/vector operations.44

- **ANN/PQ Libraries:**

- **Faiss:** Provides highly optimized C++ implementations of PQ (training, encoding, ADC), k-means, various ANN indices (including HNSW, IVF), and distance functions. Can be used as a component library.32

- **Threading:**

- std::thread, std::mutex, std::condition_variable: Standard C++ concurrency primitives.
- **OpenMP:** Directive-based parallelism, good for loop parallelization.
- **Intel TBB (Threading Building Blocks):** High-level library for task-based parallelism.

- **Memory Allocators:**

- **tcmalloc (Google):** Often provides better performance for multi-threaded applications with many small allocations.38
- **jemalloc (Facebook):** Another high-performance allocator focused on reducing fragmentation and scaling concurrency.

- **Build System:**

- **CMake:** Cross-platform build system generator, widely used in C++ projects.38

## **6. Case Study: Analyzing vectordiskann.c (libsql Implementation)**

While direct access to the C source code was unavailable 47, documentation and blog posts from Turso provide significant details about their libsql implementation of vector search, which they explicitly state uses LM-DiskANN.6

### **6.1. Confirmation of LM-DiskANN Approach**

Multiple sources from Turso confirm their implementation is based on the LM-DiskANN variant.6 The stated goal is low memory consumption suitable for SQLite's typical use cases (embedded, multi-tenant, mobile).25 They highlight the key difference from standard DiskANN: storing compressed neighbor vectors on disk within each node's block to avoid the large in-memory cache.25 This directly aligns with the LM-DiskANN paper.6

### **6.2. Analysis of Data Structures (Based on Descriptions)**

- **Node Representation:** The implementation uses fixed-size binary blocks stored within the SQLite database file itself.25 Each block contains the node's full vector, its neighbor IDs, and compressed representations of those neighbors.25 This structure enables the "local decisions" characteristic of LM-DiskANN.25
- **Configurable Compression:** libsql allows configuring the compression type used for storing neighbor vectors via index creation parameters.11 Options include no compression (neighbors stored as full vectors, likely FLOAT32), custom byte-based quantization (FLOAT8), and even 1-bit quantization (FLOAT1BIT), in addition to standard types like FLOAT16/BFLOAT16.11 This flexibility allows trading disk space and accuracy. PQ is likely achievable through these mechanisms or future extensions, although simpler quantization might be the default.
- **SQLite Integration:** The vector index is tightly integrated with SQLite. It's created using CREATE INDEX... USING libsql_vector_idx(column, 'option=value',...).11 Special SQLite bytecode instructions (OP_OpenVectorIdx, OP_IdxInsert) are generated to handle index opening and updates during standard SQL operations like INSERT.25 Search is performed via a virtual table function vector_top_k('index_name', query_vector, k).25

### **6.3. Dissection of the Search Algorithm Implementation**

The search algorithm implemented is described as a Best First Search (equivalent to beam search with W=1, or configurable via parameters not fully detailed but implied) on the Vamana graph.25

- **Initialization:** Starts from a random entry point node in the graph.25
- **Iteration:** Maintains a candidate set (priority queue) of bounded size L (configurable index parameter).25 In each step:

1. Selects the nearest candidate node from the set that hasn't been visited.
2. Marks the node as visited.
3. Reads the node's block from the SQLite file (disk).
4. Calculates distances to its neighbors using the compressed neighbor vectors stored *within that block*.
5. Adds unvisited neighbors to the candidate set, maintaining the size limit L by pruning the farthest nodes.25

- **Termination:** Continues until the candidate set is exhausted or another condition is met. The vector_top_k function returns the row IDs of the found neighbors.25

### **6.4. Examination of Disk I/O Handling**

Disk I/O is managed implicitly through SQLite's architecture. The LM-DiskANN node blocks are stored within SQLite's B-tree pages. When a node block needs to be read or written, libsql relies on SQLite's pager component to handle the interaction with the operating system for reading/writing database file pages from/to disk and managing the page cache.25 The fixed-size block approach likely simplifies mapping nodes to SQLite pages.25

### **6.5. Lessons Learned and Adaptability to C++**

The libsql implementation serves as a valuable case study:

- **Viability:** It demonstrates that the LM-DiskANN architecture is practical and can be implemented effectively within a production database system.
- **Configurability:** The inclusion of parameters for neighbor compression type and candidate list size (L) highlights the importance of tunability for different use cases.11
- **Integration:** It shows how such an index can be integrated into a larger data management framework, hooking into existing mechanisms for updates and queries.
- **Adaptation to C++:** The core logic – the self-contained node structure enabling local routing decisions using on-disk compressed neighbors, and the best-first/beam search traversal – is directly transferable to a standalone C++ implementation. Key differences in a C++ version would be the need for explicit file management, I/O optimization (buffering, async I/O, alignment), caching strategies, and thread safety management, aspects largely handled by the SQLite framework in the libsql case. The described libsql data structures and search flow provide a solid conceptual blueprint.

## **7. Conclusion and Implementation Recommendations**

### **7.3. Final Recommendations**

For a successful LM-DiskANN implementation:

1. **Start Simple, Iterate:** Begin by implementing the core search functionality for a static, pre-built index. Focus on correctly implementing the on-disk node structure, the beam search using local compressed neighbor information, and the PQ distance calculations. Tackle dynamic updates later if required.
2. **Prioritize and Profile I/O:** Disk performance is paramount.9 Design the disk layout carefully (consider alignment 25), implement efficient buffering, and profile I/O patterns early and often. Investigate asynchronous I/O to overlap computation and disk access.9
3. **Systematic Tuning:** Performance is highly sensitive to hyperparameters (R,L,α,W, PQ params).25 Establish a robust evaluation framework using representative query datasets and ground truth. Tune parameters methodically to navigate the latency-recall trade-off and meet performance goals.
4. **Invest in PQ:** Ensure the Product Quantization component is robust. Use a sufficiently large and representative dataset for training the codebooks.35 Experiment with PQ parameters (m, code_size) to find the best balance between compression ratio (affecting node size and disk usage) and accuracy (affecting recall).35 Consider leveraging Faiss for its optimized PQ implementation.
5. **Modular Design:** Structure the C++ code logically into distinct components (e.g., Vamana graph logic, PQ handling, disk I/O layer, caching, search algorithm). This facilitates unit testing, debugging, and future extensions or optimizations.
6. **Reference Implementation Logic:** Use the described mechanics of Turso's libsql implementation (vectordiskann.c) 25 as a conceptual guide for the LM-DiskANN node structure, the reliance on local compressed neighbors for routing, and the best-first search flow. Adapt this logic to the specific requirements of a standalone C++ implementation, paying close attention to explicit I/O management, caching, and concurrency control.

By carefully considering these architectural nuances, trade-offs, and implementation details, it is possible to build a robust and efficient LM-DiskANN system capable of delivering high-performance approximate nearest neighbor search on massive datasets with remarkably low memory requirements.

#### **Works cited**

1. Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters - Harsha Simhadri, accessed April 26, 2025, https://harsha-simhadri.org/pubs/Filtered-DiskANN23.pdf
2. LM-DiskANN: Low Memory Footprint in Disk-Native Dynamic Graph-Based ANN Indexing - University of Nebraska–Lincoln, accessed April 26, 2025, [https://cse.unl.edu/~yu/homepage/publications/paper/2023.LM-DiskANN-Low%20Memory%20Footprint%20in%20Disk-Native%20Dynamic%20Graph-Based%20ANN%20Indexing.pdf](https://cse.unl.edu/~yu/homepage/publications/paper/2023.LM-DiskANN-Low Memory Footprint in Disk-Native Dynamic Graph-Based ANN Indexing.pdf)
3. FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search - arXiv, accessed April 26, 2025, https://arxiv.org/pdf/2105.09613
4. DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node, accessed April 26, 2025, https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node
5. A Topology-Aware Localized Update Strategy for Graph-Based ANN Index - arXiv, accessed April 26, 2025, https://arxiv.org/html/2503.00402beta2