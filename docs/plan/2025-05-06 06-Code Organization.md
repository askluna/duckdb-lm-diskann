## DiskANN C++ Code Organization and structure

This document presents a formalized C++ architectural specification for the DiskANN DuckDB extension. The proposed structure is centered upon an `Orchestrator` class, situated within a dedicated `diskann::core` namespace, which is tasked with the management of core indexing logic. This design paradigm is intended to promote modularity, enhance testability, and ensure maintainability. The present revision incorporates considerations pertaining to nomenclature, directory organization, and offers detailed recommendations for class design, the application of functional programming paradigms, and adherence to C++20 best practices.

### 1. Fundamental Design Principle: The `diskann::core::Orchestrator`

The `Orchestrator` class constitutes the central architectural element. The DuckDB-specific `DiskannIndex` class, which inherits from DuckDB's `Index` base class, will assume the following responsibilities:

- Implementation of the DuckDB `Index` interface, encompassing methods such as `Bind`, `InitializeScan`, `Scan`, `Append`, `Delete`, and `Verify`.
- Interaction with DuckDB's systemic components, including the catalog, storage manager, buffer manager, and transaction manager.
- Translation of DuckDB data structures and operational requests into a format comprehensible to the `diskann::core::Orchestrator`.
- Management of the lifecycle pertaining to the `Orchestrator` instance associated with a given index.

The `diskann::core::Orchestrator` is designed to encapsulate the algorithmic logic of the DiskANN implementation:

- **State Management:** It will possess ownership and control over the state of the DiskANN graph, including elements such as entry point(s), graph metadata, and configuration parameters. This remit extends to managing the file system paths for on-disk structures.
- **Core Operations:** It will implement high-level operations, including index construction, search execution, new vector insertion, and the handling of deletions and updates. The latter may involve sophisticated mechanisms, such as those delineated in the "Shadow Implementation.md" document.
- **Coordination:** It will delegate specific tasks to specialized components residing within the `diskann::core` namespace, such as `GraphManager`, `StorageManager`, and `Searcher`.
- **Testability:** The `Orchestrator` is engineered for isolated testing, independent of the DuckDB environment. Its dependencies, for instance `IStorageManager`, will be injectable, thereby facilitating the use of mock objects and enabling focused unit and integration testing of the core Approximate Nearest Neighbor (ANN) algorithm.

### 2. Proposed Component Architecture and Functional Responsibilities

#### Namespace Allocation:

- `diskann::duckdb`: Designated for components that directly interface with the DuckDB system.
- `diskann::core`: Encompasses the core DiskANN algorithmic logic, maintaining independence from DuckDB.
- `diskann::common`: Contains shared data structures and utility functions utilized across the extension.

#### Component Delineation:

- **`diskann/duckdb/`**

  - **`DiskannIndex` (.hpp/.cpp):**
    - Serves as the primary interface to the DuckDB system.
    - Manages DuckDB-specific tasks and data type conversions (e.g., transformation of DuckDB vectors into internal representations).
    - Instantiates and invokes the `diskann::core::Orchestrator`, injecting its dependencies.
    - Oversees DuckDB-specific index metadata and storage considerations (e.g., interaction with DuckDB's `BlockManager` or custom storage mechanisms via the `StorageManager`).
  - **`DiskannScanState` (.hpp/.cpp):**
    - Manages state variables during a scan operation within the DuckDB context (e.g., the current query vector, iterators for result sets).
  - **`DiskannBindData`, `DiskannCreateIndexInfo` (.hpp/.cpp):**
    - Custom data structures tailored for DuckDB's bind, create index, and other operational phases. These structures will encapsulate parsed options and information specific to a DiskANN index.
  - **`diskann_extension.cpp`:** (Typically located at the `src/` level or a designated extension loading point)
    - Handles DuckDB extension registration, definition of the index type, and associated functions or pragmas.

- **`diskann/core/`**

  - **`Orchestrator` (.hpp/.cpp):** (Class)
    - Functions as the central coordinator for all core DiskANN operations.
    - Manages the overall index state, including in-memory metadata and references to on-disk structures, mediated by the `StorageManager`.
    - Orchestrates build, search, insert, delete, update, load, and save operations.
    - Interacts with `IStorageManager`, `IGraphManager`, `ISearcher`, and `IndexConfig` (all typically received as dependencies).
    - Represents a critical class for the implementation of logic described in "Shadow Implementation.md," such as the coordination of merge operations from a shadow store to the main graph.
  - **`IndexConfig` (.hpp/.cpp):** (Struct or Class)
    - Stores all DiskANN operational parameters (e.g., R, L_build, L_search, alpha, dimensionality, distance metric type, quantization settings, paths to index files, on-disk format version).
    - Manages the loading and saving of configuration data from and to persistent storage (via `StorageManager` or directly).
    - It is anticipated that this will be a struct for straightforward data aggregation, potentially evolving into a class if validation or complex default logic becomes necessary.
  - **`Distance` (.hpp/.cpp):** (Namespace containing functions, or a templated class/strategy pattern)
    - Provides a suite of distance functions (e.g., L2 squared, Cosine similarity).
    - Implementation may take the form of a collection of templated free functions or employ a strategy pattern utilizing an `IDistanceMetric` interface with concrete implementations. Templated free functions are often favored for performance due to the potential for inlining.
    - Illustrative example: `namespace diskann::core::distance { template<typename T> T l2_squared(const T* vec1, const T* vec2, uint32_t dim); }`
  - **`IStorageManager` / `StorageManager` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`IStorageManager`):** Defines the contractual obligations for all disk input/output operations.
    - **Class (`StorageManager`):** Provides the concrete implementation of the `IStorageManager` interface.
    - **Responsibilities:** Abstracts the processes of reading and writing graph nodes, vectors, index metadata, and configuration data. Manages file handles, memory mapping (if utilized), and block allocation strategies. This includes managing different storage areas (e.g., main graph file, shadow/delta stores).
    - This component is of critical importance for the shadow implementation, as it will handle read/write operations for both the primary graph file (`graph.lmd`) and any delta or shadow storage mechanisms (e.g., `__lmd_blocks`).
    - Exemplar methods: `Initialize(base_path, config)`, `LoadMetadata() -> IndexConfig`, `SaveMetadata(const IndexConfig& config)`, `ReadNode(common::node_id_t node_id, StorageTier tier = StorageTier::MAIN) -> std::optional<common::Node>`, `WriteNode(common::node_id_t node_id, const common::Node&, StorageTier tier = StorageTier::SHADOW)`, `AllocateNodeBlock(StorageTier tier = StorageTier::MAIN) -> common::node_id_t`, `ReadVector(common::node_id_t node_id, StorageTier tier = StorageTier::MAIN) -> std::optional<std::vector<DataType>>`, `CommitShadowWrites()`, `MergeShadowToMain()`. (The `StorageTier` enum and specific methods related to shadow management are illustrative).
  - **`IGraphManager` / `GraphManager` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`IGraphManager`):** Defines the contractual obligations for managing the logical structure of the Vamana graph.
    - **Class (`GraphManager`):** Provides the concrete implementation of the `IGraphManager` interface.
    - **Responsibilities:** Manages adding nodes, robust pruning procedures, neighbor discovery, connectivity maintenance, and the enforcement of graph invariants (e.g., maximum degree).
    - Utilizes `IStorageManager` for the persistence of modifications and the retrieval of node data.
    - Logic previously associated with `NodeAccessors` would be integrated within this manager or implemented as helper classes utilized by it.
    - The role of a `NodeManager` (if distinct) in managing the node lifecycle beyond raw storage is appropriately situated here.
  - **`ISearcher` / `Searcher` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`ISearcher`):** Defines the contractual obligations for performing ANN search operations.
    - **Class (`Searcher`):** Provides the concrete implementation of the `ISearcher` interface.
    - **Responsibilities:** Implements the ANN search algorithms (e.g., greedy search, beam search).
    - Utilizes `IGraphManager` for graph traversal and `IStorageManager` for fetching node data and vectors (potentially from different storage tiers if shadow reads are supported during search).
    - Manages the search context, including the query vector, visited lists, candidate heaps or pools, and current search parameters such as `L_search`.
  - **`IQuantizer` / `Quantizer` (.hpp/.cpp):** (Interface and Class, if applicable)
    - **Interface (`IQuantizer`):** Defines the contract for vector quantization and dequantization operations.
    - **Class (`Quantizer`):** Provides a concrete implementation (e.g., for ternary quantization).
    - **Responsibilities:** Handles the conversion of vectors to and from a compressed representation.

- **`diskann/common/`**

  - **`types.hpp`:**

    - `node_id_t`: A typedef for node identifiers (e.g., `uint32_t` or `uint64_t`).

    - `StorageTier`: Enum to specify storage locations (e.g., `MAIN`, `SHADOW`), relevant for `IStorageManager` and shadow implementation.

    - `Node`: A struct representing a graph node. It contains its vector (or a reference/pointer if stored separately by `StorageManager`), a list of neighbor `node_id_t`s, and potentially other metadata.

      ```
      // Example Node struct
      struct Node {
          node_id_t id;
          std::vector<float> vector_data; // Or std::span<float> if providing a view
          std::vector<node_id_t> neighbors;
          // Potentially other metadata, such as level for multi-layered graph architectures
      };
      ```

    - `Candidate`: A struct utilized during search operations, typically holding a `node_id_t` and its calculated distance to the query vector. This is frequently employed within a priority queue structure.

    - `GraphNodeView`: A non-owning struct or class that provides a view into a node's data (e.g., utilizing `std::span` for vector and neighbor collections), designed to obviate unnecessary data copying.

    - `BuildProgress`: A struct for reporting progress during the index construction phase (e.g., number of nodes processed, current operational phase).

    - Custom enumerations for distance metrics, status codes, etc.

  - **`utils.hpp` / `utils.cpp`:**

    - General utility functions (e.g., logging helpers, timer utilities, file system operations not encompassed by `StorageManager`).
    - Potentially, simple serialization/deserialization helpers for POD structs if not part of `StorageManager`.
    - Mathematical helper functions not appropriately categorized under `Distance`.

  - **`constants.hpp`:**

    - Global constants pertinent to the DiskANN implementation (e.g., default configuration values if not solely in `IndexConfig`, magic numbers for file formats).

#### Distinction between Classes and Functional Approaches (Free Functions) & C++20 Considerations:

- **Classes:** Employed for components that encapsulate substantial state and behavior (e.g., `Orchestrator`, `StorageManager`, `GraphManager`, `Searcher`). The methods of these classes operate upon this encapsulated state. Interfaces (e.g., `ISomethingManager`) are of paramount importance for these components to enable polymorphism and facilitate mocking for testing purposes.
- **Structs:** Primarily utilized for Plain Old Data (POD) to aggregate related data fields (e.g., `Node`, `Candidate`, `IndexConfig` if its nature is simple, `BuildProgress`). Such structures typically exhibit minimal or no behavior beyond constructors and destructors.
- **Namespaces with Free Functions:**
  - Appropriate for stateless utility functions or operations that do not naturally align with class-based encapsulation (e.g., `diskann::core::distance::l2_squared(...)`, `diskann::common::utils::log_message(...)`).
  - Related free functions should be grouped within namespaces to prevent pollution of the global namespace and to enhance organizational clarity.
- **C++20 for Enhanced Testability and Modern Software Practices:**
  - **`std::span`:** Recommended for extensive use to provide non-owning views of contiguous data sequences (e.g., vectors, neighbor lists, buffers managed by `StorageManager`). This practice improves memory safety and efficiency by obviating unnecessary data copies and by clearly demarcating ownership.
  - **Concepts:** May be utilized to constrain template parameters for interfaces or utility functions, thereby improving compile-time verification and the clarity of error messages (e.g., `template<typename T> requires std::is_floating_point_v<T> ...`).
  - **Ranges (`std::ranges`):** Offer a means to simplify algorithms that operate on sequences (e.g., iterating and transforming neighbor lists).
  - **`const` correctness and `noexcept` specifiers:** Should be applied rigorously to enhance code clarity, enable compiler optimizations, and aid in reasoning about program behavior.
  - **`std::optional`:** Suitable for functions that may not invariably return a value (e.g., `ReadNode` if a specified node ID does not exist and this condition is not considered an exceptional error).
  - **`std::expected` (C++23, or a library equivalent for C++20):** Merits consideration for functions that can return either a value or an error, offering a more explicit error handling mechanism than exceptions for common, anticipated failure modes.
  - **Smart Pointers (`std::unique_ptr`, `std::shared_ptr`):** Essential for managing dynamically allocated memory, ensuring Resource Acquisition Is Initialization (RAII) principles are upheld and memory leaks are prevented. `std::unique_ptr` should be the default choice for exclusive ownership when injecting dependencies.
  - **Modules:** While representing a promising future direction for C++ development (offering improved encapsulation and potentially faster compilation times), widespread adoption and comprehensive tooling support may still be in a developmental phase. Consequently, traditional header/source file organization with explicit include guards remains the standard practice for the present.

### 3. Suggested Directory Structure

```
src/
├── diskann/
│   ├── core/                   // Core DiskANN logic, independent of DuckDB
│   │   ├── orchestrator.hpp
│   │   ├── orchestrator.cpp
│   │   │
│   │   ├── index_config.hpp
│   │   ├── index_config.cpp
│   │   │
│   │   ├── distance.hpp        // Likely header-only templates or inline functions
│   │   │
│   │   ├── istorage_manager.hpp // Interface
│   │   ├── storage_manager.hpp
│   │   ├── storage_manager.cpp
│   │   │
│   │   ├── igraph_manager.hpp  // Interface
│   │   ├── graph_manager.hpp
│   │   ├── graph_manager.cpp
│   │   │
│   │   ├── isearcher.hpp       // Interface
│   │   ├── searcher.hpp
│   │   ├── searcher.cpp
│   │   │
│   │   ├── iquantizer.hpp      // Interface (Optional)
│   │   ├── quantizer.hpp       // (Optional)
│   │   ├── quantizer.cpp       // (Optional)
│   │
│   ├── duckdb/                 // Components interacting directly with DuckDB
│   │   ├── diskann_index.hpp
│   │   ├── diskann_index.cpp
│   │   ├── diskann_scan_state.hpp
│   │   ├── diskann_bind_data.hpp // And other DuckDB-specific structs
│   │   └── ...
│   │
│   └── common/                 // Shared utilities and data types for the extension
│       ├── types.hpp           // e.g., Node struct, Candidate struct, node_id_t, StorageTier
│       ├── utils.hpp
│       ├── utils.cpp
│       └── constants.hpp
│
├── diskann_extension.cpp       // Main extension loading file (registers the index)
└── CMakeLists.txt              // Main CMake file

// CMakeLists.txt within src/diskann/core/ could define a static library (e.g., diskann_core),
// which is subsequently linked by the main DuckDB extension library.
// A similar approach can be adopted for src/diskann/common/.
```

### 4. Advantages of the Proposed Architecture

- **Enhanced Testability:**
  - The `diskann::core` module, particularly the `Orchestrator` and its constituent components (accessed via interfaces such as `IStorageManager`), can be subjected to unit testing with mocked dependencies, thereby obviating the need for a running DuckDB instance.
- **Clear Separation of Concerns (SoC):**
  - `diskann::duckdb::DiskannIndex` is responsible for DuckDB integration.
  - `diskann::core::Orchestrator` manages the "business logic" of the ANN index.
  - Specialized classes (`StorageManager`, `GraphManager`, `Searcher`) address their specific domains, adhering to the Single Responsibility Principle.
- **Improved Maintainability & Readability:**
  - Smaller, focused classes and functions, organized within well-defined namespaces, are more readily understood, debugged, and modified.
  - Modifications to DuckDB's internal APIs are more likely to be confined to the `diskann::duckdb` layer.
  - Alterations to the core DiskANN algorithm are isolated within `diskann::core`.
- **Modularity:**
  - The `diskann::core` library possesses the theoretical potential for reuse or adaptation in other contexts, although this is a secondary consideration for an extension tightly coupled with DuckDB.
- **Scalability of Development Efforts:**
  - Multiple developers can concurrently work on distinct components (`core`, `duckdb`, `common`) with clearly demarcated boundaries.

### 5. Salient Considerations and Recommended Best Practices

- **Interfaces (Abstract Base Classes):**

  - Define abstract interfaces (e.g., `IStorageManager`, `IGraphManager`, `ISearcher`) for major components within `diskann::core`. The `Orchestrator` will depend on these interfaces.

  - Concrete implementations are subsequently provided. This is of critical importance for facilitating dependency injection and mocking in testing environments.

  - Illustrative Example:

    ```
    // diskann/core/istorage_manager.hpp
    namespace diskann::core {
    class IStorageManager {
    public:
        virtual ~IStorageManager() = default;
        virtual bool Initialize(const std::string& base_path, const IndexConfig& config) = 0;
        virtual std::optional<common::Node> ReadNode(common::node_id_t node_id, common::StorageTier tier = common::StorageTier::MAIN) = 0;
        virtual bool WriteNode(common::node_id_t node_id, const common::Node& node_data, common::StorageTier tier = common::StorageTier::SHADOW) = 0;
        // ... other pure virtual functions including those for shadow management
    };
    } // namespace diskann::core
    ```

- **Dependency Injection:**

  - The `diskann::duckdb::DiskannIndex` is responsible for the creation of the `diskann::core::Orchestrator` and its concrete dependencies.
  - The `Orchestrator` receives its dependencies (e.g., `std::unique_ptr<IStorageManager>`, `std::unique_ptr<IGraphManager>`) via its constructor. Constructor injection is generally preferred due to its clarity and facilitation of testability, as mock objects can be readily injected. This is achieved in C++ by defining abstract base classes (interfaces) with pure virtual functions. The dependent class (e.g., `Orchestrator`) takes pointers or references (typically smart pointers like `std::unique_ptr`) to these interface types in its constructor. The calling code (e.g., within `DiskannIndex`) then provides concrete implementations that fulfill these interfaces. This decouples components and allows for easy substitution of mock implementations during testing without needing specialized DI framework libraries.

- **Explicit Ownership and State Management:**

  - The `Orchestrator` maintains ownership of high-level index state (such as the `IndexConfig` instance if not passed as a const reference) and orchestrates operations. It takes ownership of its injected dependencies (e.g., via `std::unique_ptr`).
  - `StorageManager` assumes ownership of file handles and manages disk I/O buffers.
  - `GraphManager` owns the in-memory representation of graph metadata (e.g., entry point, current size) and the rules governing graph modification.
  - Utilize smart pointers (`std::unique_ptr` for exclusive ownership, `std::shared_ptr` for shared ownership where demonstrably necessary) to manage the lifetime of dynamically allocated objects.

- **Error Handling Strategies:**

  - Employ exceptions for genuinely exceptional situations that preclude the continuation of an operation (e.g., critical I/O failures, corrupted index files). Define custom exception types derived from `std::exception` to provide enhanced contextual information.
  - For recoverable errors or conditions where a value may not be present, preference should be given to `std::optional` (for indicating presence or absence) or `std::expected` (C++23, or a library equivalent for C++20, for representing a value-or-error state). This approach renders error paths more explicit.

- **Concurrency Management:**

  - The `Orchestrator` serves as a natural locus for managing high-level concurrency control mechanisms (e.g., read-write locks to protect the index during concurrent modifications and searches), should such control not be entirely managed by DuckDB's transactional framework or the specifics of the shadow file mechanism.
  - Interfaces and implementations of `StorageManager` and `GraphManager` must be designed with consideration for thread safety if they are intended for concurrent access. The thread-safety guarantees of each class and method should be explicitly documented.

- **Lifecycle Management:**

  - `DiskannIndex` is responsible for managing the lifecycle of the `Orchestrator`:
    - **Creation:** Instantiated when `CREATE INDEX` is executed or when an existing index is loaded. The `Orchestrator` is initialized with the index path, configuration, and its injected dependencies.
    - **Operations:** Calls made by DuckDB on `DiskannIndex` are translated into corresponding calls on the `Orchestrator`.
    - **Destruction:** Occurs when the index is dropped or during DuckDB shutdown. The `Orchestrator` (and its owned dependencies via smart pointers) ensures resources are released, and data is flushed via `IStorageManager`.

### 6. Further Organizational and Planning Considerations

To augment the planning and design process, the following aspects warrant detailed consideration:

1. **Comprehensive Testing Strategy:**
   - **Unit Tests:** Each class within `diskann::core` (e.g., `StorageManager`, `GraphManager`, `Searcher`, `Orchestrator` itself) should have thorough unit tests. Dependencies will be mocked using the defined interfaces (`IStorageManager`, etc.). Focus on testing individual methods, edge cases, and invariants. These tests should reside alongside the core code (e.g., in a `tests/` subdirectory within `diskann/core/`).
   - **Integration Tests (Core):** Test interactions between core components (e.g., `Orchestrator` correctly uses `GraphManager` and `StorageManager` for an insert operation). These can still use mocked storage at the lowest level if full disk I/O is too slow or complex for this stage.
   - **Integration Tests (DuckDB Extension):** These tests will involve actual DuckDB queries (`CREATE INDEX`, `SELECT ... ORDER BY L2Distance(...) LIMIT K`). They will verify the end-to-end functionality of the `diskann::duckdb` layer and its interaction with the `diskann::core` components. DuckDB's testing framework should be leveraged.
   - **Performance Benchmarks:** Establish a suite of benchmarks for indexing time, search latency, recall, and throughput under various conditions (dataset size, dimensionality, batch inserts, concurrent queries).
   - **Stress Tests and Durability Tests:** Specifically for the shadow implementation and disk-based aspects, tests involving abrupt shutdowns, large volumes of concurrent writes/reads, and recovery scenarios are crucial.
2. **Configuration Management and Propagation:**
   - The `diskann::duckdb::DiskannIndex` will parse `CREATE INDEX ... WITH (...)` options. These options will be used to populate an `diskann::core::IndexConfig` object.
   - This `IndexConfig` object should be passed (e.g., by const reference or shared pointer) to the `Orchestrator` upon its initialization.
   - The `Orchestrator` will then pass relevant parts of the configuration (or the entire config object) to its dependent components (`StorageManager`, `GraphManager`, `Searcher`) during their construction or initialization. For example, `StorageManager` needs file paths and block sizes; `GraphManager` needs parameters like `R` (degree bound); `Searcher` needs `L_search`.
   - The `IndexConfig` should also include a field for the on-disk format version of the index files.
3. **Shadow Implementation – Key Interface Considerations:**
   - The `IStorageManager` interface will be central. It must abstract the existence of different storage tiers (e.g., main graph file, shadow/delta store). Methods like `ReadNode`, `WriteNode` might need an optional parameter indicating which tier to target or a strategy for resolving node locations (e.g., "read from shadow first, then main").
   - The `Orchestrator` will coordinate operations involving the shadow mechanism, such as:
     - Directing new writes/updates to the shadow store via `IStorageManager`.
     - Triggering merge operations from the shadow store to the main graph file (potentially managed by `IStorageManager` but initiated by `Orchestrator`).
     - Ensuring search operations (via `ISearcher`) correctly query across both the main graph and the shadow store, or a consistent snapshot, as per the shadow design's visibility rules.
   - The `IndexConfig` might need to store metadata related to the shadow store's state (e.g., current shadow file, high-water marks).
4. **Build System and Modularity (CMake):**
   - Confirm the plan to build `diskann::core` and `diskann::common` as static libraries.
   - The main `CMakeLists.txt` for the extension will link against these static libraries.
   - This structure facilitates separate compilation and testing of the core logic.
   - Ensure clear separation of include directories and dependencies between these modules.
5. **On-Disk Format Versioning and Migration:**
   - The `IndexConfig` (and persisted metadata on disk) must include a version number for the on-disk data structures (e.g., `graph.lmd` block layout, metadata file format).
   - The `StorageManager` (or `Orchestrator` during initialization) must check this version upon loading an existing index.
   - Develop a strategy for:
     - Supporting older versions (read-only if complex).
     - Providing a mechanism to upgrade an index to the latest format (either automatically on load or via a user-initiated command).
     - Clearly erroring out if an unsupported/corrupt version is detected.
   - This is critical for long-term maintenance and allowing users to upgrade the extension without necessarily rebuilding all their indexes.
6. **Logging, Metrics, and Observability:**
   - Define different log levels (DEBUG, INFO, WARNING, ERROR).
   - Key operations (index build, search, merge) should have clear start/end log messages and report progress.
   - Expose critical metrics (e.g., cache hit/miss rates in `StorageManager`, average number of distance computations per search, graph density, merge operation duration) that can be queried or logged for performance analysis and debugging. This could be via DuckDB's `PRAGMA` system or specific functions.

By proactively planning for these aspects, the development process will be more structured, potential issues can be identified earlier, and the resulting extension will be more robust, maintainable, and user-friendly.