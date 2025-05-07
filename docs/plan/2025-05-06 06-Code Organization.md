## Formalized DiskANN C++ Architectural Specification for DuckDB Extension Integration

This document outlines a C++ architectural specification for the DiskANN DuckDB extension. It centers on an `Orchestrator` class within a `diskann::core` namespace to manage core indexing logic, promoting modularity, testability, and maintainability. This specification details component responsibilities, namespace allocation, directory structure, and C++20 best practices.

### 1. Fundamental Design Principle: The `diskann::core::Orchestrator`

The `diskann::core::Orchestrator` class is the central architectural component. The DuckDB-specific `DiskannIndex` class (in `diskann::duckdb`) interfaces with DuckDB, translating requests for the `Orchestrator`.

The `diskann::core::Orchestrator` encapsulates DiskANN's algorithmic logic:

- **State Management:** Owns and manages the DiskANN graph state (entry points, metadata, configuration, on-disk paths).
- **Core Operations:** Implements index building, searching, insertions, deletions, and updates, including mechanisms for the shadow implementation.
- **Coordination:** Delegates tasks to specialized `diskann::core` components (`GraphManager`, `StorageManager`, `Searcher`).
- **Testability:** Designed for isolated testing via injectable dependencies (e.g., `IStorageManager`), facilitating unit and integration tests of the core ANN algorithm.

### 2. Proposed Component Architecture and Functional Responsibilities

#### Namespace Allocation:

- `diskann::duckdb`: Designated for components that directly interface with the DuckDB system.
- `diskann::core`: Encompasses the core DiskANN algorithmic logic, including shared data structures and utility functions.

#### Component Delineation:

- **`diskann/duckdb/`**
  - **`DiskannIndex` (.hpp/.cpp):**
    - Primary interface to DuckDB; handles DuckDB-specific tasks, data conversions.
    - Instantiates and invokes `diskann::core::Orchestrator`, injecting dependencies.
    - Oversees DuckDB-specific index metadata and storage.
  - **`DiskannScanState` (.hpp/.cpp):**
    - Manages state during a DuckDB scan operation.
  - **`DiskannBindData`, `DiskannCreateIndexInfo` (.hpp/.cpp):**
    - Custom data structures for DuckDB's operational phases (bind, create index).
  - **`diskann_extension.cpp`:** (Typically at `src/` level)
    - Handles DuckDB extension registration.
- **`diskann/core/`**
  - **`Orchestrator` (.hpp/.cpp):** (Class)
    - Central coordinator for core DiskANN operations.
    - Manages overall index state; interacts with `IStorageManager`, `IGraphManager`, `ISearcher`, `IndexConfig`.
    - Key for shadow implementation logic.
  - **`IndexConfig` (.hpp/.cpp):** (Struct or Class)
    - Stores DiskANN parameters (R, L_build, L_search, alpha, paths, format version, etc.).
    - Handles configuration loading/saving.
  - **`Distance` (.hpp/.cpp):** (Namespace with functions, or templated class/strategy)
    - Provides distance functions (L2, Cosine).
    - Example: `namespace diskann::core::distance { template<typename T> T l2_squared(...); }`
  - **`IStorageManager` / `StorageManager` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`IStorageManager`):** Contract for disk I/O.
    - **Class (`StorageManager`):** Concrete implementation.
    - **Responsibilities:** Abstracts read/write of nodes, vectors, metadata. Manages files, memory mapping, block allocation, different storage tiers (main, shadow).
    - Exemplar methods: `Initialize()`, `LoadMetadata()`, `SaveMetadata()`, `ReadNode()`, `WriteNode()`, `AllocateNodeBlock()`, `ReadVector()`, `CommitShadowWrites()`, `MergeShadowToMain()`.
  - **`IGraphManager` / `GraphManager` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`IGraphManager`):** Contract for Vamana graph structure management.
    - **Class (`GraphManager`):** Concrete implementation.
    - **Responsibilities:** Node addition, pruning, neighbor discovery, connectivity. Uses `IStorageManager`.
  - **`ISearcher` / `Searcher` (.hpp/.cpp):** (Interface and Class)
    - **Interface (`ISearcher`):** Contract for ANN searches.
    - **Class (`Searcher`):** Concrete implementation.
    - **Responsibilities:** Implements search algorithms. Uses `IGraphManager` and `IStorageManager`.
  - **`IQuantizer` / `Quantizer` (.hpp/.cpp):** (Interface and Class, if applicable)
    - Handles vector quantization.
  - **`types.hpp`:** (Within `diskann::core` namespace)
    - `node_id_t`, `StorageTier` (enum: `MAIN`, `SHADOW`).
    - `Node` struct (id, vector_data, neighbors).
    - `Candidate` struct (for search).
    - `GraphNodeView`, `BuildProgress`.
    - Custom enums (distance metrics, status codes).
  - **`utils.hpp` / `utils.cpp`:** (Functions within `diskann::core::utils` sub-namespace)
    - General utilities (logging, timers, file system helpers).
  - **`constants.hpp`:** (Constants within `diskann::core` namespace)
    - Global constants (defaults, magic numbers).

#### Distinction between Classes and Functional Approaches (Free Functions) & C++20 Considerations:

- **Classes:** For components with state and behavior (`Orchestrator`, `StorageManager`). Interfaces are key.
- **Structs:** For POD (`diskann::core::Node`, `diskann::core::Candidate`).
- **Namespaces with Free Functions:** For stateless utilities (`diskann::core::distance`, `diskann::core::utils`).
- **C++20:** Use `std::span`, Concepts, Ranges, `const` correctness, `noexcept`, `std::optional`, `std::expected`, Smart Pointers (`std::unique_ptr` for DI). Modules are future consideration.

### 3. Suggested Directory Structure

```
src/
├── diskann/
│   ├── core/                   // Core DiskANN logic, types, and utilities
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
│   │   │
│   │   ├── types.hpp           // Core data structures
│   │   ├── constants.hpp       // Core constants
│   │   │
│   │   └── utils/              // Utility functions (optional subdirectory)
│   │       ├── utils.hpp
│   │       └── utils.cpp
│   │
│   ├── duckdb/                 // Components interacting directly with DuckDB
│   │   ├── diskann_index.hpp
│   │   ├── diskann_index.cpp
│   │   ├── diskann_scan_state.hpp
│   │   ├── diskann_bind_data.hpp // And other DuckDB-specific structs
│   │   └── ...
│
├── diskann_extension.cpp       // Main extension loading file (registers the index)
└── CMakeLists.txt              // Main CMake file

// CMakeLists.txt within src/diskann/core/ would define a static library (e.g., diskann_core),
// encompassing all core logic, types, and utilities.
// This diskann_core library is then linked by the main DuckDB extension shared library.
```

### 4. Advantages of the Proposed Architecture

- **Enhanced Testability:** `diskann::core` is testable with mocked dependencies.
- **Clear Separation of Concerns (SoC):** `diskann::duckdb` for DuckDB integration, `diskann::core` for ANN logic, types, and utilities.
- **Improved Maintainability & Readability:** Smaller, focused components within `diskann::core`.
- **Modularity:** `diskann::core` forms a cohesive and potentially reusable unit.
- **Scalability of Development Efforts:** Clear boundaries for teamwork.

### 5. Salient Considerations and Recommended Best Practices

- **Interfaces (Abstract Base Classes):** Define for major `diskann::core` components (`IStorageManager`, etc.) for DI and mocking.

  ```
  // diskann/core/istorage_manager.hpp
  namespace diskann::core {
  class IStorageManager {
  public:
      virtual ~IStorageManager() = default;
      virtual bool Initialize(const std::string& base_path, const IndexConfig& config) = 0;
      // Note: Assuming Node and StorageTier are now part of diskann::core
      virtual std::optional<Node> ReadNode(node_id_t node_id, StorageTier tier = StorageTier::MAIN) = 0;
      virtual bool WriteNode(node_id_t node_id, const Node& node_data, StorageTier tier = StorageTier::SHADOW) = 0;
      // ... other pure virtual functions
  };
  } // namespace diskann::core
  ```

- **Dependency Injection:** `diskann::duckdb::DiskannIndex` creates `diskann::core::Orchestrator` and injects concrete dependencies (e.g., `std::unique_ptr<diskann::core::IStorageManager>`) via its constructor. This uses standard C++ features (interfaces, smart pointers) without needing external DI libraries.

- **Explicit Ownership and State Management:** `Orchestrator` owns high-level state and injected dependencies. Components like `StorageManager` own their specific resources (file handles). Use smart pointers.

- **Error Handling Strategies:** Exceptions for critical errors; `std::optional` or `std::expected` for recoverable/absence cases.

- **Concurrency Management:** `Orchestrator` for high-level control; component interfaces designed for thread safety.

- **Lifecycle Management:** `DiskannIndex` manages `Orchestrator` lifecycle (creation, operations, destruction).

### 6. Further Organizational and Planning Considerations

1. **Comprehensive Testing Strategy:** Unit tests (core components with mocks), integration tests (core, DuckDB extension), performance benchmarks, stress/durability tests.
2. **Configuration Management and Propagation:** `DiskannIndex` parses options to `diskann::core::IndexConfig`, passed to `Orchestrator` and then to its dependencies. `IndexConfig` includes on-disk format version.
3. **Shadow Implementation – Key Interface Considerations:** `diskann::core::IStorageManager` abstracts storage tiers. `Orchestrator` coordinates shadow operations. `diskann::core::IndexConfig` stores shadow state metadata.
4. **Build System and Modularity (CMake):** Build `diskann::core` (now including types, utils) as a single static library, linked by the main extension.
5. **On-Disk Format Versioning and Migration:** `diskann::core::IndexConfig` and `diskann::core::StorageManager` handle versioning. Strategy for supporting/upgrading older formats.
6. **Logging, Metrics, and Observability:** Define log levels. Log key operations and progress. Expose critical metrics.

### VII. Mapping Existing `lm_diskann` Files to Proposed Structure

This section provides guidance on how the functionality within your current `src/lm_diskann/` directory and related files is expected to map to the proposed architecture:

- **`src/lm_diskann/LmDiskannIndex.hpp`** (and its corresponding `.cpp` file, if separate):
  - **Maps to:** `src/diskann/duckdb/DiskannIndex.hpp` and `src/diskann/duckdb/DiskannIndex.cpp`.
- **`src/lm_diskann/LmDiskannScanState.hpp`**:
  - **Maps to:** `src/diskann/duckdb/DiskannScanState.hpp`.
- **`src/lm_diskann/config.hpp`, `src/lm_diskann/config.cpp`**:
  - **Maps to:** `src/diskann/core/IndexConfig.hpp` and `src/diskann/core/IndexConfig.cpp`.
- **`src/lm_diskann/storage.hpp`, `src/lm_diskann/storage.cpp`**:
  - **Maps to:** `src/diskann/core/IStorageManager.hpp` (interface) and `src/diskann/core/StorageManager.hpp`, `src/diskann/core/StorageManager.cpp`.
- **`src/lm_diskann/GraphOperations.hpp`, `src/lm_diskann/GraphOperations.cpp`**:
  - **Maps to:** Functionality largely integrated into `src/diskann/core/IGraphManager.hpp` and `src/diskann/core/GraphManager.hpp`, `src/diskann/core/GraphManager.cpp`.
- **`src/lm_diskann/NodeManager.hpp`, `src/lm_diskann/NodeManager.cpp`**:
  - **Maps to:** Functionality split between `diskann::core::GraphManager` (logical node management) and `diskann::core::StorageManager` (physical node block I/O).
- **`src/lm_diskann/NodeAccessors.hpp`, `src/lm_diskann/NodeAccessors.cpp`**:
  - **Maps to:** Likely internal helper methods/classes for `diskann::core::GraphManager` or `diskann::core::Searcher`, or part of `diskann::core::Node` / `diskann::core::GraphNodeView`.
- **`src/lm_diskann/search.hpp`, `src/lm_diskann/search.cpp`**:
  - **Maps to:** `src/diskann/core/ISearcher.hpp` and `src/diskann/core/Searcher.hpp`, `src/diskann/core/Searcher.cpp`.
- **`src/lm_diskann/distance.hpp`, `src/lm_diskann/distance.cpp`**:
  - **Maps to:** `src/diskann/core/Distance.hpp` (within `diskann::core::distance` sub-namespace).
- **`src/lm_diskann/ternary_quantization.hpp`**:
  - **Maps to:** `src/diskann/core/IQuantizer.hpp` and `src/diskann/core/Quantizer.hpp`, `src/diskann/core/Quantizer.cpp`.
- **`src/include/lm_diskann_extension.hpp`**:
  - **Maps to:** Contents integrated/forwarded as needed; main class def in `src/diskann/duckdb/DiskannIndex.hpp`.
- **`src/lm_diskann_extension.cpp`**:
  - **Maps to:** `src/diskann_extension.cpp` (top level of extension `src`).
- **`src/lm_diskann/CMakeLists.txt`**:
  - **Maps to:** Adapted for new structure. A top-level `CMakeLists.txt` in `src/`. A `CMakeLists.txt` in `src/diskann/core/` to build the `diskann_core` static library. `src/diskann/duckdb/` components compiled as part of the main extension library.

This mapping aims to consolidate logic into the new component structure, enhancing separation of concerns and testability.

### VIII. Further Organizational and Planning Considerations

1. **Comprehensive Testing Strategy:**
   - **Unit Tests:** Each class within `diskann::core` (e.g., `StorageManager`, `GraphManager`, `Searcher`, `Orchestrator` itself) should have thorough unit tests. Dependencies will be mocked using the defined interfaces (`IStorageManager`, etc.). Focus on testing individual methods, edge cases, and invariants. These tests should reside alongside the core code (e.g., in a `tests/` subdirectory within `diskann/core/`).
   - **Integration Tests (Core):** Test interactions between core components.
   - **Integration Tests (DuckDB Extension):** These tests will involve actual DuckDB queries.
   - **Performance Benchmarks:** Establish a suite of benchmarks.
   - **Stress Tests and Durability Tests:** Crucial for shadow implementation and disk-based aspects.
2. **Configuration Management and Propagation:**
   - The `diskann::duckdb::DiskannIndex` will parse `CREATE INDEX ... WITH (...)` options into a `diskann::core::IndexConfig` object.
   - This `IndexConfig` object is passed to the `Orchestrator` and its dependencies.
   - The `IndexConfig` should also include a field for the on-disk format version.
3. **Shadow Implementation – Key Interface Considerations:**
   - The `diskann::core::IStorageManager` interface will be central, abstracting storage tiers.
   - The `diskann::core::Orchestrator` will coordinate shadow operations.
   - The `diskann::core::IndexConfig` might need to store shadow store state metadata.
4. **Build System and Modularity (CMake):**
   - Build `diskann::core` (which now includes former `common` elements) as a static library.
   - The main `CMakeLists.txt` for the extension will link against this static library.
   - This structure facilitates separate compilation and testing of the core logic.
5. **On-Disk Format Versioning and Migration:**
   - `diskann::core::IndexConfig` (and persisted metadata) must include a version number.
   - `diskann::core::StorageManager` (or `Orchestrator`) must check this version.
   - Develop a strategy for supporting/upgrading older formats or erroring out.
6. **Logging, Metrics, and Observability:**
   - Define log levels. Log key operations and progress.
   - Expose critical metrics for performance analysis and debugging.

By proactively planning for these aspects, the development process will be more structured, potential issues can be identified earlier, and the resulting extension will be more robust, maintainable, and user-friendly.