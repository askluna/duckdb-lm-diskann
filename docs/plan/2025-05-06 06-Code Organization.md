## Formalized DiskANN C++ Architectural Specification for DuckDB Extension Integration

This document presents a formalized C++ architectural specification for the DiskANN DuckDB extension. The proposed structure is centered upon an `Orchestrator` class, situated within the `diskann` namespace, which is tasked with the management of core indexing logic. This design paradigm is intended to promote modularity, enhance testability, and ensure maintainability. The present revision simplifies the namespace structure to `duckdb` (for DuckDB interface components) and `diskann` (for core logic and common types), and adopts PascalCase for filenames primarily defining classes. It is important to note that placing extension-specific components directly within the `duckdb` namespace, while simplifying naming, carries an inherent risk of naming collisions with current or future core DuckDB components. This specification otherwise incorporates considerations pertaining to nomenclature, directory organization, and offers detailed recommendations for class design, the application of functional programming paradigms, and adherence to C++20 best practices.

### 1. Fundamental Design Principle: The `diskann::Orchestrator`

The `diskann::Orchestrator` class constitutes the central architectural element. The DuckDB-specific `DiskannIndex` class (residing in the `duckdb` namespace), which inherits from DuckDB's `Index` base class, will assume the following responsibilities:

- Implementation of the DuckDB `Index` interface, encompassing methods such as `Bind`, `InitializeScan`, `Scan`, `Append`, `Delete`, and `Verify`.
- Interaction with DuckDB's systemic components, including the catalog, storage manager, buffer manager, and transaction manager.
- Translation of DuckDB data structures and operational requests into a format comprehensible to the `diskann::Orchestrator`.
- Management of the lifecycle pertaining to the `diskann::Orchestrator` instance associated with a given index.

The `diskann::Orchestrator` is designed to encapsulate the algorithmic logic of the DiskANN implementation:

- **State Management:** It will possess ownership and control over the state of the DiskANN graph, including elements such as entry point(s), graph metadata, and configuration parameters. This remit extends to managing the file system paths for on-disk structures.
- **Core Operations:** It will implement high-level operations, including index construction, search execution, new vector insertion, and the handling of deletions and updates. The latter may involve sophisticated mechanisms, such as those delineated in the "Shadow Implementation.md" document.
- **Coordination:** It will delegate specific tasks to specialized components residing within the `diskann` namespace, such as `GraphManager`, `StorageManager`, and `Searcher`.
- **Testability:** The `Orchestrator` is engineered for isolated testing, independent of the DuckDB environment. Its dependencies, for instance `diskann::IStorageManager`, will be injectable, thereby facilitating the use of mock objects and enabling focused unit and integration testing of the core Approximate Nearest Neighbor (ANN) algorithm.

### 2. Proposed Component Architecture and Functional Responsibilities

#### Namespace Allocation:

- `duckdb`: Designated for components that directly interface with the DuckDB system (e.g., `duckdb::DiskannIndex`).
- `diskann`: Encompasses the core DiskANN algorithmic logic, shared data structures, and utility functions (e.g., `diskann::Orchestrator`, `diskann::Node`).

#### Component Delineation:

- **`diskann/duckdb/` (Files within this directory will use `namespace duckdb { ... }`)**
  - **`DiskannIndex.hpp`/`.cpp`:** (Class `DiskannIndex` in `duckdb` namespace)
    - Serves as the primary interface to the DuckDB system.
    - Manages DuckDB-specific tasks and data type conversions.
    - Instantiates and invokes the `diskann::Orchestrator`, injecting its dependencies.
    - Oversees DuckDB-specific index metadata and storage considerations.
  - **`DiskannScanState.hpp`/`.cpp`:** (Class `DiskannScanState` in `duckdb` namespace)
    - Manages state variables during a scan operation within the DuckDB context.
  - **`DiskannBindData.hpp`/`.cpp`, `DiskannCreateIndexInfo.hpp`/`.cpp`:** (Structs/Classes like `DiskannBindData`, `DiskannCreateIndexInfo` in `duckdb` namespace)
    - Custom data structures tailored for DuckDB's bind, create index, and other operational phases.
  - **`diskann_extension.cpp`:** (Typically located at the `src/` level or a designated extension loading point)
    - Handles DuckDB extension registration, definition of the index type, and associated functions or pragmas. (Classes/functions defined here for DuckDB registration might also be in the `duckdb` namespace or global namespace as per DuckDB extension guidelines).
- **`diskann/core/` and `diskann/common/` (Files within these directories will use `namespace diskann { ... }`)**
  - **`Orchestrator.hpp`/`.cpp`:** (Class `Orchestrator`, in `diskann` namespace)
    - Functions as the central coordinator for all core DiskANN operations.
    - Interacts with `diskann::IStorageManager`, `diskann::IGraphManager`, `diskann::ISearcher`, and `diskann::IndexConfig`.
  - **`IndexConfig.hpp`/`.cpp`:** (Struct or Class `IndexConfig`, in `diskann` namespace)
    - Stores all DiskANN operational parameters.
  - **`distance.hpp`:** (Namespace `diskann::distance` containing functions, or a templated class/strategy pattern in `diskann` namespace)
    - Provides a suite of distance functions.
    - Illustrative example: `namespace diskann { namespace distance { template<typename T> T l2_squared(const T* vec1, const T* vec2, uint32_t dim); } }`
  - **`IStorageManager.hpp`, `StorageManager.hpp`/`.cpp`:** (Interface `IStorageManager` and Class `StorageManager`, in `diskann` namespace)
    - **Interface (`diskann::IStorageManager`):** Defines the contractual obligations for all disk input/output operations.
    - **Class (`diskann::StorageManager`):** Provides the concrete implementation.
    - Exemplar methods: `Initialize(base_path, config)`, `LoadMetadata() -> diskann::IndexConfig`, `SaveMetadata(const diskann::IndexConfig& config)`, `ReadNode(diskann::node_id_t node_id, diskann::StorageTier tier = diskann::StorageTier::MAIN) -> std::optional<diskann::Node>`, etc.
  - **`IGraphManager.hpp`, `GraphManager.hpp`/`.cpp`:** (Interface `IGraphManager` and Class `GraphManager`, in `diskann` namespace)
  - **`ISearcher.hpp`, `Searcher.hpp`/`.cpp`:** (Interface `ISearcher` and Class `Searcher`, in `diskann` namespace)
  - **`IQuantizer.hpp`, `Quantizer.hpp`/`.cpp`:** (Interface `IQuantizer` and Class `Quantizer`, in `diskann` namespace, if applicable)
  - **`types.hpp` (from `diskann/common/` but types are in `diskann` namespace):**
    - `diskann::node_id_t`
    - `diskann::StorageTier`
    - `diskann::Node`
    - `diskann::Candidate`
    - `diskann::GraphNodeView`
    - `diskann::BuildProgress`
  - **`utils.hpp` / `utils.cpp` (from `diskann/common/` but functions/types are in `diskann` or `diskann::utils` namespace):**
    - General utility functions, e.g., `namespace diskann { namespace utils { ... } }`.
  - **`constants.hpp` (from `diskann/common/` but constants are in `diskann` namespace):**
    - Global constants, e.g., `namespace diskann { constexpr int MY_CONST = 5; }`.

#### Distinction between Classes and Functional Approaches (Free Functions) & C++20 Considerations:

- **Classes:** Employed for components that encapsulate substantial state and behavior (e.g., `diskann::Orchestrator`, `diskann::StorageManager`). Interfaces (e.g., `diskann::ISomethingManager`) are paramount.
- **Structs:** Primarily utilized for Plain Old Data (POD) (e.g., `diskann::Node`, `diskann::Candidate`).
- **Namespaces with Free Functions:**
  - Appropriate for stateless utility functions (e.g., `diskann::distance::l2_squared(...)`, `diskann::utils::log_message(...)`).
- **C++20 for Enhanced Testability and Modern Software Practices:** (Content remains the same, references to `std::span`, Concepts, Ranges, `const` correctness, `std::optional`, `std::expected`, Smart Pointers, Modules).

### 3. Suggested Directory Structure

The physical directory structure is updated to reflect PascalCase for filenames primarily defining classes:

```
src/
├── diskann/
│   ├── core/                   // Files here declare 'namespace diskann { ... }'
│   │   ├── Orchestrator.hpp
│   │   ├── Orchestrator.cpp
│   │   │
│   │   ├── IndexConfig.hpp
│   │   ├── IndexConfig.cpp
│   │   │
│   │   ├── distance.hpp        // Declares 'namespace diskann { namespace distance { ... } }'
│   │   │
│   │   ├── IStorageManager.hpp // Declares 'namespace diskann { ... }'
│   │   ├── StorageManager.hpp
│   │   ├── StorageManager.cpp
│   │   │
│   │   ├── IGraphManager.hpp  // Declares 'namespace diskann { ... }'
│   │   ├── GraphManager.hpp
│   │   ├── GraphManager.cpp
│   │   │
│   │   ├── ISearcher.hpp       // Declares 'namespace diskann { ... }'
│   │   ├── Searcher.hpp
│   │   ├── Searcher.cpp
│   │   │
│   │   ├── IQuantizer.hpp      // (Optional) Declares 'namespace diskann { ... }'
│   │   ├── Quantizer.hpp       // (Optional)
│   │   ├── Quantizer.cpp       // (Optional)
│   │
│   ├── duckdb/                 // Files here declare 'namespace duckdb { ... }' (with caution)
│   │   ├── DiskannIndex.hpp
│   │   ├── DiskannIndex.cpp
│   │   ├── DiskannScanState.hpp
│   │   ├── DiskannScanState.cpp
│   │   ├── DiskannBindData.hpp // And other similar PascalCase files
│   │   └── ...
│   │
│   └── common/                 // Files here declare 'namespace diskann { ... }' or sub-namespaces like 'diskann::utils'
│       ├── types.hpp           // Declares types in 'namespace diskann { ... }'
│       ├── utils.hpp           // Declares 'namespace diskann { namespace utils { ... } }'
│       ├── utils.cpp
│       └── constants.hpp       // Declares constants in 'namespace diskann { ... }'
│
├── diskann_extension.cpp       // Main extension loading file
└── CMakeLists.txt              // Main CMake file
```

### 4. Advantages of the Proposed Architecture

(Content remains largely the same, focusing on testability, SoC, maintainability, modularity, and scalability, independent of the specific top-level namespace choice for the interface layer, though the clarity of SoC might be slightly impacted by using the `duckdb` namespace directly for extension parts.)

- **Enhanced Testability:** The `diskann` module, particularly the `Orchestrator` and its components, remains highly testable.
- **Clear Separation of Concerns (SoC):** `duckdb::DiskannIndex` handles DuckDB integration; `diskann::Orchestrator` manages core logic.
- **Improved Maintainability & Readability:** Smaller, focused classes. Algorithmic changes are isolated within the `diskann` namespace.
- **Modularity:** The `diskann` namespace components form a reusable core.
- **Scalability of Development Efforts:** Clear boundaries.

### 5. Salient Considerations and Recommended Best Practices

- **Interfaces (Abstract Base Classes):**

  - Define abstract interfaces (e.g., `diskann::IStorageManager`) within the `diskann` namespace, typically in their own PascalCase header files (e.g., `IStorageManager.hpp`).

  - Illustrative Example:

    ```
    // diskann/core/IStorageManager.hpp
    namespace diskann {
    class IStorageManager {
    public:
        virtual ~IStorageManager() = default;
        virtual bool Initialize(const std::string& base_path, const IndexConfig& config) = 0;
        virtual std::optional<Node> ReadNode(node_id_t node_id, StorageTier tier = StorageTier::MAIN) = 0;
        virtual bool WriteNode(node_id_t node_id, const Node& node_data, StorageTier tier = StorageTier::SHADOW) = 0;
        // ... other pure virtual functions
    };
    } // namespace diskann
    ```

- **Dependency Injection:**

  - The `duckdb::DiskannIndex` is responsible for creating the `diskann::Orchestrator` and its concrete dependencies (which are also in the `diskann` namespace).
  - The `diskann::Orchestrator` receives dependencies (e.g., `std::unique_ptr<diskann::IStorageManager>`) via its constructor.

- **Explicit Ownership and State Management:** (Content remains the same, adjusted for new `diskann` namespace).

- **Error Handling Strategies:** (Content remains the same).

- **Concurrency Management:** (Content remains the same).

- **Lifecycle Management:**

  - `duckdb::DiskannIndex` manages the `diskann::Orchestrator`'s lifecycle.

### 6. Further Organizational and Planning Considerations

(Content remains the same, but internal references to namespaces for core components will now point to the unified `diskann` namespace, e.g., `diskann::StorageManager`, `diskann::IndexConfig`).

1. **Comprehensive Testing Strategy:**
   - Unit Tests: Each class within `diskann` (e.g., `diskann::StorageManager`, `diskann::Orchestrator`) should have thorough unit tests.
   - Integration Tests (Core): Test interactions between `diskann` components.
   - Integration Tests (DuckDB Extension): Test `duckdb::DiskannIndex` and its interaction with `diskann` components.
2. **Configuration Management and Propagation:**
   - `duckdb::DiskannIndex` parses options into a `diskann::IndexConfig` object.
3. **Shadow Implementation – Key Interface Considerations:**
   - `diskann::IStorageManager` interface is central.
   - `diskann::Orchestrator` coordinates.
   - `diskann::IndexConfig` stores metadata.
4. **Build System and Modularity (CMake):**
   - Build `diskann` components (formerly core and common) potentially as a single static library or multiple, linked by the main extension.
5. **On-Disk Format Versioning and Migration:**
   - `diskann::IndexConfig` and `diskann::StorageManager` handle versioning.
6. **Logging, Metrics, and Observability:** (Content remains the same).

By proactively planning for these aspects, the development process will be more structured, potential issues can be identified earlier, and the resulting extension will be more robust, maintainable, and user-friendly.