## DiskANN DuckDB Extension: C++ Architectural Specification

Version: 1.1

Date: 2025-05-07

**Abstract:** This document specifies the C++ architecture for the DiskANN DuckDB extension. The design prioritizes modularity, testability, and maintainability, centering on an `Orchestrator` class within the `diskann` namespace for core indexing logic. This specification details a two-namespace structure (`duckdb` for DuckDB interface components; `diskann` for core logic and common types), PascalCase for class-defining filenames, and explicit characterization of components regarding state, purity, immutability, and callback usage. Direct use of the `duckdb` namespace for extension-specific interface components, while simplifying naming, introduces a potential risk of naming collisions with core DuckDB components; this risk is acknowledged. Vector quantization (e.g., ternary quantization) is handled by specific utility components within the `diskann` namespace as needed, rather than through a dedicated general quantizer interface at the orchestrator level.

### 1. Fundamental Design Principle: The `diskann::Orchestrator`

The `diskann::Orchestrator` class is the central architectural component. It is instantiated and managed by the `duckdb::DiskannIndex` class, which serves as the primary integration point with the DuckDB system.

**1.1. `duckdb::DiskannIndex` Responsibilities:**

- Implementation of the DuckDB `Index` interface (e.g., `Bind`, `InitializeScan`, `Scan`, `Append`, `Delete`, `Verify`).
- Interaction with DuckDB systemic components (catalog, storage, buffer manager, transaction manager).
- Translation of DuckDB data structures and operational requests into invocations on the `diskann::Orchestrator`.
- Lifecycle management of the associated `diskann::Orchestrator` instance.

**1.2. `diskann::Orchestrator` Responsibilities:**

- **State Management:** Owns and controls the DiskANN graph's state, including entry points, metadata, configuration parameters, and paths to on-disk structures. (Primarily **Stateful**).
- **Core Operations:** Implements high-level operations: index construction, search execution, vector insertion, and data modification (deletions/updates), potentially utilizing mechanisms like a shadow store as detailed in auxiliary design documents.
- **Coordination:** Delegates tasks to specialized components within the `diskann` namespace (e.g., `GraphManager`, `StorageManager`, `Searcher`).
- **Testability:** Engineered for isolated testing. Dependencies (e.g., `diskann::IStorageManager`) are injected, enabling focused unit and integration testing of the core ANN algorithm using mock objects.

### 2. Component Architecture and Functional Responsibilities

#### 2.1. Namespace Allocation:

- `duckdb`: Designated for components directly interfacing with the DuckDB system (e.g., `duckdb::DiskannIndex`).
- `diskann`: Encompasses core DiskANN algorithmic logic, shared data structures, and utility functions (e.g., `diskann::Orchestrator`, `diskann::Node`).

#### 2.2. Component Delineation:

- **`diskann/duckdb/`** (Components within this directory will declare `namespace duckdb { ... }`)
  - **`DiskannIndex.hpp`/`.cpp`:** (Class `DiskannIndex`)
    - **Nature:** **Stateful** (manages `diskann::Orchestrator`, interacts with DuckDB state).
    - Primary DuckDB interface; translates DuckDB operations to `Orchestrator` calls; manages `Orchestrator` lifecycle and dependency injection.
  - **`DiskannScanState.hpp`/`.cpp`:** (Class `DiskannScanState`)
    - **Nature:** **Stateful** (holds state for an active scan operation).
  - **`DiskannBindData.hpp`/`.cpp`, `DiskannCreateIndexInfo.hpp`/`.cpp`:** (Structs/Classes like `DiskannBindData`)
    - **Nature:** Data containers; **Immutable** once populated during specific DuckDB phases (e.g., bind, create).
  - **`diskann_extension.cpp`:**
    - **Nature:** Contains **Stateless** registration functions (extension entry points).
- **`diskann/core/` and `diskann/common/`** (Components within these directories will declare `namespace diskann { ... }`)
  - **`Orchestrator.hpp`/`.cpp`:** (Class `Orchestrator`)
    - **Nature:** **Stateful** (manages overall index state, configuration, core component lifecycles).
    - Central coordinator; interacts with `diskann::IStorageManager`, `diskann::IGraphManager`, `diskann::ISearcher`, `diskann::IndexConfig`.
    - **Callbacks:** May provide `std::function` callbacks (e.g., `void(const diskann::BuildProgress&)>`) to managed components for progress reporting.
  - **`IndexConfig.hpp`/`.cpp`:** (Struct or Class `IndexConfig`)
    - **Nature:** Data container; ideally **Immutable** after initial setup and passed by `const&` or immutable shared pointer. Stores all operational parameters.
  - **`distance.hpp`:** (Namespace `diskann::distance`)
    - **Nature:** Contains **Stateless**, **Pure Functions** (e.g., `l2_squared`). Output depends solely on input vectors and dimension. Templated for various data types, potentially constrained by C++20 Concepts for type safety.
  - **`IStorageManager.hpp`, `StorageManager.hpp`/`.cpp`:** (Interface `IStorageManager`, Class `StorageManager`)
    - **Nature:** **Stateful** (manages file handles, caches, on-disk layout, shadow store state).
    - `IStorageManager`: Defines the contract for disk I/O.
    - `StorageManager`: Concrete stateful implementation. May utilize quantization utilities (e.g., from `ternary_quantization.hpp`) internally if vectors are stored in a quantized form.
    - **Callbacks:** May accept `std::function` for progress reporting on long operations (e.g., `MergeShadowToMain`).
  - **`IGraphManager.hpp`, `GraphManager.hpp`/`.cpp`:** (Interface `IGraphManager`, Class `GraphManager`)
    - **Nature:** **Stateful** (manages graph structure, Vamana algorithm state, metadata).
    - **Callbacks:** Methods like `BuildGraph` may accept `std::function<void(const diskann::BuildProgress&)>` for progress updates.
  - **`ISearcher.hpp`, `Searcher.hpp`/`.cpp`:** (Interface `ISearcher`, Class `Searcher`)
    - **Nature:** **Stateful** (manages search context: candidate lists, visited sets, query parameters).
    - **Callbacks:** May accept `std::function<bool(diskann::node_id_t)> filter_predicate` for dynamic filtering, or callbacks for incremental result processing. May utilize quantization utilities if query vectors or on-disk vectors need dequantization/quantization during search.
  - **`ternary_quantization.hpp`:** (Header providing quantization utilities/classes within `diskann` namespace)
    - **Nature:** Provides specific vector quantization logic (e.g., ternary quantization). Functions are likely **Stateless** and **Pure** if applying a fixed scheme, or a helper class might be **Stateless** if it doesn't learn parameters. Used by other components like `StorageManager` or `Searcher` as needed.
  - **`types.hpp`:** (Types within `diskann` namespace)
    - **Nature:** Defines data structures (e.g., `Node`, `Candidate`) and enumerations. Instances of `Node` read from disk are treated as **Immutable** within the scope of a single read operation.
  - **`utils.hpp`/`.cpp`:** (Utilities within `diskann` or `diskann::utils` namespace)
    - **Nature:** Primarily **Stateless** utility functions. Computational utilities should be **Pure Functions**. Logging functions inherently have side effects but should be deterministic.
  - **`constants.hpp`:** (Constants within `diskann` namespace)
    - **Nature:** Compile-time constants; inherently **Immutable** and **Stateless**.

#### 2.3. Distinction between Classes, Structs, and Functional Approaches:

- **Classes (Stateful Services & Complex Logic):** Employed for components encapsulating significant state and behavior (e.g., `diskann::Orchestrator`, `diskann::StorageManager`). Interfaces (`diskann::ISomethingManager`) are mandatory for these to enable polymorphism, dependency injection, and mocking.
- **Structs (Data Aggregation):** Used for Plain Old Data (POD) or simple data aggregation (e.g., `diskann::Node`, `diskann::IndexConfig`). Typically **Immutable** post-initialization or represent data snapshots.
- **Free Functions (Stateless Operations & Pure Computations):** Grouped in namespaces (e.g., `diskann::distance`, `diskann::utils`, functions within `ternary_quantization.hpp`). Appropriate for **Stateless** utilities and **Pure Functions**. C++20 Concepts are recommended for constraining template parameters in generic stateless functions.
- **`std::function` (Callbacks for Extensible Behavior):** Injected into methods of stateful classes to provide specific, often stateless, behavioral customizations (e.g., progress reporting, search filtering) without requiring subclassing.

#### 2.4. C++20 Considerations:

- **`std::span`:** For non-owning, often **Immutable**, views of contiguous data.
- **Concepts:** To define compile-time contracts for templated code, enhancing type safety and diagnostics, especially for stateless pure functions.
- **`const` correctness:** To enforce immutability and improve reasoning about state.
- Other features (Ranges, `std::optional`, `std::expected`, Smart Pointers) as applicable to enhance code safety and expressiveness.

### 3. Suggested Directory Structure

The physical directory structure reflects PascalCase for filenames primarily defining classes, and includes `ternary_quantization.hpp`:

```
src/
├── diskann/
│   ├── core/                   // Files declare 'namespace diskann { ... }'
│   │   ├── Orchestrator.hpp
│   │   ├── Orchestrator.cpp
│   │   ├── IndexConfig.hpp
│   │   ├── IndexConfig.cpp
│   │   ├── distance.hpp        // Declares 'namespace diskann { namespace distance { ... } }'
│   │   ├── IStorageManager.hpp
│   │   ├── StorageManager.hpp
│   │   ├── StorageManager.cpp
│   │   ├── IGraphManager.hpp
│   │   ├── GraphManager.hpp
│   │   ├── GraphManager.cpp
│   │   ├── ISearcher.hpp
│   │   ├── Searcher.hpp
│   │   ├── Searcher.cpp
│   │   ├── ternary_quantization.hpp // Provides specific quantization logic
│   │
│   ├── duckdb/                 // Files declare 'namespace duckdb { ... }'
│   │   ├── DiskannIndex.hpp
│   │   ├── DiskannIndex.cpp
│   │   ├── DiskannScanState.hpp
│   │   ├── DiskannScanState.cpp
│   │   ├── DiskannBindData.hpp // And other similar PascalCase files
│   │   └── ...
│   │
│   └── common/                 // Files declare 'namespace diskann { ... }' or 'diskann::utils'
│       ├── types.hpp
│       ├── utils.hpp
│       ├── utils.cpp
│       └── constants.hpp
│
├── diskann_extension.cpp       // Main extension loading file
└── CMakeLists.txt              // Main CMake file
```

### 4. Architectural Advantages

- **Enhanced Testability:** Stateful components are testable via DI and interfaces; stateless pure functions are inherently testable.
- **Clear Separation of Concerns:** `duckdb::DiskannIndex` handles DuckDB integration; `diskann::Orchestrator` manages core stateful logic; stateless computations are isolated.
- **Improved Maintainability & Readability:** Clear distinction between stateful services and stateless utilities.
- **Modularity:** `diskann` namespace components form a reusable core.
- **Scalability of Development Efforts:** Defined component boundaries.

### 5. Salient Design Principles and Best Practices

- **Interfaces for Stateful Services:** Define abstract interfaces (e.g., `diskann::IStorageManager`) using abstract base classes with pure virtual functions for all major stateful services within the `diskann` namespace.
- **Dependency Injection (Constructor Injection):** `duckdb::DiskannIndex` instantiates and injects concrete implementations of `diskann` interfaces (e.g., `std::unique_ptr<diskann::IStorageManager>`) into the constructor of dependent components like `diskann::Orchestrator`. This is the primary mechanism for decoupling and enabling mock injection for tests.
- **Explicit Ownership and State Management:** Clearly define ownership using smart pointers (`std::unique_ptr` for exclusive ownership of injected dependencies). Stateful components manage their internal state and resources.
- **Error Handling:** Employ exceptions for unrecoverable errors. Use `std::optional` or `std::expected` (C++23 or library) for recoverable errors or optional return values.
- **Concurrency Management:** Stateful components, particularly `diskann::StorageManager` and `diskann::GraphManager`, must be designed for thread safety if concurrent access is anticipated. Concurrency control mechanisms (e.g., locks) will be managed by the `Orchestrator` or within components as appropriate.
- **Lifecycle Management:** `duckdb::DiskannIndex` manages the lifecycle of the `diskann::Orchestrator` and, through it, other core components. Destruction ensures resource release and data flushing.

### 6. Further Organizational and Planning Considerations

The following strategic points guide implementation based on this architecture:

1. **Comprehensive Testing Strategy:**
   - **Unit Tests:** For all `diskann` components. Stateless pure functions tested with diverse inputs. Stateful classes tested by mocking injected dependencies.
   - **Core Integration Tests:** Validate interactions between `diskann` components.
   - **DuckDB Extension Integration Tests:** SQL-based tests (`sqllogictest`) verifying end-to-end functionality.
   - **Performance Benchmarks & Stress Tests:** For indexing, search, and durability, especially concerning the shadow mechanism.
2. **Configuration Management:** `duckdb::DiskannIndex` parses SQL options into an immutable `diskann::IndexConfig` object, propagated to relevant `diskann` components.
3. **Shadow Implementation Integration:** The `diskann::IStorageManager` interface is critical for abstracting interactions with main and shadow stores. The `diskann::Orchestrator` coordinates shadow operations.
4. **Build System (CMake):** `diskann` components (core and common) built as static libraries, linked by the main extension library.
5. **On-Disk Format Versioning:** `diskann::IndexConfig` and `diskann::StorageManager` will manage on-disk format versions to ensure compatibility and support migration paths.
6. **Logging and Observability:** Implement structured logging with configurable levels. Expose key operational metrics for monitoring and debugging.

This specification provides the architectural blueprint for a robust, maintainable, and testable DiskANN extension for DuckDB.