## DiskANN DuckDB Extension: C++ Architectural Specification

Version: 1.3

Date: 2025-05-07

**Abstract:** This document specifies the C++ architecture for the DiskANN DuckDB extension. The design prioritizes modularity, testability, and maintainability. Core indexing logic within the `diskann` namespace is structured using **C++20 modules**, centering on an `Orchestrator` module. DuckDB-facing components (in the `duckdb` namespace) and the main extension entry point (`diskann_extension.cpp`) will use traditional `#include` for DuckDB headers and `import` `diskann` modules. This specification details a two-namespace structure, PascalCase for class-defining module interface filenames (e.g., `Orchestrator.cppm`), and explicit characterization of components regarding state, purity, immutability, and callback usage. Direct use of the `duckdb` namespace for extension-specific interface components, while simplifying naming, introduces a potential risk of naming collisions with core DuckDB components; this risk is acknowledged. Vector quantization is handled by specific utility components within the `diskann` namespace.

### 1. Fundamental Design Principle: The `diskann::Orchestrator` Module

The `diskann::Orchestrator` module (specifically the `diskann::Orchestrator` class it exports) is the central architectural component. It is instantiated and managed by the `duckdb::DiskannIndex` class, which serves as the primary integration point with the DuckDB system.

**1.1. `duckdb::DiskannIndex` Responsibilities:**

- Implementation of the DuckDB `Index` interface (e.g., `Bind`, `InitializeScan`, `Scan`, `Append`, `Delete`, `Verify`).
- Interaction with DuckDB systemic components (catalog, storage, buffer manager, transaction manager) via traditional `#include` of DuckDB headers.
- Translation of DuckDB data structures and operational requests into invocations on the imported `diskann::Orchestrator` class.
- Lifecycle management of the associated `diskann::Orchestrator` instance.

**1.2. `diskann::Orchestrator` Module Responsibilities (exporting `diskann::Orchestrator` class):**

- **State Management:** Owns and controls the DiskANN graph's state, including entry points, metadata, configuration parameters, and paths to on-disk structures. (Primarily **Stateful**).
- **Core Operations:** Implements high-level operations: index construction, search execution, vector insertion, and data modification (deletions/updates), potentially utilizing mechanisms like a shadow store.
- **Coordination:** Delegates tasks to other imported `diskann` modules/components (e.g., `GraphManager`, `StorageManager`, `Searcher` classes from their respective modules).
- **Testability:** Engineered for isolated testing. Dependencies (e.g., interfaces like `diskann::IStorageManager` imported from other modules) are injected, enabling focused unit and integration testing of the core ANN algorithm using mock objects.

### 2. Component Architecture and Functional Responsibilities

#### 2.1. Namespace Allocation:

- `duckdb`: Designated for components directly interfacing with the DuckDB system (e.g., `duckdb::DiskannIndex`). These use traditional `#include`.
- `diskann`: Encompasses core DiskANN algorithmic logic, shared data structures, and utility functions. All components within this namespace are structured as **C++20 modules**.

#### 2.2. Component Delineation (Modules and Classes):

- **`diskann/duckdb/`** (Components within this directory will declare `namespace duckdb { ... }` and use `#include`)
  - **`DiskannIndex.hpp`/`.cpp`:** (Class `DiskannIndex`)
    - **Nature:** **Stateful**.
    - Primary DuckDB interface; uses `#include` for DuckDB headers; `import`s `diskann.Orchestrator`, `diskann.IndexConfig`, etc.
  - **`DiskannScanState.hpp`/`.cpp`:** (Class `DiskannScanState`)
    - **Nature:** **Stateful**.
  - **`DiskannBindData.hpp`/`.cpp`, `DiskannCreateIndexInfo.hpp`/`.cpp`:** (Structs/Classes like `DiskannBindData`)
    - **Nature:** Data containers; **Immutable** once populated.
  - **`diskann_extension.cpp`:**
    - **Nature:** Contains **Stateless** registration functions. Uses `#include "duckdb.hpp"`; `import`s necessary `diskann` modules (likely via `duckdb::DiskannIndex`).
- **`diskann/core/` and `diskann/common/`** (Components are C++20 modules, e.g., `export module diskann.ComponentName;`)
  - **`Orchestrator.cppm`:** (Module `diskann.Orchestrator`, exports class `Orchestrator`)
    - **Nature:** **Stateful**.
    - Central coordinator; `import`s `diskann.IStorageManager`, `diskann.IGraphManager`, `diskann.ISearcher`, `diskann.IndexConfig`.
    - **Callbacks:** May provide `std::function` callbacks to managed components.
  - **`IndexConfig.cppm`:** (Module `diskann.IndexConfig`, exports struct/class `IndexConfig`)
    - **Nature:** Data container; ideally **Immutable** after setup.
  - **`distance.cppm`:** (Module `diskann.distance`, exports stateless, pure functions)
    - **Nature:** Contains **Stateless**, **Pure Functions** (e.g., `l2_squared`). Templated, potentially constrained by C++20 Concepts.
  - **`IStorageManager.cppm`, `StorageManager.cppm` (and `StorageManager_impl.cpp` if separate):** (Module `diskann.StorageManager`, exports interface `IStorageManager` and class `StorageManager`)
    - **Nature:** `StorageManager` class is **Stateful**.
    - `IStorageManager`: Defines the contract.
    - `StorageManager`: Concrete implementation. May `import diskann.ternary_quantization` internally.
    - **Callbacks:** May accept `std::function` for progress reporting.
  - **`IGraphManager.cppm`, `GraphManager.cppm` (and `GraphManager_impl.cpp` if separate):** (Module `diskann.GraphManager`, exports interface `IGraphManager` and class `GraphManager`)
    - **Nature:** `GraphManager` class is **Stateful**.
    - **Callbacks:** May accept `std::function` for progress updates.
  - **`ISearcher.cppm`, `Searcher.cppm` (and `Searcher_impl.cpp` if separate):** (Module `diskann.Searcher`, exports interface `ISearcher` and class `Searcher`)
    - **Nature:** `Searcher` class is **Stateful**.
    - **Callbacks:** May accept `std::function` for filtering or incremental results. May `import diskann.ternary_quantization`.
  - **`ternary_quantization.cppm`:** (Module `diskann.ternary_quantization`, exports utility functions/classes)
    - **Nature:** Provides specific vector quantization logic. Functions likely **Stateless** and **Pure**.
  - **`types.cppm`:** (Module `diskann.types`, exports types like `Node`, `Candidate`, `StorageTier`)
    - **Nature:** Defines data structures and enumerations. Instances of `Node` are treated as **Immutable** post-read.
  - **`utils.cppm`:** (Module `diskann.utils`, exports utility functions)
    - **Nature:** Primarily **Stateless** utility functions; computational utilities should be **Pure**.
  - **`constants.cppm`:** (Module `diskann.constants`, exports compile-time constants)
    - **Nature:** **Immutable** and **Stateless**.

#### 2.3. Distinction between Classes, Structs, and Functional Approaches (within Modules):

- **Exported Classes from Modules (Stateful Services & Complex Logic):** Employed for components encapsulating significant state and behavior (e.g., `diskann::Orchestrator` class exported from the `diskann.Orchestrator` module). Interfaces (e.g., `diskann::IStorageManager` exported from `diskann.StorageManager` module) are key exports, defining contracts for these stateful services.
- **Exported Structs from Modules (Data Aggregation):** Used for Plain Old Data (POD) or simple data aggregation (e.g., `diskann::Node` exported from `diskann.types` module). Typically **Immutable** post-initialization or represent data snapshots.
- **Exported Free Functions from Modules (Stateless Operations & Pure Computations):** Grouped in dedicated modules (e.g., functions in `diskann.distance` module, utilities in `diskann.utils` module). Appropriate for **Stateless** utilities and **Pure Functions**.
- **`std::function` (Callbacks):** Used within methods of exported classes to allow injection of specific, often stateless, behavioral customizations without altering the module's primary exported class interface or requiring subclassing.

#### 2.4. C++20 Considerations:

- **Modules:** This is a foundational aspect of the `diskann` internal architecture. All internal `diskann` components are structured as C++20 modules (e.g., `diskann.Orchestrator`, `diskann.StorageManager`, `diskann.types`). This approach is chosen for improved organization, stronger encapsulation (via explicit `export` statements), and the potential for faster compilation of internal components by replacing textual inclusion with Binary Module Interface (BMI) consumption. Module units can still `#include` necessary traditional headers (e.g., standard library, or DuckDB headers if an internal `diskann` module has a legitimate, direct need for DuckDB types not passed through its own module interface).
- **`std::span`:** For non-owning, often **Immutable**, views of contiguous data.
- **Concepts:** To define compile-time contracts for templated code, particularly for generic functions exported from modules like `diskann.distance`.
- **`const` correctness:** To enforce immutability for data structures and parameters where appropriate.
- Other features (Ranges, `std::optional`, `std::expected`, Smart Pointers) as applicable to enhance code safety and expressiveness within module implementations.

### 3. Suggested Directory Structure (with Module Files)

Module interface files typically use `.cppm` (or compiler-specific alternatives). Implementation can be in the `.cppm` or separate `.cpp` files that are part of the module.

```
src/
├── diskann/
│   ├── core/                   // Files declare 'export module diskann.ComponentName;'
│   │   ├── Orchestrator.cppm
│   │   ├── IndexConfig.cppm
│   │   ├── distance.cppm
│   │   ├── IStorageManager.cppm
│   │   ├── StorageManager.cppm // May contain class definition or be interface-only
│   │   ├── StorageManager_impl.cpp // Optional: Implementation if not in .cppm
│   │   ├── IGraphManager.cppm
│   │   ├── GraphManager.cppm
│   │   ├── GraphManager_impl.cpp // Optional
│   │   ├── ISearcher.cppm
│   │   ├── Searcher.cppm
│   │   ├── Searcher_impl.cpp   // Optional
│   │   ├── ternary_quantization.cppm
│   │
│   ├── duckdb/                 // Files declare 'namespace duckdb { ... }' and use #include
│   │   ├── DiskannIndex.hpp
│   │   ├── DiskannIndex.cpp
│   │   ├── DiskannScanState.hpp
│   │   ├── DiskannScanState.cpp
│   │   ├── DiskannBindData.hpp
│   │   └── ...
│   │
│   └── common/                 // Files declare 'export module diskann.ComponentName;'
│       ├── types.cppm
│       ├── utils.cppm
│       ├── constants.cppm
│
├── diskann_extension.cpp       // Main extension loading file (#includes and imports)
└── CMakeLists.txt              // Main CMake file
```

### 4. Architectural Advantages

The adoption of C++20 modules for internal `diskann` components, alongside the established principles, yields several advantages:

- **Enhanced Testability:** Stateful service modules (e.g., `diskann.Orchestrator`) are testable by injecting mock implementations of their imported interface dependencies (e.g., a mock `diskann::IStorageManager`). Stateless pure functions exported from modules (e.g., `diskann.distance`) are inherently testable with varied inputs. Modules provide clean boundaries for unit testing.
- **Clear Separation of Concerns (SoC):** `duckdb::DiskannIndex` handles DuckDB integration using traditional includes. The `diskann` namespace, structured as modules, encapsulates all core ANN logic. Modules enforce stronger compile-time separation than headers alone, as only explicitly exported entities are visible externally.
- **Improved Maintainability & Readability:** Explicit `import` statements clarify dependencies between internal `diskann` components. The `export` keyword clearly defines the public API of each module, making it easier to understand component contracts and reducing the impact of internal refactoring.
- **Modularity and Encapsulation:** C++20 modules provide superior encapsulation. Implementation details not `export`ed from a module are truly hidden, preventing unintended use and macro leakage. This makes `diskann` components more self-contained and reusable.
- **Potential Build Performance Gains:** For the internal `diskann` components, compiling module interfaces to BMIs once and having importers consume these BMIs can lead to faster overall build times compared to repeated textual processing of headers, especially as the codebase grows.
- **Scalability of Development Efforts:** Defined module interfaces serve as stable contracts between different parts of the `diskann` system, allowing developers to work on different modules with reduced risk of interference.

### 5. Salient Design Principles and Best Practices (in a Modular Context)

The core design principles are maintained and applied within the C++20 module structure:

- **Interfaces for Stateful Services (Exported from Modules):** Abstract interfaces (e.g., `diskann::IStorageManager` class) are defined and `export`ed from their respective modules (e.g., `diskann.StorageManager` module). These serve as the contracts for stateful services.
- **Dependency Injection (Constructor Injection of Imported Interfaces):** Components like `duckdb::DiskannIndex` or higher-level `diskann` modules (e.g., `diskann.Orchestrator`) will `import` modules that export interfaces (e.g., `import diskann.IStorageManager;`). They then inject concrete implementations (which might also be `import`ed from their defining modules or constructed locally if the concrete class is exported) into dependent components via constructors. For example, `diskann::Orchestrator`'s constructor would take `std::unique_ptr<diskann::IStorageManager>`.
- **Explicit Ownership and State Management:** Ownership of resources and injected dependencies (via `std::unique_ptr`) is clearly managed within each module's components. Stateful modules manage their internal state.
- **Error Handling:** Exceptions for unrecoverable errors and `std::optional` or `std::expected` for recoverable errors/optional values are used within module implementations. Error types, if custom, may be exported from a common module like `diskann.types`.
- **Concurrency Management:** Stateful modules like `diskann.StorageManager` and `diskann.GraphManager` must be designed for thread safety if concurrent access is anticipated. Concurrency control mechanisms will be encapsulated within these modules or coordinated by the `diskann.Orchestrator` module.
- **Lifecycle Management:** `duckdb::DiskannIndex` manages the lifecycle of the `diskann::Orchestrator` instance. The `Orchestrator` module, in turn, manages the lifecycle of the core service components it imports and instantiates. Smart pointers ensure proper destruction and resource release.

### 6. Further Organizational and Planning Considerations (with Modules)

The adoption of C++20 modules influences several planning aspects:

1. **Comprehensive Testing Strategy:**
   - **Unit Tests:** Each `diskann` module will have corresponding unit tests. Tests for a module like `diskann.Orchestrator` will `import diskann.Orchestrator;` and `import` the interface modules for its dependencies (e.g., `diskann.IStorageManager;`), then inject mock implementations.
   - **Core Integration Tests:** These will `import` multiple `diskann` modules to test their interactions.
   - **DuckDB Extension Integration Tests:** SQL-based tests (`sqllogictest`) remain crucial for end-to-end validation.
2. **Configuration Management:** The `diskann.IndexConfig` module exports the `IndexConfig` structure. Instances are populated by `duckdb::DiskannIndex` and passed to the `diskann.Orchestrator` module, which then propagates it to other `diskann` modules that `import diskann.IndexConfig`.
3. **Shadow Implementation Integration:** The design relies heavily on the contract defined by the `diskann.IStorageManager` interface, exported from its module. The `diskann.Orchestrator` module `import`s this interface and coordinates shadow operations through it.
4. **Build System (CMake):**
   - This is a critical consideration. CMake configuration must fully support C++20 modules for the `diskann` components. This involves:
     - Using `target_sources` with `FILE_SET TYPE CXX_MODULES` to identify module interface units (e.g., `.cppm`) and implementation units.
     - Ensuring CMake correctly deduces or is explicitly told about dependencies between modules (e.g., `diskann.Orchestrator` `import`s `diskann.IStorageManager`) to build Binary Module Interfaces (BMIs) in the correct order.
     - Integrating these CMake module settings into DuckDB's extension build infrastructure. This might require custom CMake functions or adapting existing DuckDB extension templates.
     - Requires a sufficiently modern CMake version (e.g., 3.25+, ideally 3.28+) and compiler toolchain (GCC, Clang, MSVC) with robust and compatible module support.
5. **On-Disk Format Versioning:** The `diskann.IndexConfig` module's exported `IndexConfig` type will include versioning information. The `diskann.StorageManager` module will be responsible for checking and handling these versions.
6. **Logging and Observability:** Logging utilities may be provided by a `diskann.utils` module and `import`ed by other `diskann` modules. Metrics exposure will be coordinated through the `diskann.Orchestrator` or specific service modules.

This specification provides the architectural blueprint for a robust, maintainable, and testable DiskANN extension for DuckDB, leveraging C++20 modules for its internal `diskann` components to achieve stronger encapsulation and potentially more efficient builds.