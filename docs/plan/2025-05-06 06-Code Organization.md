## DiskANN DuckDB Extension: C++ Architectural Specification

Version: 1.8

Date: 2025-05-07

**Abstract:** This document specifies the C++ architecture for the DiskANN DuckDB extension, adopting a **C++20 modules-first approach** for all custom logic. Core indexing components reside in modules within the `diskann` namespace (e.g., `diskann.Orchestrator`). DuckDB-facing integration components also reside in modules, typically within the `duckdb` namespace (e.g., `duckdb.DiskannIndex`), which `#include` DuckDB's traditional headers and `import` `diskann` core modules. The main extension entry point (`diskann_extension.cpp`) remains a traditional C++ file, using `#include` for DuckDB and `import` for the primary DuckDB-facing module. A strict unidirectional dependency is enforced: `diskann` core/common/store modules must not depend on `duckdb` namespace modules. Interactions required by the `diskann` core with DuckDB-specific resources (e.g., for shadow table operations) are abstracted via interfaces like `diskann::store::IShadowStorageService`, defined in `diskann/store/` modules. Implementations of these interfaces that depend on DuckDB are provided by modules in `diskann/duckdb/`. This specification details namespace organization, PascalCase for module interface filenames defining classes (e.g., `Orchestrator.cppm`), and component characteristics.

### 1. Fundamental Design Principle: The `diskann::Orchestrator` Module

The `diskann::Orchestrator` module (exporting the `diskann::Orchestrator` class) is the central architectural component for core ANN logic. It is instantiated and managed by the `duckdb::DiskannIndex` class (exported from the `duckdb.DiskannIndex` module), which serves as the primary integration point with the DuckDB system.

**1.1. `duckdb.DiskannIndex` Module Responsibilities (exporting `duckdb::DiskannIndex` class):**

- Implementation of the DuckDB `Index` interface. This module will `#include` necessary DuckDB headers.
- Interaction with DuckDB systemic components (catalog, storage, etc.) via included DuckDB APIs.
- Translation of DuckDB data structures and requests into invocations on the imported `diskann::Orchestrator` class.
- Creation and injection of concrete implementations for platform-specific interfaces (e.g., `diskann::store::IShadowStorageService`) required by the `diskann` core. These implementations will also be modules within the `duckdb` namespace (e.g., `duckdb.DiskannShadowStorageService`).
- Lifecycle management of the `diskann::Orchestrator` instance and its direct dependencies.

**1.2. `diskann.Orchestrator` Module Responsibilities (exporting `diskann::Orchestrator` class):**

- **State Management:** Owns and controls the DiskANN graph's state (entry points, metadata, configuration, on-disk paths). (Primarily **Stateful**).
- **Core Operations:** Implements high-level indexing operations (build, search, insert, update, delete), including shadow store interactions via injected services.
- **Coordination:** Delegates tasks to other imported `diskann` modules (e.g., `diskann.GraphManager`, `diskann.StorageManager`, `diskann.Searcher`).
- **Testability:** Dependencies (interfaces like `diskann.IStorageManager`, `diskann.store.IShadowStorageService`) are imported and injected, enabling isolated testing with mocks.

### 2. Component Architecture and Functional Responsibilities

#### 2.1. Namespace Allocation and Dependency Direction:

- `duckdb`: Namespace for modules directly interfacing with DuckDB (e.g., `duckdb.DiskannIndex`, `duckdb.DiskannShadowStorageService`). These modules `#include` DuckDB headers and `import` `diskann` modules.
- `diskann`: Namespace for core DiskANN logic, common types, and platform-agnostic store interfaces. All components are C++20 modules.
  - `diskann::store`: Sub-namespace for modules defining abstract interfaces for storage and platform services (e.g., `diskann.store.IShadowStorageService`, `diskann.store.IFileSystem`).
- **Strict Unidirectional Dependency:** `diskann` core, common, and store modules **must not** depend on (import or include) `duckdb` namespace modules or DuckDB headers directly. Dependency flows from `diskann_extension.cpp` -> `duckdb` modules -> `diskann` modules.

#### 2.2. Component Delineation (All custom C++ code, except `diskann_extension.cpp`, is modular):

- **`diskann_extension.cpp`:** (Traditional C++ source file)
  - **Nature:** **Stateless** registration functions (C-API entry points).
  - Uses `#include "duckdb.hpp"`.
  - `import`s the primary integration module (e.g., `import duckdb.DiskannIndex;`) to register the index.
- **`diskann/duckdb/`** (Components are C++20 modules, typically in `duckdb` namespace. They `#include` DuckDB headers and `import diskann` modules.)
  - **`DiskannIndex.cppm`:** (Module e.g., `duckdb.DiskannIndex`, exports class `DiskannIndex`)
    - **Nature:** **Stateful**. Primary DuckDB interface logic. Instantiates and injects dependencies like `duckdb.DiskannShadowStorageService` into `diskann.Orchestrator`.
  - **`DiskannShadowStorageService.cppm`:** (Module e.g., `duckdb.DiskannShadowStorageService`, exports class `DiskannShadowStorageService` implementing `diskann::store::IShadowStorageService`)
    - **Nature:** **Stateful**. Provides DuckDB-backed implementation for shadow storage operations using DuckDB APIs.
  - **`DiskannScanState.cppm`:** (Module e.g., `duckdb.DiskannScanState`, exports class `DiskannScanState`)
    - **Nature:** **Stateful**.
  - **`DiskannBindData.cppm`, `DiskannCreateIndexInfo.cppm`:** (Modules exporting structs/classes)
    - **Nature:** Data containers; **Immutable** once populated.
- **`diskann/core/`, `diskann/common/`, `diskann/store/`** (Components are C++20 modules in the `diskann` namespace. These modules **do not** depend on `duckdb` namespace modules or DuckDB headers.)
  - **`Orchestrator.cppm`:** (Module `diskann.Orchestrator`, exports class `Orchestrator`)
    - **Nature:** **Stateful**. Imports `diskann.IStorageManager`, `diskann.IGraphManager`, `diskann.ISearcher`, `diskann.IndexConfig`, and `diskann.store.IShadowStorageService` (interface).
  - **`IndexConfig.cppm`:** (Module `diskann.IndexConfig`, exports `IndexConfig`)
    - **Nature:** Data container; ideally **Immutable**.
  - **`distance.cppm`:** (Module `diskann.distance`, exports pure functions)
    - **Nature:** **Stateless**, **Pure Functions**.
  - **`IShadowStorageService.cppm`:** (In `diskann/store/`, module `diskann.store_interfaces` or `diskann.store.IShadowStorageService`, exports interface `diskann::store::IShadowStorageService`)
    - **Nature:** Defines abstract interface for transactional operations related to the shadow store.
  - **`IFileSystem.cppm`:** (Optional, in `diskann/store/`, module `diskann.store_interfaces` or `diskann.store.IFileSystem`, exports interface `diskann::store::IFileSystem`)
    - **Nature:** Defines abstract interface for raw file system operations for `graph.lmd`.
  - **`IStorageManager.cppm`\**:\**** (Module `diskann.storage_interfaces` or `diskann.IStorageManager`, exports interface `diskann::IStorageManager`)
    - ***\*Nature:\**** Defines contract for managing the main Vamana graph file (`graph.lmd`) and logical shadow block interactions.
  - **`StorageManager.cppm`:** (Module `diskann.StorageManager`, exports class `StorageManager` implementing `diskann::IStorageManager`)
    - **Nature:** **Stateful**. `import`s and uses `diskann.store.IShadowStorageService` for `diskann_store.duckdb` interactions. Manages `graph.lmd` I/O (potentially via `diskann.store.IFileSystem` or standard C++ file I/O).
  - **`IGraphManager.cppm`, `GraphManager.cppm`:** (Modules `diskann.graph_interfaces` and `diskann.GraphManager`)
    - **Nature:** `GraphManager` class is **Stateful**.
  - **`ISearcher.cppm`, `Searcher.cppm`:** (Modules `diskann.search_interfaces` and `diskann.Searcher`)
    - **Nature:** `Searcher` class is **Stateful**.
  - **`ternary_quantization.cppm`:** (Module `diskann.ternary_quantization`)
    - **Nature:** **Stateless**, **Pure Functions** or stateless helper classes.
  - **`types.cppm`:** (Module `diskann.types`)
    - **Nature:** Defines data structures; **Immutable** post-read/creation.
  - **`utils.cppm`:** (Module `diskann.utils`)
    - **Nature:** Primarily **Stateless**, **Pure Functions**.
  - **`constants.cppm`:** (Module `diskann.constants`)
    - **Nature:** **Immutable**, **Stateless**.

#### 2.3. Module Implementation Details:

Module interface units (`.cppm`) declare exported entities. Implementations can reside within these `.cppm` files or in separate `.cpp` files that are explicitly declared as part of that module (e.g., by starting with `module diskann.ComponentName;`). The `_impl.cpp` suffix is not utilized.

#### 2.4. Distinction between Classes, Structs, and Functional Approaches (within Modules):

- ***\*Exported Classes from Modules (Stateful Services & Complex Logic):\**** Components like `diskann::Orchestrator` (from `diskann.Orchestrator` module) and `diskann::StorageManager` (from `diskann.StorageManager` module) encapsulate significant state and behavior. Their public contracts are defined by C++ classes, often implementing interfaces (like `diskann::IStorageManager` exported from its own module or `diskann.storage_interfaces`).
- ***\*Exported Interfaces from Modules (Contracts):\**** Abstract interfaces like `diskann::IStorageManager` and `diskann::store::IShadowStorageService` are defined and `export`ed from their respective modules to establish contracts for services. This enables polymorphism and dependency injection across module boundaries.
- ***\*Exported Structs from Modules (Data Aggregation):\**** Used for data aggregation (e.g., `diskann::Node` exported from `diskann.types` module). Typically ***\*Immutable\**** post-initialization or represent data snapshots.
- ***\*Exported Free Functions from Modules (Stateless Operations & Pure Computations):\**** Grouped in dedicated modules (e.g., functions in `diskann.distance` module, utilities in `diskann.utils` module). These are appropriate for ***\*Stateless\**** utilities and ***\*Pure Functions\****, promoting reusability and testability.
- **`std::function` \**(Callbacks for Extensible Behavior):\**** Utilized within methods of exported classes to allow injection of specific, often stateless, behavioral customizations (e.g., progress reporting from `GraphManager`, search filtering in `Searcher`) without modifying the module's primary exported class interface or requiring subclassing for each variation.

#### 2.5. C++20 Considerations:

- **Modules:** Foundational for the entire custom codebase (except `diskann_extension.cpp`). `diskann` modules `import` other `diskann` modules. `duckdb` modules `#include` DuckDB headers and `import diskann` modules. This structure enhances encapsulation, clarifies dependencies, and can improve build performance for the modularized parts.
- **`std::span`\**:\**** Leveraged for non-owning, often ***\*Immutable\****, views of contiguous data within module implementations.
- ***\*Concepts:\**** Employed to define compile-time contracts for templated code, particularly for generic functions exported from modules like `diskann.distance`, improving type safety and compiler diagnostics.
- **`const` \**correctness:\**** Rigorously applied to enforce immutability for data structures and parameters where appropriate, simplifying state management within and across modules.
- Other C++20 features (Ranges, `std::optional`, `std::expected`, Smart Pointers) are utilized as applicable within module implementations to enhance code safety, expressiveness, and robustness.

#### 2.6. Component Dependency Overview

The dependency flow is critical: `diskann_extension.cpp` (traditional) -> `duckdb` modules (bridge) -> `diskann` modules (core/store). `(i)` denotes include, `(m)` denotes module import.

- **`diskann_extension.cpp`**
  - `(i)->` `duckdb.hpp`
  - `(m)->` `duckdb.DiskannIndex` (module exported from `DiskannIndex.cppm`)
- **`duckdb.DiskannIndex`** (module from `DiskannIndex.cppm`)
  - `(i)->` DuckDB Core Library Headers (for Index API, types, context, etc.)
  - `(m)->` `diskann.Orchestrator`
  - `(m)->` `diskann.IndexConfig`
  - `(m)->` `duckdb.DiskannShadowStorageService` (module providing concrete `IShadowStorageService`)
  - Injects `DiskannShadowStorageService` instance (as `std::unique_ptr<diskann::store::IShadowStorageService>`) into `diskann.Orchestrator`.
- **`duckdb.DiskannShadowStorageService`** (module from `DiskannShadowStorageService.cppm`, implements `diskann::store::IShadowStorageService`)
  - `(i)->` DuckDB Core Library Headers (for SQL execution, `ClientContext`, transactions, etc.)
  - `(m)->` `diskann.store.IShadowStorageService` (to import the interface it implements)
- **`diskann.Orchestrator`** (module)
  - `(m)->` `diskann.IStorageManager` (or the module exporting this interface)
  - `(m)->` `diskann.IGraphManager` (or the module exporting this interface)
  - `(m)->` `diskann.ISearcher` (or the module exporting this interface)
  - `(m)->` `diskann.IndexConfig`
  - `(m)->` `diskann.types`
  - `(m)->` `diskann.store.IShadowStorageService` (or the module exporting this interface, dependency injected)
- **`diskann.StorageManager`** (module, class `StorageManager` implementing `diskann::IStorageManager`)
  - `(m)->` `diskann.store.IShadowStorageService` (injected)
  - `(m)->` `diskann.store.IFileSystem` (optional, injected if used)
  - `(m)->` `diskann.types`, `diskann.IndexConfig`, `diskann.utils`
  - `(m)->` `diskann.ternary_quantization` (optional internal use)
- Other `diskann` core/common/store modules: Dependencies remain internal to `diskann` (via `import`) or on standard library (via `#include` within module units). ***\*No\** `diskann` \**module depends on any\** `duckdb` \**namespace module.\****

### 3. Suggested Directory Structure (with Module Files)

The directory structure is organized to reflect the modular design and namespace conventions:

```
src/
├── diskann/
│   ├── core/                   // Core logic modules (namespace diskann)
│   │   ├── Orchestrator.cppm
│   │   ├── IndexConfig.cppm
│   │   ├── distance.cppm
│   │   ├── IStorageManager.cppm
│   │   ├── StorageManager.cppm
│   │   ├── IGraphManager.cppm
│   │   ├── GraphManager.cppm
│   │   ├── ISearcher.cppm
│   │   ├── Searcher.cppm
│   │   ├── ternary_quantization.cppm
│   │
│   ├── store/                  // Platform abstraction interface modules (namespace diskann::store)
│   │   ├── IShadowStorageService.cppm
│   │   ├── IFileSystem.cppm    // Optional
│   │
│   ├── duckdb/                 // DuckDB integration modules (namespace duckdb)
│   │   ├── DiskannIndex.cppm
│   │   ├── DiskannShadowStorageService.cppm
│   │   ├── DiskannScanState.cppm
│   │   ├── DiskannBindData.cppm
│   │   └── ...                 // Other duckdb namespace modules
│   │
│   └── common/                 // Common utility and type modules (namespace diskann)
│       ├── types.cppm
│       ├── utils.cppm
│       ├── constants.cppm
│
├── diskann_extension.cpp       // Main extension loading file (traditional C++)
└── CMakeLists.txt              // Main CMake file
```

### 4. Architectural Advantages

This "modules-first" architecture, with strict unidirectional dependencies and platform abstraction, offers significant advantages:

- **Enhanced Testability:** All `diskann` modules (core, common, store interfaces) are testable in complete isolation from DuckDB by mocking their imported dependencies (e.g., `diskann::store::IShadowStorageService`). `duckdb` namespace modules are testable by mocking the `diskann` module interfaces they `import`.
- **Clear Separation of Concerns:** The `diskann` core logic is fully decoupled from DuckDB specifics. The `duckdb` modules form a well-defined, modular bridge. The `diskann::store` modules clearly define the boundary for platform-specific services.
- **Improved Maintainability & Readability:** Explicit `export` and `import` directives clarify APIs and dependencies throughout the custom codebase. Changes to DuckDB APIs primarily impact the `diskann/duckdb/` modules.
- **Strong Encapsulation & Modularity:** C++20 modules provide superior encapsulation. Implementation details not `export`ed are truly hidden, making components more robust and self-contained.
- **Potential Build Performance Gains:** For the modular parts of the codebase, precompiled Binary Module Interfaces (BMIs) can reduce overall compilation time compared to repeated header parsing.
- ***\*Reusability of Core Logic:\**** The `diskann` core and store interface modules, having no dependencies on DuckDB, are theoretically reusable in other contexts by providing new implementations for the `diskann::store` interfaces.

### 5. Salient Design Principles and Best Practices (in a Modules-First Context)

These principles guide the implementation of the modular architecture:

- **Interfaces for Services and Platform Dependencies:** Abstract interfaces (e.g., `diskann::IStorageManager`, `diskann::store::IShadowStorageService`) are defined and `export`ed from their respective `diskann` modules. These form the contracts for service implementations.
- ***\*Dependency Injection (Constructor Injection):\**** The `duckdb.DiskannIndex` module is responsible for instantiating concrete service implementations (like `duckdb.DiskannShadowStorageService`) and injecting them (as `std::unique_ptr` to the interface type) into the `diskann.Orchestrator` module via its constructor. This pattern is used for all significant service dependencies.
- **No Reverse Dependencies:** `diskann` (core/common/store) modules **must not** `import` or `#include` from `diskann/duckdb/` modules or depend directly on DuckDB-specific types or APIs. All such interactions are mediated through the `diskann::store` interfaces.
- ***\*Explicit Ownership:\**** Smart pointers (`std::unique_ptr` for exclusive ownership, `std::shared_ptr` where shared ownership is explicitly required) are used to manage the lifetime of dynamically allocated objects and injected dependencies.
- ***\*Error Handling:\**** Exceptions are used for unrecoverable errors. `std::optional` or `std::expected` (C++23 or library equivalent) are preferred for recoverable errors or optional return values to make error paths explicit. Custom error types may be exported from a common `diskann` module if needed.
- ***\*Concurrency Management:\**** Stateful modules, particularly those involved in I/O (`diskann.StorageManager`) or shared graph structures (`diskann.GraphManager`), must be designed for thread safety if concurrent access is anticipated. Concurrency control mechanisms are encapsulated within these modules or coordinated by `diskann.Orchestrator`.
- ***\*Lifecycle Management:\**** `duckdb.DiskannIndex` manages the lifecycle of the `diskann.Orchestrator` instance. The `Orchestrator` module, in turn, manages the lifecycle of the core service components it instantiates or receives as dependencies.

### 6. Further Organizational and Planning Considerations

Successful implementation of this modules-first architecture requires careful attention to the following:

1. **Comprehensive Testing Strategy:**
   - ***\*Unit Tests:\**** Each module should have thorough unit tests. For modules exporting interfaces, mock implementations of these interfaces will be created (potentially as test-only modules or local classes) to test dependent modules in isolation. Stateless, pure functions exported from modules will be tested with a wide range of inputs.
   - ***\*Integration Tests (Module Level):\**** Test interactions between collaborating `diskann` modules and between `duckdb` and `diskann` modules, still using mocks for external systems like DuckDB itself where appropriate.
   - ***\*DuckDB Extension Integration Tests:\**** SQL-based tests (`sqllogictest`) remain essential for validating the end-to-end functionality of the loaded extension within DuckDB.
   - ***\*Performance Benchmarks & Stress Tests:\**** Critical for evaluating indexing time, search latency, recall, throughput, and the durability of the shadow storage mechanism.
2. **Configuration Management:** The `diskann.IndexConfig` module exports a DuckDB-agnostic configuration structure. The `duckdb.DiskannIndex` module translates DuckDB-specific `WITH` clause options into this structure.
3. **Shadow Implementation Integration:** The `diskann.StorageManager` module will exclusively use the injected `diskann::store::IShadowStorageService` interface for all transactional operations related to `diskann_store.duckdb` (e.g., managing shadow blocks, lookup tables, metadata).
4. **Build System (CMake):** This is a paramount consideration. The CMake configuration must:
   - Compile all `.cppm` files as C++20 module interface units (or implementation units where appropriate).
   - Correctly manage inter-module dependencies (e.g., `diskann.Orchestrator` `import`s `diskann.IStorageManager`) to ensure Binary Module Interfaces (BMIs) are built in the correct order and are available to importers.
   - Handle the linkage of the final extension, incorporating the traditionally compiled `diskann_extension.cpp` with the compiled modules.
   - This necessitates a modern CMake version (e.g., 3.28+ recommended for best support) and a compiler toolchain (GCC, Clang, MSVC) with robust and compatible C++20 module support. Thorough testing of the build process across target platforms will be essential.
5. **On-Disk Format Versioning:** Versioning applies to `graph.lmd` (managed by `diskann.StorageManager`) and the schema/data within `diskann_store.duckdb` (managed via `diskann.store.IShadowStorageService` and its `duckdb` namespace implementation). Mechanisms for checking and migrating formats must be considered.
6. **Logging and Observability:** Generic logging utilities can be exported by `diskann.utils`. The `diskann::store::IShadowStorageService` or other platform interfaces might include methods for the core to request logging of platform-specific actions, allowing the core to remain agnostic to the actual logging mechanism.

This revised specification details a comprehensive modules-first architecture, aiming for a highly decoupled, testable, and maintainable DiskANN extension for DuckDB.