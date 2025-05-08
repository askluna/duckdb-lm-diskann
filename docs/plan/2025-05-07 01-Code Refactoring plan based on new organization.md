```
# DiskANN DuckDB Extension: V1 Refactoring Plan (HPP/CPP)

## 1. Introduction

This document outlines the plan to refactor the `LmDiskannIndex` component of the DiskANN DuckDB extension. The primary goals for this V1 refactoring are:

* **Decompose `LmDiskannIndex`**: Break down its monolithic structure into smaller, more focused classes.
* **Traditional C++ Structure**: Implement these changes using `.hpp` header files and `.cpp` source files, deferring the C++20 modules approach.
* **Prevent Circular Imports**: Establish clear dependency flows to avoid circular dependencies between components.
* **Separation of Concerns**: Clearly distinguish between DuckDB integration logic and core DiskANN algorithm logic.

This plan adapts the provided "C++ Architectural Specification (Version 1.8)" for a header/source file environment.

## 2. Core Principle: Separation and Dependency Flow

The fundamental principle is to separate components into two main namespaces:

* `duckdb`: For classes directly interfacing with DuckDB, including implementations of DuckDB's `BoundIndex`. These can include DuckDB headers.
* `diskann`: For core DiskANN logic, data structures, and algorithms.
    * `diskann::core`: Core algorithmic components.
    * `diskann::store`: Interfaces for platform/storage services and their DuckDB-specific implementations (though implementations will be in the `duckdb` namespace or a sub-namespace like `diskann::duckdb_impl`).
    * `diskann::common`: Shared utilities, configuration, and basic types for the DiskANN core.

**Strict Unidirectional Dependency:**
`diskann` (core, common, store interfaces) components **must not** `#include` or depend on `duckdb` namespace components or any DuckDB-specific headers (e.g., `duckdb.hpp`).
The dependency flow will be:
`diskann_extension.cpp` -> `duckdb` namespace classes -> `diskann` namespace classes (often via interfaces).

## 3. Proposed Class Structure (HPP/CPP)

### 3.1. `duckdb` Namespace (DuckDB Integration Layer)

#### a. `duckdb::LmDiskannIndex`
* **Files**: `duckdb/LmDiskannIndex.hpp`, `duckdb/LmDiskannIndex.cpp` (Refactors existing)
* **Responsibility**:
    * Implements DuckDB's `BoundIndex` interface.
    * Handles interaction with DuckDB systems (catalog, storage, query lifecycle).
    * Parses `WITH` clause options and translates them into `diskann::core::IndexConfig`.
    * Instantiates and owns the `diskann::core::Coordinator`.
    * Instantiates and injects DuckDB-specific service implementations (e.g., `diskann::duckdb_impl::DiskannShadowStorageService`) into the `Coordinator`.
    * Translates DuckDB data structures and requests into calls on the `Coordinator`.
* **Key Members (State from current `LmDiskannIndex` that stays):**
    * `LmDiskannDBState db_state_`
    * `string index_data_path_` (or managed via Coordinator/StorageManager)
    * `unique_ptr<diskann::core::Coordinator> coordinator_`
* **Key Methods (Wrappers around Coordinator calls):**
    * Constructor (parses options, inits Coordinator)
    * `Append`, `Insert`, `Delete`
    * `InitializeScan`, `Scan`
    * `CommitDrop`, `GetStorageInfo`, `GetInMemorySize`, `Vacuum`, `VerifyAndToString`

#### b. `diskann::duckdb_impl::DiskannShadowStorageService`
* **Files**: `diskann/duckdb_impl/DiskannShadowStorageService.hpp`, `diskann/duckdb_impl/DiskannShadowStorageService.cpp`
* **Responsibility**: Implements the `diskann::store::IShadowStorageService` interface using DuckDB's functionalities (e.g., interacting with shadow tables in `diskann_store.duckdb`).
* **Includes**: DuckDB headers, `diskann/store/IShadowStorageService.hpp`.

#### c. Other DuckDB-facing Structs/Classes
* `duckdb::LmDiskannScanState` (already in `duckdb/LmDiskannScanState.hpp`): Remains largely as is.
* `duckdb::LmDiskannBindData`, `duckdb::LmDiskannCreateIndexInfo` (if needed, similar to spec's `DiskannBindData.cppm`, etc.): For holding parsed info during DDL/binding.

### 3.2. `diskann::store` Namespace (Abstract Storage/Platform Interfaces)

#### a. `diskann::store::IShadowStorageService`
* **File**: `diskann/store/IShadowStorageService.hpp`
* **Responsibility**: Defines the abstract interface for transactional operations related to the shadow store (e.g., managing shadow blocks, lookup tables, metadata within `diskann_store.duckdb`). This allows the core DiskANN logic to be unaware of DuckDB specifics for these operations.
* **Type**: Abstract class with pure virtual methods.

#### b. `diskann::store::IFileSystem` (Optional, as per spec)
* **File**: `diskann/store/IFileSystem.hpp`
* **Responsibility**: Defines an abstract interface for raw file system operations if direct file I/O for `graph.lmd` needs to be abstracted beyond standard C++ I/O or specific DuckDB file system interactions.

### 3.3. `diskann::core` Namespace (Core DiskANN Logic)

#### a. `diskann::core::Coordinator`
* **Files**: `diskann/core/Coordinator.hpp`, `diskann/core/Coordinator.cpp` (Refactors existing skeleton)
* **Responsibility**:
    * Central component for core ANN logic, replacing much of `LmDiskannIndex`'s current internal logic.
    * Owns and manages the state of the DiskANN graph (entry points, metadata, configuration, on-disk paths via `IStorageManager`).
    * Implements high-level indexing operations (build, search, insert, update, delete), coordinating other core components.
    * Owns instances of `IGraphManager`, `IStorageManager`, `ISearcher`.
    * Receives injected `IShadowStorageService` for shadow store operations.
* **Key Members**:
    * `IndexConfig config_`
    * `NodeLayoutOffsets node_layout_` (derived from config)
    * `idx_t block_size_bytes_` (derived from config)
    * `unique_ptr<IGraphManager> graph_manager_`
    * `unique_ptr<IStorageManager> storage_manager_`
    * `unique_ptr<ISearcher> searcher_`
    * `unique_ptr<store::IShadowStorageService> shadow_storage_service_` (injected)
    * `IndexPointer graph_entry_point_ptr_` (managed by GraphManager, cached here if needed)
    * `row_t graph_entry_point_rowid_` (managed by GraphManager)
    * `IndexPointer delete_queue_head_ptr_` (managed by StorageManager)
    * `bool is_dirty_`
* **Key Methods**:
    * Constructor (receives injected services and config)
    * `InitializeNewIndex(estimated_cardinality)`
    * `LoadFromStorage(storage_info)` (uses `IStorageManager`)
    * `PersistIndex()` (uses `IStorageManager`)
    * `Build(data_iterator)`
    * `Search(query_vector, k, result_row_ids)`
    * `Insert(vector, row_id)`
    * `Delete(row_id)`
    * `HandleCommitDrop()`
    * `GetIndexStorageInfo()`
    * `GetInMemorySize()`
    * `PerformVacuum()`

#### b. `diskann::core::IGraphManager` (Interface)
* **File**: `diskann/core/IGraphManager.hpp`
* **Responsibility**: Defines the contract for managing the graph structure.
    * Node allocation/deallocation (block level, RowID mapping).
    * Accessing node data (raw vectors, neighbor lists).
    * Modifying graph connectivity (adding/removing edges).
    * Managing the graph entry point(s).
    * Robust prune logic.
* **Type**: Abstract class with pure virtual methods.

#### c. `diskann::core::GraphManager` (Concrete Implementation)
* **Files**: `diskann/core/GraphManager.hpp`, `diskann/core/GraphManager.cpp` (Refactors existing `GraphManager` and parts of `GraphOperations`)
* **Responsibility**: Implements `IGraphManager`.
    * Uses `FixedSizeAllocator` (obtained from `IStorageManager` or `LmDiskannIndex`'s `BufferManager` passed down).
    * Manages `rowid_to_node_ptr_map_`.
    * Contains logic from current `GraphOperations::RobustPrune`, `InsertNode` (graph modification parts), `SelectEntryPointForSearch`.
* **Dependencies**: `IndexConfig`, `NodeLayoutOffsets`, `ISearcher` (for finding candidates during insertion), `IDistanceFunctions`.

#### d. `diskann::core::IStorageManager` (Interface)
* **File**: `diskann/core/IStorageManager.hpp`
* **Responsibility**: Defines the contract for managing persistence of the index.
    * Loading/saving the main graph file (`graph.lmd`).
    * Loading/saving index metadata.
    * Managing the `FixedSizeAllocator` for node blocks.
    * Interacting with `IShadowStorageService` for transactional parts of storage.
    * Managing the delete queue.
* **Type**: Abstract class with pure virtual methods.

#### e.
`diskann::core::StorageManager` (Concrete Implementation)
* **Files**: `diskann/core/StorageManager.hpp`, `diskann/core/StorageManager.cpp` (Refactors existing `StorageManager.hpp` free functions into a class)
* **Responsibility**: Implements `IStorageManager`.
    * Handles file I/O for `graph.lmd`.
    * Serializes/deserializes metadata (`LmDiskannMetadata`).
    * Owns/Manages the `FixedSizeAllocator` instance.
    * Uses injected `IShadowStorageService` for operations on `diskann_store.duckdb`.
    * Implements `EnqueueDeletion`, `ProcessDeletionQueue` (from current `StorageManager.hpp` and `LmDiskannIndex`).
* **Dependencies**: `IndexConfig`, `NodeLayoutOffsets`, `IShadowStorageService`, DuckDB `BufferManager` (passed for allocator).

#### f. `diskann::core::ISearcher` (Interface)
* **File**: `diskann/core/ISearcher.hpp`
* **Responsibility**: Defines the contract for performing ANN searches on the graph.
* **Type**: Abstract class with pure virtual methods.

#### g. `diskann::core::Searcher` (Concrete Implementation)
* **Files**: `diskann/core/Searcher.hpp`, `diskann/core/Searcher.cpp` (Refactors existing `Searcher.hpp` and `LmDiskannIndex::PerformSearch`)
* **Responsibility**: Implements `ISearcher`.
    * Contains the beam search logic (`PerformSearch`).
    * Uses `IGraphManager` to access node data and neighbors.
    * Uses `IDistanceFunctions` for comparisons.
* **Dependencies**: `IndexConfig`, `NodeLayoutOffsets`, `IGraphManager`, `IDistanceFunctions`.

### 3.4. `diskann::common` Namespace (Shared Utilities)

#### a. `diskann::common::IndexConfig`
* **Files**: `diskann/common/IndexConfig.hpp`, `diskann/common/IndexConfig.cpp` (Refactors existing `index_config.hpp/.cpp`)
* **Responsibility**: Holds configuration parameters (parsed from `WITH` options and derived from column types). Includes `LmDiskannConfig`, `NodeLayoutOffsets`, enums, constants, validation, and layout calculation logic.
* **Note**: The spec places `IndexConfig.cppm` in `diskann.core`. Moving to `common` might be better if it's truly shared widely without core logic. For V1, keeping it in `core` as `diskann::core::IndexConfig` is also fine if it's primarily used by core components. Let's assume `diskann::core::IndexConfig` for now, consistent with user's current structure.

#### b. `diskann::common::Distance` (or `diskann::core::Distance` and an `IDistanceFunctions` interface)
* **Files**: `diskann/common/distance.hpp`, `diskann/common/distance.cpp` (Refactors existing `distance.hpp/.cpp`)
* **Responsibility**: Provides distance calculation functions (exact, approximate), vector compression/conversion for edges.
* **Note**: The public helper functions in `LmDiskannIndex` (`PublicCalculateApproxDistance`, etc.) will likely call these. The spec puts `distance.cppm` in `diskann.core`.

#### c. `diskann::common::Types`
* **File**: `diskann/common/types.hpp`
* **Responsibility**: Basic type definitions used across the `diskann` core, if any are needed beyond standard types and DuckDB's `idx_t`, `row_t`. (e.g., `candidate_pair_t`).

#### d. `diskann::common::Utils`
* **Files**: `diskann/common/utils.hpp`, `diskann/common/utils.cpp`
* **Responsibility**: General utility functions, logging helpers (if any).

## 4. Breakdown of `LmDiskannIndex`

The current `LmDiskannIndex` class will be significantly slimmed down.

### Functionality Staying in `duckdb::LmDiskannIndex`:
* Implementation of `BoundIndex` virtual methods (as top-level entry points).
* Constructor:
    * Parsing `WITH` options into `diskann::core::IndexConfig`.
    * Deriving `dimensions`, `node_vector_type` from DuckDB's `LogicalType`.
    * Validating the initial `IndexConfig`.
    * Calculating `NodeLayoutOffsets` and `block_size_bytes_`.
    * Creating the `diskann::duckdb_impl::DiskannShadowStorageService`.
    * Creating and initializing the `diskann::core::Coordinator`, injecting dependencies.
    * Handling `storage_info` to decide between `Coordinator::InitializeNewIndex` or `Coordinator::LoadFromStorage`.
    * Managing `index_data_path_` creation.
* `LmDiskannDBState db_state_` member.
* `InitializeScan`: Translates DuckDB query vector into a format suitable for `Coordinator::Search` and sets up `LmDiskannScanState`.
* `Scan`: Calls `Coordinator::Search` and populates the DuckDB `result` vector.
* Methods like `GetConstraintViolationMessage` (if simple).

### Functionality Moving to `diskann::core::Coordinator`:
* Ownership of `IndexConfig` (after initial setup by `LmDiskannIndex`).
* Ownership and lifecycle management of `IGraphManager`, `IStorageManager`, `ISearcher`.
* High-level logic for `Append`, `Insert`, `Delete` (coordinating the managers).
* Logic for `InitializeNewIndex`, `LoadFromStorage`, `PersistIndex`.
* Management of `is_dirty_` flag.
* Vacuum logic (coordinating `IStorageManager`).
* `GetInMemorySize`, `GetStorageInfo` (delegating to managers).
* The core state like `graph_entry_point_ptr_`, `delete_queue_head_ptr_` will be managed by `Coordinator` or its delegates (`GraphManager`, `StorageManager`).

### Interaction Example: `Insert`
1.  `duckdb::LmDiskannIndex::Insert(lock, data_chunk, row_ids)` is called by DuckDB.
2.  It extracts the vector and `row_id` from `data_chunk`.
3.  It calls `coordinator_->Insert(vector_ptr, row_id)`.
4.  `diskann::core::Coordinator::Insert` then:
    * Uses `ISearcher` (or `IGraphManager` with search capabilities) to find candidate neighbors for the new vector.
    * Uses `IGraphManager` to:
        * Allocate a new node for `row_id`.
        * Store the vector data.
        * Perform robust pruning to select final neighbors.
        * Update edges for the new node and its neighbors.
        * Update the graph entry point if necessary.
    * Uses `IStorageManager` (which uses `IShadowStorageService`) to ensure the operation is durable if required by shadow store semantics.
    * Sets `is_dirty_ = true`.

## 5. Dependency Management and Circular Imports

* **Interfaces are Key**: `IShadowStorageService`, `IStorageManager`, `IGraphManager`, `ISearcher` break dependencies. Core components depend on these interfaces, not concrete DuckDB-specific implementations.
* **Forward Declarations**: Use forward declarations in `.hpp` files where a full definition is not needed (e.g., if only pointers or references to a type are used in function signatures).
* **Include Order**:
    * `diskann::common` headers should be self-contained or only include other `diskann::common` headers / standard library.
    * `diskann::store` interface headers should be self-contained.
    * `diskann::core` interface headers should be self-contained.
    * `diskann::core` concrete class headers include their own interfaces, other `diskann::core` or `diskann::common` headers, and `diskann::store` interfaces. **No DuckDB headers.**
    * `diskann::duckdb_impl` headers include their `diskann::store` interface and DuckDB headers.
    * `duckdb` namespace headers (like `LmDiskannIndex.hpp`) include DuckDB headers and relevant `diskann` headers (interfaces or concrete classes).

## 6. Directory Structure

The suggested directory structure from the specification is suitable for `.hpp`/`.cpp` files as well:
```

src/

├── diskann/

│   ├── core/                   // Core logic (Coordinator, GraphManager, StorageManager, Searcher, interfaces IGraphManager etc.)

│   │   ├── Coordinator.hpp

│   │   ├── Coordinator.cpp

│   │   ├── IGraphManager.hpp

│   │   ├── GraphManager.hpp

│   │   ├── GraphManager.cpp

│   │   ├── ... (other core components and interfaces)

│   │

│   ├── store/                  // Platform abstraction interfaces

│   │   ├── IShadowStorageService.hpp

│   │   ├── IFileSystem.hpp     // Optional

│   │

│   ├── duckdb_impl/            // DuckDB-specific implementations of store interfaces

│   │   ├── DiskannShadowStorageService.hpp

│   │   ├── DiskannShadowStorageService.cpp

│   │

│   └── common/                 // Common utilities, types, config, distance

│       ├── IndexConfig.hpp     // (Or move to core/ if preferred)

│       ├── IndexConfig.cpp

│       ├── distance.hpp

│       ├── distance.cpp

│       ├── types.hpp           // (If needed)

│       ├── utils.hpp           // (If needed)

│

├── duckdb/                     // DuckDB integration layer (LmDiskannIndex, LmDiskannScanState)

│   ├── LmDiskannIndex.hpp

│   ├── LmDiskannIndex.cpp

│   ├── LmDiskannScanState.hpp

│   ├── LmDiskannScanState.cpp  // (If it has non-inline methods)

│   └── ...

│

├── diskann_extension.cpp       // Main extension loading file

└── CMakeLists.txt

```
*Note: `IndexConfig` and `distance` are placed in `common` here, but keeping them in `core` as per your current file structure (`core/index_config.hpp`, `core/distance.hpp`) is also acceptable if their primary use is within the core logic.*

## 7. Key Changes to Existing Files (Summary)

* **`LmDiskannIndex.hpp/.cpp`**: Major refactoring. Becomes thinner, delegating most logic to `Coordinator`.
* **`Coordinator.hpp/.cpp`**: Skeleton filled out to become the central core logic coordinator.
* **`GraphManager.hpp/.cpp`**: Enhanced to implement `IGraphManager`. Takes on more responsibilities from current `LmDiskannIndex` and `GraphOperations` related to graph structure, node data, and entry points. `NodeAccessors` likely remains a helper class used by `GraphManager`.
* **`GraphOperations.hpp/.cpp`**: Much of its logic (like `RobustPrune`, `InsertNode`'s graph manipulation parts, entry point selection) will be merged into `GraphManager` or `Coordinator`. This class might be deprecated or significantly reduced.
* **`StorageManager.hpp/.cpp`**: Changed from free functions to a class implementing `IStorageManager`. Manages `FixedSizeAllocator`, metadata I/O, main graph file I/O, and delete queue.
* **`Searcher.hpp/.cpp`**: Implements `ISearcher`, contains beam search logic.
* **`index_config.hpp/.cpp`**: Remains for `LmDiskannConfig`, `NodeLayoutOffsets`, parsing, validation. May move to `diskann::common` or stay in `diskann::core`.
* **`distance.hpp/.cpp`**: Remains for distance functions. May move to `diskann::common` or stay in `diskann::core`.
* **`ternary_quantization.hpp`**: Likely used by `distance.cpp` or components dealing with compressed vectors.

## 8. Next Steps

1.  **Define Interfaces**: Start by creating the `.hpp` files for `IShadowStorageService`, `IStorageManager`, `IGraphManager`, `ISearcher`.
2.  **Implement Core Components**: Begin implementing `Coordinator`, `StorageManager`, `GraphManager`, `Searcher` based on these interfaces and the refactored logic from `LmDiskannIndex`.
3.  **Refactor `LmDiskannIndex`**: Modify it to use the `Coordinator` and other new components.
4.  **Implement `DiskannShadowStorageService`**.
5.  **Update CMakeLists.txt**: Ensure all new `.cpp` files are compiled and linked correctly.
6.  **Testing**: Thorough unit and integration testing will be crucial at each step.

This plan provides a roadmap for the V1 refactoring. The key is a disciplined approach to separating concerns and managing dependencies through interfaces.
```

This plan should guide the initial refactoring. Remember that this is an iterative process, and some details might be refined as you delve into the implementation. The focus on using interfaces from the start, even with traditional hpp/cpp files, will make a future transition to C++20 modules much smoother.