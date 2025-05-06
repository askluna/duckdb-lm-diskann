#define DUCKDB_EXTENSION_MAIN

#include "lm_diskann_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension_util.hpp"
#include "duckdb/parser/parsed_data/create_scalar_function_info.hpp"

#include "lm_diskann/LmDiskannIndex.hpp"

namespace duckdb {

static void LoadInternal(DatabaseInstance &instance) {
  // Register the HNSW index module
  // LmDiskannIndex::Register(instance);
}

void LmDiskannExtension::Load(DuckDB &db) { LoadInternal(*db.instance); }

std::string LmDiskannExtension::Name() { return "lm_diskann"; }

} // namespace duckdb

extern "C" {

DUCKDB_EXTENSION_API void lm_diskann_init(duckdb::DatabaseInstance &db) {
  duckdb::DuckDB db_wrapper(db);
  db_wrapper.LoadExtension<duckdb::LmDiskannExtension>();
}

DUCKDB_EXTENSION_API const char *lm_diskann_version() {
  return duckdb::DuckDB::LibraryVersion();
}
}

#ifndef DUCKDB_EXTENSION_MAIN
#error DUCKDB_EXTENSION_MAIN not defined
#endif
