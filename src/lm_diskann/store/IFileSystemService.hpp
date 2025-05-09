#pragma once

#include "../common/duckdb_types.hpp" // For common::idx_t, common::data_ptr_t, etc.

#include <string>

namespace diskann {
namespace store {

/**
 * @brief Interface for abstracting raw file system operations for a single data file.
 *
 * This service is intended to manage I/O for a primary data file like graph.lmd,
 * abstracting away the specifics of the underlying file system implementation.
 */
class IFileSystemService {
	public:
	virtual ~IFileSystemService() = default;

	/**
	 * @brief Opens the file associated with this service.
	 * @param file_path Path to the file.
	 * @param read_only True if the file should be opened in read-only mode.
	 * @throw common::IOException on failure.
	 */
	virtual void Open(const std::string &file_path, bool read_only = false) = 0;

	/**
	 * @brief Closes the currently open file.
	 */
	virtual void Close() = 0;

	/**
	 * @brief Reads a block of data from the file at a specific offset.
	 * @param offset The offset in the file to read from.
	 * @param size The number of bytes to read.
	 * @param buffer_out Pointer to the buffer where data will be read.
	 * @throw common::IOException on failure or if read is incomplete.
	 */
	virtual void ReadBlock(common::idx_t offset, common::idx_t size, common::data_ptr_t buffer_out) = 0;

	/**
	 * @brief Writes a block of data to the file at a specific offset.
	 * @param offset The offset in the file to write to.
	 * @param size The number of bytes to write.
	 * @param buffer_in Pointer to the buffer containing data to write.
	 * @throw common::IOException on failure or if write is incomplete.
	 */
	virtual void WriteBlock(common::idx_t offset, common::idx_t size, common::const_data_ptr_t buffer_in) = 0;

	/**
	 * @brief Gets the current size of the file.
	 * @return The size of the file in bytes.
	 * @throw common::IOException on failure.
	 */
	virtual common::idx_t GetFileSize() = 0;

	/**
	 * @brief Truncates the file to a specific size.
	 * @param new_size The size to truncate the file to.
	 * @throw common::IOException on failure.
	 */
	virtual void Truncate(common::idx_t new_size) = 0;

	/**
	 * @brief Ensures that all written data is flushed to the storage device.
	 * @throw common::IOException on failure.
	 */
	virtual void Sync() = 0;

	/**
	 * @brief Checks if the file is currently open.
	 * @return True if open, false otherwise.
	 */
	virtual bool IsOpen() const = 0;
};

} // namespace store
} // namespace diskann