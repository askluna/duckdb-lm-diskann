import duckdb
import os
import pytest

# Get a fresh connection to DuckDB with the lm_diskann extension binary loaded
@pytest.fixture
def duckdb_conn():
    extension_binary = os.getenv('LM_DISKANN_EXTENSION_BINARY_PATH')
    if (extension_binary == ''):
        raise Exception('Please make sure the `LM_DISKANN_EXTENSION_BINARY_PATH` is set to run the python tests')
    conn = duckdb.connect('', config={'allow_unsigned_extensions': 'true'})
    conn.execute(f"load '{extension_binary}'")
    return conn

def test_lm_diskann(duckdb_conn):
    duckdb_conn.execute("SELECT lm_diskann('Sam') as value;");
    res = duckdb_conn.fetchall()
    assert res[0][0] == "Lm_diskann Sam 🐥"

def test_lm_diskann_openssl_version_test(duckdb_conn):
    duckdb_conn.execute("SELECT lm_diskann_openssl_version('Michael');");
    res = duckdb_conn.fetchall()
    assert res[0][0][0:51] == "Lm_diskann Michael, my linked OpenSSL version is OpenSSL"
