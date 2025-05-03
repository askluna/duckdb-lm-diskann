var duckdb = require('../../duckdb/tools/nodejs');
var assert = require('assert');

describe(`lm_diskann extension`, () => {
    let db;
    let conn;
    before((done) => {
        db = new duckdb.Database(':memory:', {"allow_unsigned_extensions":"true"});
        conn = new duckdb.Connection(db);
        conn.exec(`LOAD '${process.env.LM_DISKANN_EXTENSION_BINARY_PATH}';`, function (err) {
            if (err) throw err;
            done();
        });
    });

    it('lm_diskann function should return expected string', function (done) {
        db.all("SELECT lm_diskann('Sam') as value;", function (err, res) {
            if (err) throw err;
            assert.deepEqual(res, [{value: "Lm_diskann Sam 🐥"}]);
            done();
        });
    });

    it('lm_diskann_openssl_version function should return expected string', function (done) {
        db.all("SELECT lm_diskann_openssl_version('Michael') as value;", function (err, res) {
            if (err) throw err;
            assert(res[0].value.startsWith('Lm_diskann Michael, my linked OpenSSL version is OpenSSL'));
            done();
        });
    });
});