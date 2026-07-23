-- Copyright (c) Kaden Schutt
-- ar.db durable store for the autoresearch loop.
-- Rebuilt idempotently from autoresearch/ledger/*.jsonl + autoresearch/state/bod_*.json
-- by ar.db.ingest(). The ledger (git-tracked) is the source of truth; ar.db is a
-- queryable index. `measurement_hash` (sha256(gpu_arch|model|base_sha|var_sha|
-- prompt_md5|kv|maxtok)[:16]) is the row identity and the ingest idempotency key.

CREATE TABLE IF NOT EXISTS attempts(
  id               INTEGER PRIMARY KEY,
  arch             TEXT,
  kernel           TEXT,
  lever            TEXT,
  verdict          TEXT,
  tok_delta        REAL,          -- kernel_decode_tok_s delta % (conjunctive perf: UP)
  dur_delta        REAL,          -- rocprof kernel-duration delta % (conjunctive perf: DOWN)
  profile          TEXT,          -- roofline / profile-feedback blurb (the WHY)
  base_sha         TEXT,
  var_sha          TEXT,
  measurement_hash TEXT UNIQUE,   -- idempotency key; INSERT OR IGNORE dedups re-ingest
  ts               INTEGER
);

CREATE TABLE IF NOT EXISTS bod(
  arch     TEXT,
  kernel   TEXT,
  wall_pct REAL,
  l2_hit   REAL,
  mem_busy REAL,
  occ      REAL,
  vgpr     INTEGER,
  snap_ts  INTEGER
);

CREATE TABLE IF NOT EXISTS runs(
  id      TEXT PRIMARY KEY,
  arch    TEXT,
  model   TEXT,
  card    INTEGER,
  status  TEXT,
  budget  INTEGER,
  calls   INTEGER,
  ttl     INTEGER,
  pid     INTEGER,
  ts      INTEGER
);

CREATE INDEX IF NOT EXISTS ix_att ON attempts(arch, kernel, lever);

-- ── harvested fleet corpus (scripts/harvest_ledgers.py) ─────────────────────
-- `attempts` above is the LIVE loop's table, keyed on canonical measurement_hash.
-- The harvested corpus gets its own tables because ~97% of historical rows have
-- no canonical hash and must key on `fallback_key` instead; mixing the two
-- keyspaces in one UNIQUE column would either drop rows or mint look-alike
-- hashes that fail to match future loop output.

CREATE TABLE IF NOT EXISTS corpus_attempts(
  id               INTEGER PRIMARY KEY,
  corpus_key       TEXT UNIQUE,   -- COALESCE(measurement_hash, fallback_key)
  measurement_hash TEXT,          -- canonical only; NULL when uncomputable
  fallback_key     TEXT,
  key_confidence   TEXT,          -- canonical | strong | weak | collision
  arch             TEXT,
  kernel           TEXT,
  lever            TEXT,
  label            TEXT,
  variant          TEXT,
  verdict          TEXT,
  tok_delta        REAL,
  dur_delta        REAL,
  mwu_dominance    REAL,
  rounds           INTEGER,
  base_decode      REAL,
  var_decode       REAL,
  win_commit       TEXT,
  profile          TEXT,          -- the WHY
  base_sha         TEXT,
  var_sha          TEXT,
  model            TEXT,
  prompt_md5       TEXT,
  ts               INTEGER
);

CREATE TABLE IF NOT EXISTS kernel_static(
  id              INTEGER PRIMARY KEY,
  static_key      TEXT UNIQUE,
  static_key_kind TEXT,           -- isa | content
  arch            TEXT,
  kernel          TEXT,
  phase           TEXT,
  workload_kind   TEXT,
  quant           TEXT,
  shape_bucket    TEXT,
  bound_class     TEXT,
  isa_fingerprint TEXT,
  vgpr            INTEGER,
  sgpr            INTEGER,
  lds             INTEGER,
  scratch         INTEGER,
  fixture_path    TEXT
);

CREATE TABLE IF NOT EXISTS provenance(
  corpus_key  TEXT,
  table_name  TEXT,
  box         TEXT,
  path        TEXT,
  mtime       INTEGER,
  also_seen   INTEGER,            -- identical copies collapsed into this row
  boxes       TEXT                -- comma-joined boxes the row was found on
);

CREATE INDEX IF NOT EXISTS ix_corpus ON corpus_attempts(arch, kernel, lever);
CREATE INDEX IF NOT EXISTS ix_static ON kernel_static(arch, kernel);
CREATE INDEX IF NOT EXISTS ix_prov   ON provenance(corpus_key);
