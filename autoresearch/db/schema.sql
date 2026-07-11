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
