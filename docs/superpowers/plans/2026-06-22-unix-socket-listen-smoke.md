# Unix-Socket Serve — Integration Smoke Checklist

> **Requires a real GPU box with `hipfire serve` working.**
> Pure-TS behavior is covered by the unit tests in Tasks 1–7 (no GPU).
> Steps 1–8 mirror spec §10; 9–10 are the R2 N1/N4 regressions.
> Record pass/fail per step in the results table at the bottom.

---

## Checklist

- [ ] **Step 1: Socket bind + health**

```bash
hipfire serve --socket-path /run/user/$UID/hf-test.sock --no-prewarm &
curl --unix-socket /run/user/$UID/hf-test.sock http://localhost/health
```

Expected: `{"status":"ok","token":...}`.

---

- [ ] **Step 2: `run` and `chat` proxy over the socket**

With `cfg.socket_path = /run/user/$UID/hf-test.sock`: `hipfire run …` and `hipfire chat` connect over the socket (`isServeUp` true; chat POSTs over the socket — §6.11).

---

- [ ] **Step 3: Stale-socket recovery**

`kill -9` the serve; confirm the socket file lingers; `scripts/serve-restart.sh --socket /run/user/$UID/hf-test.sock --kill-only` removes it; a fresh `serve --socket-path …` binds (pre-bind stale-unlink, §6.9.3).

---

- [ ] **Step 4: `stop` / `restart`**

`hipfire stop` of a socket-mode serve kills it and removes the socket file; `hipfire restart` rebinds the same socket (§6.7).

---

- [ ] **Step 5: `ps`**

`hipfire ps` reports `serve unix:<path>: ACTIVE` and `--json` carries `transport: "unix"` (§6.12).

---

- [ ] **Step 6: Non-socket file guard (M6)**

```bash
touch /tmp/notasock
hipfire serve --socket-path /tmp/notasock
```

Expected: refuses to bind, does NOT delete the file. Output: `refusing to bind: /tmp/notasock exists and is not a socket`.

---

- [ ] **Step 7: `serve -d`**

`hipfire serve -d --socket-path …` → readiness poll over the socket succeeds; child re-binds the socket (§6.8).

---

- [ ] **Step 8: Mutual exclusion + relative path (M10)**

```bash
hipfire serve --socket-path /run/x.sock 0.0.0.0:11435   # → usage error
hipfire serve --socket-path foo.sock                      # → usage error (not absolute)
```

Expected: usage error in both cases.

---

- [ ] **Step 9: N1 — chat socket reuse**

`cfg.socket_path` set; `hipfire chat`, exit → socket is gone (not leaked). `hipfire chat` again → starts cleanly (no "socket in use"). `kill -9` a chat-spawned serve → next `hipfire chat` reclaims the leftover socket (§6.9.3/§6.9.6, §6.11).

---

- [ ] **Step 10: N4 — socket-mode restart/force vs foreign TCP holder**

With a foreign process holding TCP `:11435` (e.g. `nc -l 11435 &`): `hipfire restart` of a socket-mode serve, and `hipfire stop --socket <path> --force`, kill only the `--socket-path` serve + `rm -f` the socket, and leave the TCP `:11435` holder **alive** (§6.7.1).

---

## Results

| Step | Description | Command / Action | Expected | Actual | Pass/Fail |
|------|-------------|------------------|----------|--------|-----------|
| 1 | Socket bind + health | `hipfire serve --socket-path /run/user/$UID/hf-test.sock --no-prewarm &` then `curl --unix-socket /run/user/$UID/hf-test.sock http://localhost/health` | `{"status":"ok","token":...}` | `{"status":"ok","model":null,"pid":...,"token":"..."}` | ✅ PASS |
| 2 | `run`/`chat` proxy over socket | Set `cfg.socket_path`; run `hipfire run …` and `hipfire chat` | Both connect via socket; `isServeUp` true | `run` connects over socket (daemon crashed on model load — missing hipcc, pre-existing). Socket transport verified: request reaches the serve's HTTP handler via Unix socket. `chat` not tested (needs TUI + model). | ✅ TRANSPORT / ⏭️ INFERENCE |
| 3 | Stale-socket recovery | `kill -9` serve; confirm socket lingers; `scripts/serve-restart.sh --socket … --kill-only`; fresh `serve --socket-path …` | Socket removed by restart script; fresh serve binds cleanly | socket lingered after kill -9; serve-restart --kill-only removed it | ✅ PASS |
| 4 | `stop` / `restart` | `hipfire stop` then `hipfire restart` on a socket-mode serve | `stop` kills + removes socket file; `restart` rebinds same socket | stop: "hipfire serve stopped (PID N)", socket gone; restart: health responds | ✅ PASS |
| 5 | `ps` | `hipfire ps` and `hipfire ps --json` | `serve unix:<path>: ACTIVE`; JSON carries `transport: "unix"` | "serve unix:/run/user/1000/hf-test.sock: ACTIVE"; JSON: "transport":"unix" | ✅ PASS |
| 6 | Non-socket file guard (M6) | `touch /tmp/notasock; hipfire serve --socket-path /tmp/notasock` | Refuses to bind; file NOT deleted; prints refusal message | "refusing to bind: /tmp/notasock exists and is not a socket"; exit 1; file intact | ✅ PASS |
| 7 | `serve -d` detach | `hipfire serve -d --socket-path …` | Readiness poll over socket succeeds; child binds socket | "hipfire serve started in background (PID N, bind unix:...)", health OK, log: "listening on unix:..." | ✅ PASS |
| 8 | Mutual exclusion + relative path (M10) | `hipfire serve --socket-path /run/x.sock 0.0.0.0:11435`; `hipfire serve --socket-path foo.sock` | Usage error in both cases | "--socket-path and an explicit host/port are mutually exclusive" (exit 2); "--socket-path must be an absolute path" (exit 2) | ✅ PASS |
| 9 | N1 — chat socket reuse | `hipfire chat` exit → socket gone; second `hipfire chat` → clean start; `kill -9` spawned serve → third `hipfire chat` reclaims socket | No socket leaks; no "socket in use" errors | SKIP — requires a downloaded model for chat to spawn serve | ⏭️ SKIP |
| 10 | N4 — socket restart vs foreign TCP holder | `nc -l 11435 &`; `hipfire restart` (socket mode); `hipfire stop --socket <path> --force` | Only socket-path serve killed + socket removed; TCP `:11435` holder left alive | socket serve stopped + socket removed; Python TCP holder on :11435 still LISTENing | ✅ PASS |
