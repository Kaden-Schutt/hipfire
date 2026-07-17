# Agentic PR static review

This skill is **manual-only** and operates as a read-only controller: it
does not mutate GitHub. It reads a bounded capsule JSON and writes or
reports a structured proposal JSON. It uses toolless inference only; no
provider may receive tools or execute repository commands.

The controller must not run `git checkout`; test execution is out of scope.
It does not inspect arbitrary branches or invoke a shell-backed coding agent.

Run the inspector with:

```text
python3 -m autoresearch.ar.review.cli inspect --capsule FILE --proposal FILE
```

Use `preflight.sh` in `controller` mode to validate protected configuration,
read-only API access, and capsule source access before inspection. The
controller and publisher are separate: only a publisher with the required
write-permission operator credential may perform GitHub mutations.
