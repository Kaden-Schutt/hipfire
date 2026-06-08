import json, re, sys, subprocess, tempfile, os, time, argparse, urllib.request
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base-url", default="http://localhost:11435/v1/chat/completions")
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

ds = load_dataset("openai_humaneval")["test"]
probs = list(ds)
if args.limit: probs = probs[:args.limit]

def ask(prompt):
    body = json.dumps({"model": args.model, "messages": [
        {"role": "user", "content":
         "Complete the following Python function. Return the COMPLETE function "
         "(including imports and the signature) inside a single ```python code block.\n\n"
         "```python\n" + prompt + "\n```"}],
        "max_tokens": 1024, "temperature": 0}).encode()
    req = urllib.request.Request(args.base_url, data=body, headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=600))
            return r["choices"][0]["message"].get("content") or ""
        except Exception as e:
            time.sleep(2)
    return ""

def extract(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return m.group(1) if m else text

def run_prog(program, test, entry):
    src = program + "\n\n" + test + f"\n\ncheck({entry})\n"
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, timeout=15)
        return p.returncode == 0
    except Exception:
        return False

passed = 0; n = 0; results = []
t0 = time.time()
for ex in probs:
    gen = ask(ex["prompt"])
    code = extract(gen)
    ok = run_prog(code, ex["test"], ex["entry_point"])
    passed += int(ok); n += 1
    results.append({"task_id": ex["task_id"], "pass": ok, "gen_len": len(gen)})
    if n % 20 == 0:
        print(f"  {n}/{len(probs)}  pass@1={passed/n:.3f}  ({n/(time.time()-t0):.2f} q/s)", flush=True)

out = {"model": args.model, "n": n, "pass@1": passed / n if n else 0.0, "results": results}
json.dump(out, open(args.out, "w"), indent=2)
print(f"DONE {args.model}: pass@1 = {out['pass@1']:.4f}  (n={n})")
