#!/usr/bin/env python
"""Run the split invariant tests and emit data/splits_report.json (feeds the
Data panel's "Split tests" checklist).

Drives the report off the ACTUAL tests in _shared/tests/test_splits.py — each
test function is executed, its docstring becomes the description, and pass/fail
is captured. So the panel can never drift from the tests: if a test changes, the
report changes. (This complements `pytest`, it doesn't replace it.)

    python projects/genre/src/split_tests_report.py
"""
from __future__ import annotations
import inspect, json, os, sys, tempfile, time, traceback
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from _shared.tests import test_splits as T


def _first_line(doc: str | None) -> str:
    if not doc:
        return "(no description)"
    return " ".join(doc.strip().split("\n")[0].split())


def run_one(fn):
    """Execute a test fn; supply tmp_path if it asks for one (pytest fixture)."""
    sig = inspect.signature(fn)
    kwargs = {}
    tmp = None
    if "tmp_path" in sig.parameters:
        tmp = tempfile.TemporaryDirectory()
        kwargs["tmp_path"] = Path(tmp.name)
    try:
        fn(**kwargs)
        return True, ""
    except Exception:
        return False, traceback.format_exc(limit=2).strip().splitlines()[-1]
    finally:
        if tmp:
            tmp.cleanup()


def main():
    out_path = Path(__file__).resolve().parents[1] / "data" / "splits_report.json"
    tests = [(name, fn) for name, fn in vars(T).items()
             if name.startswith("test_") and callable(fn)]
    tests.sort(key=lambda kv: kv[1].__code__.co_firstlineno)   # source order

    results = []
    for name, fn in tests:
        passed, err = run_one(fn)
        results.append({
            "name": name.replace("test_", "", 1),
            "description": _first_line(fn.__doc__),
            "passed": passed,
            "error": err if not passed else "",
        })
        flag = "ok" if passed else "FAIL"
        print(f"  [{flag}] {name}")

    n = len(results)
    n_pass = sum(r["passed"] for r in results)
    report = {
        "schema": "splits_report/1.0",
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "_shared/tests/test_splits.py",
        "n_tests": n,
        "n_passed": n_pass,
        "all_green": n_pass == n,
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"[split-tests] {n_pass}/{n} passing -> {out_path}")


if __name__ == "__main__":
    main()
