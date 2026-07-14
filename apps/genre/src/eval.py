"""Scorecard: load a run.json, compare final_metrics vs paper_target, emit verdict.
    python src/eval.py --run runs/beardown/<ts>/
"""
import argparse, json, os
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = json.load(open(os.path.join(args.run, "run.json")))
    # TODO: print ours vs paper_target, delta, and reproduction verdict
    print(run.get("final_metrics"), "vs", run.get("paper_target"))
if __name__ == "__main__":
    main()
