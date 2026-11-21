"""Offline eval: exact hit@k, MRR, refusal accuracy.

Reads eval/questions.yaml, calls retrieve() for each question, and
computes:
  - hit@3   : % of answerable questions where >=1 gold chunk_id is in top-3
  - MRR     : mean reciprocal rank of the first gold chunk (1/rank)
  - refusal : % of refusal-flagged questions returning no chunks
              (correct "Not in notes" short-circuit, no LLM call)

When expected_chunks are missing for a question, falls back to the
keyword-overlap proxy (>=2 keywords in retrieved text). Both are
reported; the exact metric is primary when gold ids exist.

Usage:
  uv run python eval/eval_hitrate.py [--top-k 3 --threshold 0.35]
  (Retrieval-only; never calls an LLM, zero token cost.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `retrieve` importable when run as `python eval/eval_hitrate.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieve.retrieval import retrieve


def load_questions(path: Path) -> list[dict]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["questions"]


def keywords_hit(keywords: list[str], retrieved_text: str) -> int:
    """Count how many expected keywords appear in retrieved chunk text."""
    text = retrieved_text.lower()
    return sum(1 for kw in keywords if kw.lower() in text)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval eval")
    parser.add_argument("--questions", default="eval/questions.yaml")
    parser.add_argument("--top-k", type=int, default=3)
    # Calibrated default (REPORT §19 sweep): 0.70 = refusal 1.0 + hit@3 1.0.
    # Matches .env / infra/main.tf / retrieval.py — keep all in sync.
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))

    answerable = [q for q in questions if not q.get("refusal")]
    refusals = [q for q in questions if q.get("refusal")]

    hit_at_k = 0
    mrr_sum = 0.0
    mrr_count = 0
    refusal_correct = 0
    per_question: list[dict] = []
    failures: list[dict] = []

    for q in questions:
        chunks = retrieve(q["question"], top_k=args.top_k, threshold=args.threshold)
        retrieved_ids = [c.get("chunk_id") for c in chunks]

        if q.get("refusal"):
            ok = not chunks  # Correct refusal = no chunks cleared threshold
            if ok:
                refusal_correct += 1
            else:
                failures.append({"id": q["id"], "reason": "refusal_failed_returned_chunks", "chunks": len(chunks)})
            per_question.append({"id": q["id"], "type": "refusal", "pass": ok, "retrieved": retrieved_ids})
            continue

        gold = set(q.get("expected_chunks", []))
        if gold:
            # Exact hit@3 + MRR using gold chunk_ids
            rank = next((i + 1 for i, cid in enumerate(retrieved_ids) if cid in gold), None)
            hit = rank is not None
            rr = 1.0 / rank if hit else 0.0
            if hit:
                hit_at_k += 1
            mrr_sum += rr
            mrr_count += 1
            if not hit:
                failures.append({"id": q["id"], "reason": "gold_chunk_not_in_top_k",
                                   "retrieved": retrieved_ids, "gold": sorted(gold)})
            per_question.append({"id": q["id"], "type": "exact", "rank": rank, "rr": round(rr, 3),
                                   "retrieved": retrieved_ids, "gold": sorted(gold)})
        else:
            # Fallback keyword proxy when gold ids not yet filled
            retrieved_text = "\n".join(c.get("text", "") for c in chunks)
            hit_kw = keywords_hit(q.get("expected_keywords", []), retrieved_text)
            hit = hit_kw >= 2
            if hit:
                hit_at_k += 1
            mrr_count += 1
            mrr_sum += 1.0 if hit else 0.0  # proxy MRR: 1 if pass, 0 if fail
            if not hit:
                failures.append({"id": q["id"], "reason": f"weak_keyword_overlap {hit_kw}/2", "chunks": len(chunks)})
            per_question.append({"id": q["id"], "type": "proxy", "kw_hit": hit_kw, "retrieved": retrieved_ids})

    n_ans = max(len(answerable), 1)
    metrics = {
        "hit_at_k": round(hit_at_k / n_ans, 3),
        "mrr": round(mrr_sum / max(mrr_count, 1), 3),
        "refusal_accuracy": round(refusal_correct / max(len(refusals), 1), 3),
        "answerable_passed": f"{hit_at_k}/{len(answerable)}",
        "refusals_passed": f"{refusal_correct}/{len(refusals)}",
        "per_question": per_question,
    }

    print(f"Hit@{args.top_k}:      {metrics['hit_at_k']}  ({metrics['answerable_passed']} answerable passed)")
    print(f"MRR:           {metrics['mrr']}")
    print(f"Refusal acc:   {metrics['refusal_accuracy']}  ({metrics['refusals_passed']} correctly short-circuited)")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  {f['id']}: {f['reason']} (retrieved {f.get('chunks', len(f.get('retrieved', [])))} chunks)")

    Path("eval/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path("eval/failed_cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print("\nWrote eval/metrics.json + eval/failed_cases.json")


if __name__ == "__main__":
    main()
