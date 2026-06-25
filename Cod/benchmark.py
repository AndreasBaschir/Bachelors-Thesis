#!/usr/bin/env python3
"""
Benchmark: Naive RAG vs Advanced RAG on a sample from the SQuAD training set.

Retrieval metrics (always computed):
  context_recall  - fraction of questions where a gold answer span appears in
                    a top-K chunk from the correct article
  mrr             - Mean Reciprocal Rank of the first such chunk
  avg_retrieval_ms - mean query-processing latency in milliseconds. NOTE: for the
                    Advanced pipeline this includes the multi-query expansion LLM
                    call, not just vector/BM25 retrieval.

Answer quality metrics (opt-in, requires LLM calls):
  answer_f1       - SQuAD token-F1 against reference answer spans
  exact_match     - SQuAD exact match against reference answer spans

RAGAS metrics (opt-in, implies --with-generation, very slow):
  faithfulness      - how well the answer is grounded in the retrieved context
  answer_relevancy  - how relevant the answer is to the question

Results are reported as mean +/- std across all seeds.

Usage:
  python benchmark.py
  python benchmark.py --samples 100 --num-seeds 4
  python benchmark.py --with-generation --samples 50
  python benchmark.py --with-ragas --samples 50 --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import string
import time
from collections import Counter

import datasets_registry
from advanced import pipeline as adv
from naive import pipeline as naive

DEFAULT_SAMPLES = 200
DEFAULT_SEED = 42
DEFAULT_NUM_SEEDS = 4

REFUSAL_MARKER = "does not contain enough information"


def main() -> None:
    """Parse CLI arguments and run the full benchmark pipeline."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Questions per seed (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Base random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS,
                        help=f"Number of seeds to run (default: {DEFAULT_NUM_SEEDS})")
    parser.add_argument("--with-generation", action="store_true",
                        help="Also evaluate answer quality (slow - one LLM call per question per pipeline)")
    parser.add_argument("--with-ragas", action="store_true",
                        help="Run RAGAS faithfulness and answer_relevancy (implies --with-generation)")
    parser.add_argument("--dataset", choices=datasets_registry.available(),
                        default="squad",
                        help="Which dataset to benchmark (default: squad)")
    parser.add_argument("--neg-frac", type=float, default=0.0,
                        help="Fraction of questions made UNANSWERABLE to measure hallucination resistance")
    parser.add_argument("--output", metavar="FILE",
                        help="Save full results to a JSON file")
    args = parser.parse_args()

    if args.with_ragas:
        args.with_generation = True

    datasets_registry.set_active(args.dataset)
    spec = datasets_registry.active()

    seeds = [args.seed + i * 1000 for i in range(args.num_seeds)]
    rows = spec.load_eval_rows()

    naive.retrieve("warm-up query")
    adv.retrieve("warm-up query")

    pipelines = [("naive", naive), ("advanced", adv)]
    per_seed_summaries: list[dict] = []
    all_records: list[dict] = []

    for seed_idx, seed in enumerate(seeds):
        rng = random.Random(seed)
        make_neg = spec.make_negative_record if args.with_generation else None
        # Oversample 2x so we can absorb None returns from make_record/make_neg.
        pool = rng.sample(rows, min(args.samples * 2, len(rows)))
        sample: list[dict] = []
        for row in pool:
            use_neg = make_neg is not None and args.neg_frac > 0 and rng.random() < args.neg_frac
            record = make_neg(row) if use_neg else spec.make_record(row)
            if record is not None:
                record["seed"] = seed
                sample.append(record)
            if len(sample) >= args.samples:
                break

        records: list[dict] = []
        ragas_rows: dict[str, list[dict]] = {"naive": [], "advanced": []}

        for i, record in enumerate(sample):
            question: str = record["question"]
            gold_answers: list[str] = record["gold_answers"]
            gold_title: str = record["gold_title"]

            for name, pipeline in pipelines:
                t0 = time.perf_counter()
                chunks = pipeline.retrieve(question)
                retrieval_ms = (time.perf_counter() - t0) * 1000

                rank = _context_rank(chunks, gold_answers, gold_title)

                entry: dict = {
                    "retrieval_ms": round(retrieval_ms, 2),
                    "recall": rank is not None,
                    "mrr": round(1.0 / rank, 4) if rank is not None else 0.0,
                    "rank": rank,
                }

                if args.with_generation:
                    try:
                        answer = "".join(pipeline.stream(question, chunks))
                    except Exception as exc:
                        answer = f"[ERROR: {exc}]"
                    entry["answer"] = answer
                    entry["abstained"] = REFUSAL_MARKER in answer.lower()

                    if record.get("answerable", True):
                        entry["f1"] = round(
                            max((_token_f1(answer, ref) for ref in gold_answers), default=0.0), 4
                        )
                        entry["em"] = any(_exact_match(answer, ref) for ref in gold_answers)

                        if args.with_ragas:
                            ragas_rows[name].append({
                                "question": question,
                                "answer": answer,
                                "contexts": [c["text"] for c in chunks],
                            })

                record[name] = entry

            records.append(record)

        all_records.extend(records)

        ragas_scores: dict = {}
        if args.with_ragas:
            ragas_scores = _run_ragas(ragas_rows)

        per_seed_summaries.append(_build_summary(records, args.with_generation, ragas_scores))

    sig_tests = _significance_tests(all_records, args.with_generation)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({
                "dataset": args.dataset,
                "per_seed_summaries": per_seed_summaries,
                "significance": sig_tests,
                "seeds": seeds,
                "records": all_records,
            }, fh, indent=2)


def _run_ragas(rows_by_pipeline: dict, llm_model: str = "qwen3:8b",
               ragas_workers: int = 4) -> dict:
    """Evaluate faithfulness and answer_relevancy with RAGAS for each pipeline.

    Grounded-refusal answers (containing ``REFUSAL_MARKER``) are excluded
    from scoring because penalising a correct abstention distorts the metric.

    ``num_ctx=4096`` is intentional: RAGAS prompts are well under 4 K tokens,
    so the model's 40 K default wastes KV-cache and reduces parallelism.
    """
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import Faithfulness, ResponseRelevancy
        from langchain_ollama import ChatOllama
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError as exc:
        print(f"\nRAGAS dependencies missing: {exc}")
        return {}

    llm = LangchainLLMWrapper(ChatOllama(model=llm_model, num_ctx=4096))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    results: dict = {}
    for name, rows in rows_by_pipeline.items():
        if not rows:
            continue

        usable = [r for r in rows
                  if REFUSAL_MARKER not in r["answer"].lower()
                  and not r["answer"].startswith("[ERROR:")]
        if not usable:
            continue

        samples = [
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
            )
            for r in usable
        ]
        dataset = EvaluationDataset(samples=samples)
        from ragas import RunConfig
        result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), ResponseRelevancy()],
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=ragas_workers, timeout=600),
        )
        df = result.to_pandas()
        faith_vals = [v for v in df["faithfulness"] if not math.isnan(v)]
        relev_vals = [v for v in df["answer_relevancy"] if not math.isnan(v)]
        results[name] = {
            "faithfulness": round(_avg(faith_vals), 4),
            "answer_relevancy": round(_avg(relev_vals), 4),
        }

    return results


def _context_rank(chunks: list[dict], gold_answers: list[str],
                  gold_title: str | None = None) -> int | None:
    """Return the 1-based rank of the first chunk containing a gold answer span.

    When *gold_title* is given, a chunk only counts if it also comes from the
    correct article. This prevents short answer spans (e.g. "May", "US") from
    spuriously matching unrelated passages and inflating recall.
    """
    golds = [a.lower() for a in gold_answers]
    title = gold_title.strip().lower() if gold_title else None
    for i, chunk in enumerate(chunks):
        if title is not None and chunk.get("source", "").strip().lower() != title:
            continue
        text = chunk["text"].lower()
        if any(ans in text for ans in golds):
            return i + 1
    return None


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    """Compute SQuAD-style token-F1 between prediction and reference."""
    pred_tokens = _normalize(prediction).split()
    ref_tokens  = _normalize(reference).split()
    common      = Counter(pred_tokens) & Counter(ref_tokens)
    num_common  = sum(common.values())
    if num_common == 0 or not pred_tokens or not ref_tokens:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, reference: str) -> bool:
    """Return True if the normalised reference is a substring of normalised prediction."""
    return _normalize(reference) in _normalize(prediction)


def _avg(values: list) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def _build_summary(records: list[dict], with_generation: bool, ragas_scores: dict) -> dict:
    """Build a per-seed metric summary from evaluated records.

    Retrieval and answer-quality metrics are computed over answerable questions
    only; the hallucination metric is computed over the unanswerable ones.
    """
    answerable = [r for r in records if r.get("answerable", True)]
    negatives = [r for r in records if not r.get("answerable", True)]

    summary = {}
    for name in ("naive", "advanced"):
        entries = [r[name] for r in records]
        ans = [r[name] for r in answerable]
        s: dict = {
            "n": len(entries),
            "context_recall": round(_avg([e["recall"] for e in ans]), 4),
            "mrr":            round(_avg([e["mrr"]    for e in ans]), 4),
            "avg_retrieval_ms": round(_avg([e["retrieval_ms"] for e in entries]), 2),
        }
        if with_generation:
            s["answer_f1"]   = round(_avg([e.get("f1",  0.0)          for e in ans]), 4)
            s["exact_match"] = round(_avg([float(e.get("em", False))   for e in ans]), 4)
        if negatives:
            neg = [r[name] for r in negatives]
            s["hallucination_rate"] = round(
                _avg([0.0 if e.get("abstained") else 1.0 for e in neg]), 4
            )
        if name in ragas_scores:
            s["faithfulness"]     = ragas_scores[name]["faithfulness"]
            s["answer_relevancy"] = ragas_scores[name]["answer_relevancy"]
        summary[name] = s
    return summary


def _significance_tests(records: list[dict], with_generation: bool) -> dict:
    """Run paired Advanced-vs-Naive significance tests over pooled per-question results.

    Continuous metrics (MRR, answer_f1) use the Wilcoxon signed-rank test.
    Binary metrics (recall, exact_match) use McNemar's test (exact binomial).
    All tests run over answerable questions only; the hallucination abstention
    test runs over the unanswerable subset.
    """
    try:
        from scipy.stats import binomtest, wilcoxon
    except ImportError:
        return {}

    answerable = [r for r in records if r.get("answerable", True)]
    negatives = [r for r in records if not r.get("answerable", True)]

    tests: dict = {}

    continuous = [("MRR", "mrr")]
    if with_generation:
        continuous.append(("Answer F1", "f1"))
    for label, key in continuous:
        naive_vals = [r["naive"][key] for r in answerable]
        adv_vals = [r["advanced"][key] for r in answerable]
        if not answerable or all(av == nv for av, nv in zip(adv_vals, naive_vals)):
            p = float("nan")
        else:
            try:
                _, p = wilcoxon(adv_vals, naive_vals)
            except ValueError:
                p = float("nan")
        tests[label] = {"test": "Wilcoxon", "p": p, "n": len(answerable)}

    binary = [("Context Recall@K", "recall")]
    if with_generation:
        binary.append(("Exact Match", "em"))
    for label, key in binary:
        adv_only = sum(1 for r in answerable if r["advanced"][key] and not r["naive"][key])
        naive_only = sum(1 for r in answerable if r["naive"][key] and not r["advanced"][key])
        discordant = adv_only + naive_only
        p = 1.0 if discordant == 0 else binomtest(min(adv_only, naive_only), discordant, 0.5).pvalue
        tests[label] = {
            "test": "McNemar", "p": p, "n": len(answerable),
            "adv_better": adv_only, "naive_better": naive_only,
        }

    if negatives:
        adv_only = sum(1 for r in negatives if r["advanced"].get("abstained") and not r["naive"].get("abstained"))
        naive_only = sum(1 for r in negatives if r["naive"].get("abstained") and not r["advanced"].get("abstained"))
        discordant = adv_only + naive_only
        p = 1.0 if discordant == 0 else binomtest(min(adv_only, naive_only), discordant, 0.5).pvalue
        tests["Abstention (neg)"] = {
            "test": "McNemar", "p": p, "n": len(negatives),
            "adv_better": adv_only, "naive_better": naive_only,
        }

    return tests


if __name__ == "__main__":
    main()
