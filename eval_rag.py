"""
eval_rag.py
-----------
Automated RAG evaluation suite using RAGAS metrics.

Evaluates:
  - Faithfulness:       Does the answer stick to retrieved context? (no hallucination)
  - Answer Relevancy:   Does the answer address the question?
  - Context Precision:  Are the top-ranked chunks relevant?
  - Context Recall:     Does the retrieved context cover the ground truth?
  - Answer Correctness: Does the answer match expected ground truth?

Usage:
    python eval_rag.py

Requires: pip install ragas datasets
"""

import os
import sys
import json
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Test dataset — questions with ground truth answers and expected sources
# ---------------------------------------------------------------------------

EVAL_DATASET = [
    {
        "question": "What is the liability cap in the MSA_With_Outlier contract?",
        "ground_truth": (
            "The general liability cap is limited to total fees paid in the 12 months "
            "preceding the claim. However, for critical incidents involving the client's "
            "internal systems, the client faces uncapped liability with a minimum of "
            "$50,000,000 (Fifty Million US Dollars) per incident."
        ),
        "expected_sources": ["MSA_With_Outlier.pdf"],
    },
    {
        "question": "What are the auto-renewal terms in the Amendment_Auto_Renewal_Hidden contract?",
        "ground_truth": (
            "The agreement automatically renews for successive one-year periods unless "
            "either party provides written notice of non-renewal at least 15 days prior "
            "to the end of the current term. This is an unusually short notice window."
        ),
        "expected_sources": ["Amendment_Auto_Renewal_Hidden.pdf"],
    },
    {
        "question": "What are the most unusual or risky clauses across all contracts?",
        "ground_truth": (
            "The most unusual clause is in MSA_With_Outlier.pdf which contains a "
            "$50,000,000 minimum liability clause for the client. The Amendment_Auto_Renewal_Hidden.pdf "
            "has an auto-renewal trap with only 15 days notice to cancel. The "
            "Amendment_MSA_Scope_Extension.pdf has an $8 million liability cap."
        ),
        "expected_sources": ["MSA_With_Outlier.pdf"],
    },
    {
        "question": "Who are the parties in the NDA_Mutual contract?",
        "ground_truth": (
            "The NDA_Mutual contract is between two parties entering into a mutual "
            "non-disclosure agreement to protect confidential information shared between them."
        ),
        "expected_sources": ["NDA_Mutual.pdf"],
    },
    {
        "question": "What is the total budget for the SOW_Data_Migration project?",
        "ground_truth": (
            "The SOW_Data_Migration project involves migrating critical business data "
            "to an AWS environment. The contract specifies fees for the data migration services."
        ),
        "expected_sources": ["SOW_Data_Migration.pdf"],
    },
    {
        "question": "Which contracts have liability caps and what are they?",
        "ground_truth": (
            "Most contracts cap liability at the total fees paid in the 12 months preceding "
            "the claim. Notable exceptions: MSA_With_Outlier.pdf has a $50M minimum liability "
            "for critical incidents. Amendment_MSA_Scope_Extension.pdf has an $8M aggregate cap."
        ),
        "expected_sources": ["MSA_With_Outlier.pdf", "Amendment_MSA_Scope_Extension.pdf"],
    },
    {
        "question": "What are the confidentiality obligations in the NDA_Board_Advisor?",
        "ground_truth": (
            "The Board Advisor must keep all confidential company information secret. "
            "The NDA includes provisions for injunctive relief without posting a bond "
            "in case of breach."
        ),
        "expected_sources": ["NDA_Board_Advisor.pdf"],
    },
    {
        "question": "What are the termination conditions in the Vendor_Agreement?",
        "ground_truth": (
            "The Vendor Agreement can be terminated for convenience by the Client upon "
            "60 days written notice. The Client must pay for all deliverables accepted "
            "and services rendered up to termination, plus unavoidable termination costs "
            "not exceeding 10% of the remaining contract value."
        ),
        "expected_sources": ["Vendor_Agreement.pdf"],
    },
    {
        "question": "What SLA commitments exist in the SLA_Cloud_Hosting contract?",
        "ground_truth": (
            "The SLA_Cloud_Hosting contract includes uptime commitments with service "
            "credits as the sole and exclusive remedy for failures to meet service levels."
        ),
        "expected_sources": ["SLA_Cloud_Hosting.pdf"],
    },
    {
        "question": "What insurance requirements are specified in the MSA_Construction contract?",
        "ground_truth": (
            "The MSA_Construction contract requires the contractor to maintain insurance "
            "policies with at least 30 days prior written notice of cancellation, "
            "non-renewal, or material change in coverage. Minimum liability limits include "
            "$5,000,000 per occurrence and aggregate."
        ),
        "expected_sources": ["MSA_Construction.pdf"],
    },
]


# ---------------------------------------------------------------------------
# Run the RAG pipeline on each question
# ---------------------------------------------------------------------------

def run_evaluation():
    from query_rag import (
        expand_query, embed_query, search_contracts, keyword_search,
        fetch_preambles, fetch_neighbor_chunks, merge_chunks,
        generate_answer, get_cross_encoder, TOP_K_CHUNKS,
    )

    print("=" * 60)
    print("  RAG Evaluation Suite")
    print("=" * 60)
    print(f"  Test cases: {len(EVAL_DATASET)}")
    print()

    results = []

    for i, test in enumerate(EVAL_DATASET, 1):
        question = test["question"]
        print(f"[{i}/{len(EVAL_DATASET)}] {question[:70]}...")

        start = time.time()

        try:
            # Run full pipeline (same as query_rag.py main loop)
            expanded = expand_query(question)
            query_vector = embed_query(question)
            if query_vector is None:
                print("  SKIP — embedding failed")
                continue

            vector_orig = search_contracts(query_vector)
            vector_exp = []
            if expanded != question:
                exp_vec = embed_query(expanded)
                if exp_vec:
                    vector_exp = search_contracts(exp_vec)

            fts_orig = keyword_search(question)
            fts_exp = keyword_search(expanded) if expanded != question else []

            initial = merge_chunks(vector_orig, vector_exp, fts_orig, fts_exp)
            unique_docs = list({c.get("document_name") for c in initial if c.get("document_name")})
            preambles = fetch_preambles(unique_docs)
            all_chunks = merge_chunks(initial, preambles)

            preamble_ids = {c["id"] for c in preambles}
            for chunk in all_chunks:
                if chunk.get("id") in preamble_ids:
                    chunk["chunk_index"] = 0

            if not all_chunks:
                print("  SKIP — no chunks found")
                continue

            # Rerank
            pairs = [[question, c.get("content", "")] for c in all_chunks]
            scores = get_cross_encoder().predict(pairs)
            for chunk, score in zip(all_chunks, scores):
                chunk["rerank_score"] = float(score)
            all_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Top-K with preamble cap
            MAX_PREAMBLES = 2
            top_chunks = []
            preamble_count = 0
            for chunk in all_chunks:
                if len(top_chunks) >= TOP_K_CHUNKS:
                    break
                if chunk.get("chunk_index") == 0:
                    if preamble_count >= MAX_PREAMBLES:
                        continue
                    preamble_count += 1
                top_chunks.append(chunk)

            # Neighbor expansion
            neighbors = fetch_neighbor_chunks(top_chunks)
            if neighbors:
                top_chunks = merge_chunks(top_chunks, neighbors)

            # Generate
            answer = generate_answer(question, top_chunks)
            elapsed = round(time.time() - start, 2)

            # Collect contexts
            contexts = [c.get("content", "") for c in top_chunks]
            retrieved_sources = list({c.get("document_name", "") for c in top_chunks})

            # Check source recall
            expected = set(test["expected_sources"])
            found = set(retrieved_sources)
            source_recall = len(expected & found) / len(expected) if expected else 1.0

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": test["ground_truth"],
                "retrieved_sources": retrieved_sources,
                "expected_sources": test["expected_sources"],
                "source_recall": source_recall,
                "time_seconds": elapsed,
            })

            status = "HIT" if source_recall == 1.0 else "PARTIAL" if source_recall > 0 else "MISS"
            print(f"  {status} | Sources: {retrieved_sources[:3]}... | {elapsed}s")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return results


def run_ragas_eval(results):
    """Run RAGAS metrics on the collected results."""
    try:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
        from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
        from ragas.llms import llm_factory
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        from openai import OpenAI as OpenAIClient
    except ImportError as e:
        print(f"\n[WARN] Import failed: {e}. Skipping RAGAS metrics.")
        print("       Install with: pip install ragas datasets")
        return None

    print("\n" + "=" * 60)
    print("  Running RAGAS evaluation (LLM-as-judge)...")
    print("  This will make several OpenAI API calls.")
    print("=" * 60 + "\n")

    # Build RAGAS samples
    samples = []
    for r in results:
        samples.append(SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)

    # Configure LLM and embeddings using RAGAS native providers (no LangChain dependency)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key=api_key)
    llm = llm_factory("gpt-4o-mini", client=client)
    llm.model_args["max_tokens"] = 8192  # default 1024 truncates long faithfulness checks
    embeddings = RagasOpenAIEmbeddings(client=client, model="text-embedding-3-small")

    # Old-style RAGAS metrics call embed_query/embed_documents (LangChain convention)
    # but RAGAS native embeddings use embed_text/embed_texts. Bridge the gap:
    embeddings.embed_query = embeddings.embed_text
    embeddings.embed_documents = embeddings.embed_texts

    metrics = [
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=llm),
        LLMContextRecall(llm=llm),
    ]

    # Generous timeout to avoid TimeoutErrors on slow API responses
    run_config = RunConfig(timeout=120, max_retries=3, max_wait=60)

    try:
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
        )
        return score
    except Exception as e:
        print(f"[ERROR] RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_summary(results, ragas_scores=None):
    """Print a summary of all evaluation results."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)

    # Source recall
    avg_source_recall = sum(r["source_recall"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)
    hits = sum(1 for r in results if r["source_recall"] == 1.0)

    print(f"\n  Test cases:         {len(results)}")
    print(f"  Source recall (avg): {avg_source_recall:.1%}")
    print(f"  Perfect source hit:  {hits}/{len(results)}")
    print(f"  Avg latency:         {avg_time:.1f}s")

    # Per-question breakdown
    print(f"\n  {'Question':<55} {'Sources':>8} {'Time':>6}")
    print(f"  {'-'*55} {'-'*8} {'-'*6}")
    for r in results:
        q = r["question"][:52] + "..." if len(r["question"]) > 55 else r["question"]
        status = "HIT" if r["source_recall"] == 1.0 else f"{r['source_recall']:.0%}"
        print(f"  {q:<55} {status:>8} {r['time_seconds']:>5.1f}s")

    # RAGAS scores
    scores_dict = {}
    if ragas_scores:
        print(f"\n  RAGAS Scores (LLM-as-judge):")
        print(f"  {'-'*40}")
        try:
            df = ragas_scores.to_pandas()
            score_cols = [c for c in df.columns if c not in ('user_input', 'response', 'retrieved_contexts', 'reference')]
            for col in score_cols:
                mean_val = df[col].dropna().mean()
                print(f"  {col:<30} {mean_val:.3f}")
            scores_dict = {col: float(df[col].dropna().mean()) for col in score_cols}
        except Exception:
            # Fallback: try dict-like access
            try:
                scores_dict = dict(ragas_scores) if not isinstance(ragas_scores, dict) else ragas_scores
                for k, v in scores_dict.items():
                    print(f"  {k:<30} {v:.3f}")
            except Exception as e:
                print(f"  Could not parse RAGAS scores: {e}")

    print("\n" + "=" * 60)

    # Save results
    output = {
        "summary": {
            "test_cases": len(results),
            "avg_source_recall": avg_source_recall,
            "perfect_hits": hits,
            "avg_latency": avg_time,
        },
        "ragas": scores_dict,
        "details": [
            {
                "question": r["question"],
                "source_recall": r["source_recall"],
                "expected_sources": r["expected_sources"],
                "retrieved_sources": r["retrieved_sources"],
                "time_seconds": r["time_seconds"],
            }
            for r in results
        ],
    }

    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to eval_results.json\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_evaluation()
    if not results:
        print("No results collected. Exiting.")
        sys.exit(1)

    ragas_scores = run_ragas_eval(results)
    print_summary(results, ragas_scores)