"""
eval_cuad.py
------------
Validate our RAG pipeline against the CUAD benchmark dataset.
Ingests 10 real commercial contracts from CUAD, then tests whether
our pipeline retrieves the correct documents and produces answers
containing the CUAD-labeled ground truth spans.

Usage:
    python eval_cuad.py            # full run (ingest + eval)
    python eval_cuad.py --skip-ingest   # skip ingestion if already done
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Build test cases from CUAD labels
# ---------------------------------------------------------------------------

CUAD_JSON = "cuad_data/CUAD_v1/CUAD_v1.json"
CUAD_PDF_DIR = "cuad_eval/pdfs"

# Map CUAD titles to local PDF filenames
TITLE_TO_PDF = {
    "VIRTUALSCOPICS,INC_11_12_2010-EX-10.1-STRATEGIC ALLIANCE AGREEMENT":
        "VIRTUALSCOPICS,INC_11_12_2010-EX-10.1-STRATEGIC ALLIANCE AGREEMENT.PDF",
    "Sonos, Inc. - Manufacturing Agreement":
        "Sonos, Inc. - Manufacturing Agreement.PDF",
    "SigaTechnologiesInc_20190603_8-K_EX-10.1_11695818_EX-10.1_Promotion Agreement":
        "SigaTechnologiesInc_20190603_8-K_EX-10.1_11695818_EX-10.1_Promotion Agreement.pdf",
    "ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement":
        "ParatekPharmaceuticalsInc_20170505_10-KA_EX-10.29_10323872_EX-10.29_Outsourcing Agreement.pdf",
    "PHREESIA,INC_05_28_2019-EX-10.18-STRATEGIC ALLIANCE AGREEMENT":
        "PHREESIA,INC_05_28_2019-EX-10.18-STRATEGIC ALLIANCE AGREEMENT.PDF",
    "PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT":
        "PACIRA PHARMACEUTICALS, INC. - A_R STRATEGIC LICENSING, DISTRIBUTION AND MARKETING AGREEMENT.PDF",
    "NANOPHASETECHNOLOGIESCORP_11_01_2005-EX-99.1-DISTRIBUTOR AGREEMENT":
        "NANOPHASETECHNOLOGIESCORP_11_01_2005-EX-99.1-DISTRIBUTOR AGREEMENT.PDF",
    "Magenta Therapeutics, Inc. - Master Development and Manufacturing Agreement":
        "Magenta Therapeutics, Inc. - Master Development and Manufacturing Agreement.PDF",
    "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC ALLIANCE AGREEMENT":
        "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC ALLIANCE AGREEMENT.PDF",
    "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor Agreement":
        "ZogenixInc_20190509_10-Q_EX-10.2_11663313_EX-10.2_Distributor Agreement.pdf",
}

# Categories to test (these map well to natural questions)
CATEGORY_TO_QUESTION = {
    "Parties": "Who are the parties in the {doc} contract?",
    "Termination For Convenience": "What are the termination for convenience provisions in the {doc} contract?",
    "Cap On Liability": "What is the liability cap in the {doc} contract?",
    "Notice Period To Terminate Renewal": "What is the notice period to terminate renewal in the {doc} contract?",
    "Governing Law": "What is the governing law in the {doc} contract?",
    "Insurance": "What insurance requirements are in the {doc} contract?",
    "Expiration Date": "What is the expiration date or term of the {doc} contract?",
}


def load_cuad_labels():
    """Load CUAD labels and build test cases for our 10 contracts."""
    with open(CUAD_JSON) as f:
        data = json.load(f)

    test_cases = []
    for entry in data["data"]:
        title = entry["title"]
        if title not in TITLE_TO_PDF:
            continue

        pdf_name = TITLE_TO_PDF[title]
        short_name = pdf_name.rsplit(".", 1)[0]

        for p in entry["paragraphs"]:
            for qa in p["qas"]:
                q = qa["question"]
                if 'related to "' not in q:
                    continue
                cat = q.split('related to "')[1].split('"')[0]
                if cat not in CATEGORY_TO_QUESTION:
                    continue
                if qa.get("is_impossible") or not qa.get("answers"):
                    continue

                # Get all answer spans
                answer_spans = [a["text"].strip() for a in qa["answers"] if a["text"].strip()]
                if not answer_spans:
                    continue

                test_cases.append({
                    "pdf_name": pdf_name,
                    "category": cat,
                    "question": CATEGORY_TO_QUESTION[cat].format(doc=short_name),
                    "ground_truth_spans": answer_spans,
                    "expected_source": pdf_name,
                })

    return test_cases


# ---------------------------------------------------------------------------
# 2. Ingest CUAD contracts
# ---------------------------------------------------------------------------

def ingest_cuad_contracts():
    """Run the existing ingest script on the CUAD PDFs."""
    print("\n" + "=" * 60)
    print("  Ingesting 10 CUAD contracts into Supabase...")
    print("=" * 60 + "\n")

    import subprocess
    result = subprocess.run(
        [sys.executable, "ingest_to_db.py"],
        env={**os.environ, "PDF_DIR": CUAD_PDF_DIR},
        capture_output=True,
        text=True,
        timeout=600,
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print(f"[WARN] Ingest stderr: {result.stderr[-1000:]}")


# ---------------------------------------------------------------------------
# 3. Run RAG queries and check answers
# ---------------------------------------------------------------------------

def run_eval(test_cases):
    """Query the RAG pipeline and check if answers contain ground truth spans."""
    from query_rag import (
        expand_query, embed_query, search_contracts, keyword_search,
        fetch_preambles, fetch_all_doc_chunks, detect_target_document,
        merge_chunks, build_context_block, get_cross_encoder,
        generate_answer, enrich_reranker_query, DOC_BOOST, TOP_K_CHUNKS,
    )

    print("\n" + "=" * 60)
    print(f"  CUAD Validation — {len(test_cases)} test cases")
    print("=" * 60 + "\n")

    results = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        print(f"[{i}/{len(test_cases)}] {tc['category']}: {q[:80]}...")

        start = time.time()

        # Run retrieval pipeline (same logic as server.py)
        expanded = expand_query(q)
        query_vector = embed_query(q)
        if query_vector is None:
            print("  SKIP — embedding failed")
            continue

        vector_orig = search_contracts(query_vector)
        vector_exp = []
        if expanded != q:
            exp_vec = embed_query(expanded)
            if exp_vec:
                vector_exp = search_contracts(exp_vec)

        fts_orig = keyword_search(q)
        fts_exp = keyword_search(expanded) if expanded != q else []

        initial = merge_chunks(vector_orig, vector_exp, fts_orig, fts_exp)
        unique_docs = list({c.get("document_name") for c in initial if c.get("document_name")})
        preambles = fetch_preambles(unique_docs)
        all_chunks = merge_chunks(initial, preambles)

        preamble_ids = {c["id"] for c in preambles}
        for chunk in all_chunks:
            if chunk.get("id") in preamble_ids:
                chunk["chunk_index"] = 0

        target_doc = detect_target_document(q)
        if target_doc:
            doc_chunks = fetch_all_doc_chunks(target_doc)
            if doc_chunks:
                all_chunks = merge_chunks(all_chunks, doc_chunks)

        if not all_chunks:
            print("  SKIP — no chunks found")
            continue

        # Rerank
        pairs = [[q, c.get("content", "")] for c in all_chunks]
        scores = get_cross_encoder().predict(pairs)
        for chunk, score in zip(all_chunks, scores):
            boost = DOC_BOOST if target_doc and chunk.get("document_name") == target_doc else 0.0
            chunk["rerank_score"] = float(score) + boost
        all_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        effective_top_k = max(TOP_K_CHUNKS, 48) if target_doc else TOP_K_CHUNKS

        # When targeting a specific doc, filter out other docs to prevent contamination
        if target_doc:
            all_chunks = [c for c in all_chunks if c.get("document_name") == target_doc]

        top_chunks = all_chunks[:effective_top_k]

        # Check source recall
        retrieved_sources = list({c.get("document_name", "") for c in top_chunks})
        source_hit = tc["expected_source"] in retrieved_sources

        # Generate answer
        answer = generate_answer(q, top_chunks)
        elapsed = time.time() - start

        # Check if answer contains ground truth spans (fuzzy — check key terms)
        answer_lower = answer.lower()
        spans_found = 0
        for span in tc["ground_truth_spans"]:
            # Check if key terms from the span appear in the answer
            # Use first 50 chars of span as the key portion
            key_portion = span[:50].lower().strip()
            # Split into significant words (>3 chars)
            key_words = [w for w in key_portion.split() if len(w) > 3]
            if not key_words:
                key_words = key_portion.split()
            # If >60% of key words appear, count as found
            matches = sum(1 for w in key_words if w in answer_lower)
            if matches >= max(1, len(key_words) * 0.6):
                spans_found += 1

        span_recall = spans_found / len(tc["ground_truth_spans"]) if tc["ground_truth_spans"] else 0

        status = "HIT" if source_hit else "MISS"
        span_status = f"{span_recall:.0%}"
        print(f"  {status} | Span recall: {span_status} | {elapsed:.1f}s")

        results.append({
            "question": q,
            "category": tc["category"],
            "pdf_name": tc["pdf_name"],
            "source_hit": source_hit,
            "span_recall": span_recall,
            "spans_found": spans_found,
            "total_spans": len(tc["ground_truth_spans"]),
            "time_seconds": round(elapsed, 2),
            "answer": answer,
            "ground_truth_spans": tc["ground_truth_spans"],
        })

    return results


# ---------------------------------------------------------------------------
# 3b. LLM-as-judge semantic verification
# ---------------------------------------------------------------------------

def run_semantic_judge(results):
    """Use GPT-4o-mini to judge if our answers contain the same info as CUAD ground truth."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    JUDGE_PROMPT = """You are an expert legal contract reviewer acting as a judge.

You will be given:
1. A QUESTION about a contract
2. A REFERENCE CLAUSE (the ground truth from the contract)
3. An AI-GENERATED ANSWER

Your task: Does the AI answer convey the same key information as the reference clause?

Rules:
- The answer does NOT need to quote the clause verbatim — paraphrasing is fine.
- Focus on whether the essential facts (parties, dates, amounts, conditions, durations) match.
- If the answer covers the same substance with different wording, that is CORRECT.
- If the answer misses the key information or contradicts it, that is INCORRECT.

Respond with ONLY one of:
- CORRECT — the answer conveys the same key information
- PARTIAL — the answer covers some but not all key information
- INCORRECT — the answer misses or contradicts the key information"""

    print("\n" + "=" * 60)
    print("  Running LLM-as-judge semantic verification...")
    print("  This will make OpenAI API calls for each test case.")
    print("=" * 60 + "\n")

    judged = 0
    correct = 0
    partial = 0
    incorrect = 0

    for i, r in enumerate(results, 1):
        # Use first ground truth span as the reference
        ref_span = r["ground_truth_spans"][0][:500] if r["ground_truth_spans"] else ""
        if not ref_span:
            continue

        answer_text = r.get("answer", "")[:1500]

        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": (
                        f"QUESTION: {r['question']}\n\n"
                        f"REFERENCE CLAUSE: {ref_span}\n\n"
                        f"AI-GENERATED ANSWER: {answer_text}"
                    )},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = completion.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  [{i}] Judge error: {e}")
            verdict = "ERROR"

        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            correct += 1
            r["judge_verdict"] = "CORRECT"
        elif "PARTIAL" in verdict:
            partial += 1
            r["judge_verdict"] = "PARTIAL"
        else:
            incorrect += 1
            r["judge_verdict"] = verdict

        judged += 1
        if i % 10 == 0:
            print(f"  Judged {i}/{len(results)}...")

    print(f"\n  LLM Judge Results ({judged} cases):")
    print(f"  {'='*40}")
    print(f"  CORRECT:   {correct}/{judged} ({correct/judged:.0%})")
    print(f"  PARTIAL:   {partial}/{judged} ({partial/judged:.0%})")
    print(f"  INCORRECT: {incorrect}/{judged} ({incorrect/judged:.0%})")
    print(f"  Semantic accuracy: {(correct + partial * 0.5) / judged:.0%}")

    # Per-category breakdown
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        if "judge_verdict" in r:
            by_cat[r["category"]].append(r)

    print(f"\n  {'Category':<35} {'Correct':>8} {'Partial':>8} {'Wrong':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        c = sum(1 for r in rs if r["judge_verdict"] == "CORRECT")
        p = sum(1 for r in rs if r["judge_verdict"] == "PARTIAL")
        w = len(rs) - c - p
        print(f"  {cat:<35} {c:>7} {p:>8} {w:>8}")

    return results


# ---------------------------------------------------------------------------
# 4. Print summary
# ---------------------------------------------------------------------------

def print_summary(results):
    print("\n" + "=" * 60)
    print("  CUAD VALIDATION SUMMARY")
    print("=" * 60)

    if not results:
        print("  No results.")
        return

    source_hits = sum(1 for r in results if r["source_hit"])
    avg_span_recall = sum(r["span_recall"] for r in results) / len(results)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)

    print(f"\n  Test cases:         {len(results)}")
    print(f"  Source retrieval:   {source_hits}/{len(results)} ({source_hits/len(results):.0%})")
    print(f"  Avg span recall:   {avg_span_recall:.1%}")
    print(f"  Avg latency:       {avg_time:.1f}s")

    # Per-category breakdown
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)

    print(f"\n  {'Category':<35} {'Source':>7} {'Span':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7}")
    for cat in sorted(by_cat):
        cat_results = by_cat[cat]
        cat_source = sum(1 for r in cat_results if r["source_hit"]) / len(cat_results)
        cat_span = sum(r["span_recall"] for r in cat_results) / len(cat_results)
        print(f"  {cat:<35} {cat_source:>6.0%} {cat_span:>6.0%}")

    # Judge stats
    judged = [r for r in results if "judge_verdict" in r]
    judge_correct = sum(1 for r in judged if r["judge_verdict"] == "CORRECT")
    judge_partial = sum(1 for r in judged if r["judge_verdict"] == "PARTIAL")
    if judged:
        semantic_accuracy = round((judge_correct + judge_partial * 0.5) / len(judged), 3)
        print(f"\n  Semantic accuracy (LLM judge): {semantic_accuracy:.0%}")

    # Save results
    output = {
        "summary": {
            "test_cases": len(results),
            "source_retrieval_pct": round(source_hits / len(results), 3),
            "avg_span_recall": round(avg_span_recall, 3),
            "semantic_accuracy": semantic_accuracy if judged else None,
            "avg_latency": round(avg_time, 2),
        },
        "by_category": {
            cat: {
                "source_pct": round(sum(1 for r in rs if r["source_hit"]) / len(rs), 3),
                "span_recall": round(sum(r["span_recall"] for r in rs) / len(rs), 3),
            }
            for cat, rs in by_cat.items()
        },
        "details": [{k: v for k, v in r.items() if k != "ground_truth_spans"} for r in results],
    }
    with open("cuad_eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to cuad_eval_results.json")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    skip_ingest = "--skip-ingest" in sys.argv

    test_cases = load_cuad_labels()
    print(f"Loaded {len(test_cases)} test cases from CUAD labels")

    if not skip_ingest:
        ingest_cuad_contracts()

    results = run_eval(test_cases)
    results = run_semantic_judge(results)
    print_summary(results)