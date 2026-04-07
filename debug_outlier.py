import os
import sys

# Redirect stdout to a file
sys.stdout = open('debug_out.txt', 'w', encoding='utf-8')

from query_rag import (
    expand_query, embed_query, search_contracts,
    keyword_search, fetch_preambles, merge_chunks,
    generate_answer, get_cross_encoder, TOP_K_CHUNKS
)

query = "do you see any outliers in terms of money?"
print(f"Question: {query}")
expanded = expand_query(query)
print(f"Expanded: {expanded}")

query_vector = embed_query(query)
vector_chunks_orig = search_contracts(query_vector)

vector_chunks_exp = []
if expanded != query:
    expanded_vector = embed_query(expanded)
    vector_chunks_exp = search_contracts(expanded_vector)

fts_chunks_orig = keyword_search(query)
fts_chunks_exp = keyword_search(expanded) if expanded != query else []

print("\n--- Vector Orig ---")
for c in vector_chunks_orig: print(f"{c['document_name']} : {c['similarity']:.3f} : {c['content'][:50]}...")

print("\n--- Vector Exp ---")
for c in vector_chunks_exp: print(f"{c['document_name']} : {c['similarity']:.3f} : {c['content'][:50]}...")

print("\n--- FTS Orig ---")
for c in fts_chunks_orig: print(f"{c.get('document_name', '')} : {c['content'][:50]}...")

print("\n--- FTS Exp ---")
for c in fts_chunks_exp: print(f"{c.get('document_name', '')} : {c['content'][:50]}...")

initial_chunks = merge_chunks(vector_chunks_orig, vector_chunks_exp, fts_chunks_orig, fts_chunks_exp)
unique_docs = list(set(chunk.get("document_name") for chunk in initial_chunks if chunk.get("document_name")))

print(f"\nUnique docs: {unique_docs}")

preamble_chunks = fetch_preambles(unique_docs)
all_chunks = merge_chunks(initial_chunks, preamble_chunks)

pairs = [[query, chunk.get('content', '')] for chunk in all_chunks]
scores = get_cross_encoder().predict(pairs)

for chunk, score in zip(all_chunks, scores):
    chunk['rerank_score'] = float(score)
    if chunk.get('chunk_index') == 0:
        chunk['rerank_score'] += 10.0

all_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
top_chunks = all_chunks[:TOP_K_CHUNKS]

print("\n--- Top Chunks before LLM ---")
for i, c in enumerate(top_chunks):
     print(f" {i+1}. {c['document_name']} (Score: {c['rerank_score']:.3f}, Index: {c.get('chunk_index', 'N/A')}) : {c['content'][:50]}...")
