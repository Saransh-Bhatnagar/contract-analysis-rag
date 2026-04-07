import os
import numpy as np
from query_rag import (
    embed_query, supabase_client, expand_query
)

query = "do you see any outliers in terms of money?"
query_vector = embed_query(query)
expanded = expand_query(query)
expanded_vector = embed_query(expanded)

# get all chunks for MSA_With_Outlier
response = supabase_client.table('contract_chunks').select('id, content, embedding').eq('document_name', 'MSA_With_Outlier.pdf').execute()

chunks = response.data
print(f"Total chunks: {len(chunks)}")

v1 = np.array(query_vector)
v2 = np.array(expanded_vector)

results = []
for c in chunks:
    emb = np.array(c['embedding'])
    sim_orig = np.dot(v1, emb)
    sim_exp = np.dot(v2, emb)
    results.append({
        'content': c['content'],
        'sim_orig': sim_orig,
        'sim_exp': sim_exp
    })

results.sort(key=lambda x: max(x['sim_orig'], x['sim_exp']), reverse=True)

for r in results[:10]:
    txt = r['content'].replace('\n', ' ')[:80]
    print(f"Orig: {r['sim_orig']:.3f} | Exp: {r['sim_exp']:.3f} | {txt}")
