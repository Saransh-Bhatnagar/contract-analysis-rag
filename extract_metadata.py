"""
extract_metadata.py
-------------------
One-shot script that extracts structured metadata (dates, term, renewal type,
parties, governing law, contract type) from every contract already ingested
into `contract_chunks`, and upserts the result into `contract_metadata`.

This is what makes date-based questions ("contracts expiring in May 2026",
"pending renewals", "auto-renewing agreements") exact instead of approximate.
Vector search is good at semantics, bad at arithmetic — so we extract the
dates once, deterministically, and let SQL handle the filtering.

Required env vars (same as ingest_to_db.py):
  OPENAI_API_KEY
  SUPABASE_URL / SUPABASE_KEY

Supabase schema (run once in the SQL editor):

  create table if not exists contract_metadata (
      document_name              text primary key,
      effective_date             date,
      expiration_date            date,
      term_months                integer,
      renewal_type               text,
      auto_renewal_notice_days   integer,
      parties                    text[],
      governing_law              text,
      contract_type              text,
      extracted_at               timestamptz default now()
  );

  create index if not exists contract_metadata_expiration_idx
      on contract_metadata(expiration_date);
  create index if not exists contract_metadata_renewal_idx
      on contract_metadata(renewal_type);

Usage:
  python extract_metadata.py            # only processes contracts not yet extracted
  python extract_metadata.py --force    # re-extracts every contract
  python extract_metadata.py --only "MSA_Logistics.pdf"   # single contract
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Env + clients
# ---------------------------------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not (OPENAI_API_KEY and SUPABASE_URL and SUPABASE_KEY):
    sys.exit("Missing OPENAI_API_KEY / SUPABASE_URL / SUPABASE_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CHUNKS_TABLE = "contract_chunks"
METADATA_TABLE = "contract_metadata"
EXTRACTION_MODEL = "gpt-4o-mini"

# Dates usually live in the preamble / first few pages. Pulling the first
# ~20 chunks ≈ first few thousand tokens — enough to find effective date,
# term, parties, governing law without sending the whole contract.
CHUNKS_PER_CONTRACT = 20

EXTRACTION_PROMPT = """You extract structured metadata from a contract excerpt.

Rules:
- Return ISO dates (YYYY-MM-DD). If only month + year are given, use the first of the month.
- If the effective date is not stated but there's a signing/execution date, use that.
- If the contract gives a term in years, convert to months (e.g. "three (3) years" → 36).
- expiration_date: if stated explicitly, use it. If only term + effective date are given,
  compute effective_date + term_months. Otherwise null.
- renewal_type:
    "auto"    → contract auto-renews unless a party gives notice
    "manual"  → parties must affirmatively renew/extend
    "none"    → contract has a fixed end with no renewal provision
    "unknown" → can't tell from the excerpt
- auto_renewal_notice_days: days of notice required to prevent auto-renewal (integer). Null otherwise.
- parties: each entry formatted as "Role: Entity Name". Use the role that matches
  the contract type, e.g. "Lessor: Acme Corp" / "Lessee: Beta LLC",
  "Licensor: X" / "Licensee: Y", "Client: X" / "Vendor: Y" (or "Provider"),
  "Buyer: X" / "Seller: Y", "Employer: X" / "Employee: Y",
  "Disclosing Party: X" / "Receiving Party: Y", "Prime: X" / "Subcontractor: Y".
  Omit boilerplate like "(hereinafter 'Vendor')". If the role genuinely
  cannot be determined, use "Party: <name>".
- governing_law: the jurisdiction (e.g. "Delaware", "New York", "England and Wales").
- contract_type: short label like "MSA", "licensing", "NDA", "services", "distribution",
  "supply", "employment", "franchise". Use lowercase.

Return null for any field the excerpt does not clearly support. Do NOT guess."""


def fetch_all_document_names() -> list[str]:
    """Return every distinct document_name currently in contract_chunks."""
    # Paginate — Supabase returns 1000 rows max per request
    names: set[str] = set()
    page = 0
    page_size = 1000
    while True:
        resp = (
            supabase_client.table(CHUNKS_TABLE)
            .select("document_name")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        rows = resp.data or []
        if not rows:
            break
        for r in rows:
            names.add(r["document_name"])
        if len(rows) < page_size:
            break
        page += 1
    return sorted(names)


def fetch_already_extracted() -> set[str]:
    resp = supabase_client.table(METADATA_TABLE).select("document_name").execute()
    return {r["document_name"] for r in (resp.data or [])}


def fetch_contract_excerpt(document_name: str) -> str:
    """Grab the first N chunks (by chunk_index) for a contract."""
    resp = (
        supabase_client.table(CHUNKS_TABLE)
        .select("content, chunk_index")
        .eq("document_name", document_name)
        .order("chunk_index")
        .limit(CHUNKS_PER_CONTRACT)
        .execute()
    )
    rows = resp.data or []
    return "\n\n---\n\n".join(r["content"] for r in rows)


METADATA_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "effective_date",
        "expiration_date",
        "term_months",
        "renewal_type",
        "auto_renewal_notice_days",
        "parties",
        "governing_law",
        "contract_type",
    ],
    "properties": {
        "effective_date": {"type": ["string", "null"]},
        "expiration_date": {"type": ["string", "null"]},
        "term_months": {"type": ["integer", "null"]},
        "renewal_type": {
            "type": "string",
            "enum": ["auto", "manual", "none", "unknown"],
        },
        "auto_renewal_notice_days": {"type": ["integer", "null"]},
        "parties": {"type": "array", "items": {"type": "string"}},
        "governing_law": {"type": ["string", "null"]},
        "contract_type": {"type": ["string", "null"]},
    },
}


def extract_metadata(document_name: str, excerpt: str) -> dict:
    """Call the LLM with a strict JSON schema and return parsed metadata."""
    completion = openai_client.chat.completions.create(
        model=EXTRACTION_MODEL,
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "contract_metadata",
                "strict": True,
                "schema": METADATA_SCHEMA,
            },
        },
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Contract filename: {document_name}\n\n"
                    f"Excerpt (first {CHUNKS_PER_CONTRACT} chunks):\n\n{excerpt}"
                ),
            },
        ],
    )
    return json.loads(completion.choices[0].message.content)


def upsert_metadata(document_name: str, meta: dict) -> None:
    payload = {"document_name": document_name, **meta}
    supabase_client.table(METADATA_TABLE).upsert(payload).execute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-extract even if already extracted")
    parser.add_argument("--only", help="Only process this single document_name")
    args = parser.parse_args()

    if args.only:
        doc_names = [args.only]
    else:
        doc_names = fetch_all_document_names()

    if not doc_names:
        print("No contracts found in contract_chunks.")
        return

    already = set() if args.force else fetch_already_extracted()
    to_process = [d for d in doc_names if d not in already]

    print(f"Contracts total: {len(doc_names)}")
    print(f"Already extracted: {len(already)}")
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("Nothing to do. Pass --force to re-extract.")
        return

    successes = 0
    failures: list[tuple[str, str]] = []

    for doc in tqdm(to_process, desc="Extracting", unit="contract"):
        try:
            excerpt = fetch_contract_excerpt(doc)
            if not excerpt.strip():
                failures.append((doc, "empty excerpt"))
                continue
            meta = extract_metadata(doc, excerpt)
            upsert_metadata(doc, meta)
            successes += 1
        except Exception as exc:
            failures.append((doc, str(exc)))

    print("\n" + "=" * 60)
    print(f"  Extracted: {successes}")
    print(f"  Failed:    {len(failures)}")
    print("=" * 60)
    for doc, err in failures:
        print(f"  ❌ {doc}: {err}")


if __name__ == "__main__":
    main()