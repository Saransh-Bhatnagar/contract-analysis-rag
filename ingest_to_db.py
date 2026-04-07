"""
ingest_to_db.py
---------------
Reads all PDF contracts from dummy_dataset/pdfs/, chunks them with
LangChain's RecursiveCharacterTextSplitter, embeds each chunk with
OpenAI text-embedding-3-small, and upserts the results into a
Supabase table called `contract_chunks`.

Required environment variables (in .env):
  OPENAI_API_KEY
  SUPABASE_URL  (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_KEY  (or NEXT_PUBLIC_SUPABASE_ANON_KEY)

Supabase table schema (run once in the SQL editor):
  create extension if not exists vector;

  create table if not exists contract_chunks (
      id           text primary key,
      document_name text not null,
      content      text not null,
      embedding    vector(1536),
      chunk_index  integer
  );

  -- Optional: index for fast ANN search
  -- NOTE: lists should be ~sqrt(row_count). For <500 rows use lists=10,
  -- for ~1000 rows use lists=30, for 10k+ rows use lists=100.
  create index if not exists contract_chunks_embedding_idx
      on contract_chunks using ivfflat (embedding vector_cosine_ops)
      with (lists = 10);
"""

import os
import uuid
import pathlib

import pymupdf4llm
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from openai import OpenAI
from supabase import create_client, Client
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 1. Environment setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Support both plain and Next.js-prefixed variable names
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set in the .env file.")
if not SUPABASE_URL:
    raise EnvironmentError(
        "SUPABASE_URL (or NEXT_PUBLIC_SUPABASE_URL) is not set in the .env file."
    )
if not SUPABASE_KEY:
    raise EnvironmentError(
        "SUPABASE_KEY (or NEXT_PUBLIC_SUPABASE_ANON_KEY) is not set in the .env file."
    )

# ---------------------------------------------------------------------------
# 2. Client initialisation
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# 3. Configuration
# ---------------------------------------------------------------------------

PDF_DIR = pathlib.Path(os.environ.get("PDF_DIR", "dummy_dataset/pdfs"))
TABLE_NAME = "contract_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200   # characters (slightly larger for semantic integrity)
CHUNK_OVERLAP = 200  # characters — preserves legal context across clause boundaries

# Headers to split on for Markdown
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False
)

fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ---------------------------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_path: pathlib.Path) -> str:
    """Extract all text from a PDF as Markdown using pymupdf4llm."""
    return pymupdf4llm.to_markdown(str(pdf_path))


def chunk_text(text: str) -> list[str]:
    """Split markdown text into semantic chunks, using a character fallback for large blocks."""
    # 1. First split by markdown headers
    md_splits = markdown_splitter.split_text(text)

    # 2. Then constrain chunk sizes with the character splitter
    final_chunks = []
    for split in md_splits:
        # Build a section path from header metadata (e.g., "LIMITATION OF LIABILITY > Cap on Damages")
        # This gives each chunk context about where it sits in the document structure,
        # improving both embedding quality and LLM comprehension.
        header_parts = [v for k, v in split.metadata.items() if v]
        section_prefix = " > ".join(header_parts)

        sub_chunks = fallback_splitter.split_text(split.page_content)
        for sc in sub_chunks:
            if section_prefix and section_prefix not in sc:
                final_chunks.append(f"[Section: {section_prefix}]\n{sc}")
            else:
                final_chunks.append(sc)

    return final_chunks


def generate_embedding(text: str) -> list[float]:
    """Call the OpenAI Embeddings API and return the embedding vector."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


EMBED_BATCH_SIZE = 50  # OpenAI allows up to 2048, but 50 balances speed vs payload size


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in a single API call. Much faster than one-at-a-time."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # OpenAI returns embeddings sorted by index
    return [item.embedding for item in response.data]


def make_chunk_id(document_name: str, chunk_index: int, chunk_text: str) -> str:
    """
    Generate a deterministic UUID (v5) for a chunk so it is compatible with
    Supabase's `uuid` primary key type. Same inputs always produce the same
    UUID, making repeated runs safely idempotent via upsert.
    """
    raw = f"{document_name}::{chunk_index}::{chunk_text[:64]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw))


def upsert_chunk(chunk_id: str, document_name: str, content: str, embedding: list[float], chunk_index: int) -> None:
    """Upsert a single chunk record into Supabase."""
    payload = {
        "id": chunk_id,
        "document_name": document_name,
        "content": content,
        "embedding": embedding,
        "chunk_index": chunk_index,
    }
    supabase_client.table(TABLE_NAME).upsert(payload).execute()


# ---------------------------------------------------------------------------
# 5. Main ingestion pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    # Use a set to deduplicate on case-insensitive Windows (*.pdf and *.PDF match same files)
    pdf_files = sorted({p.resolve() for p in PDF_DIR.glob("*.pdf")} | {p.resolve() for p in PDF_DIR.glob("*.PDF")})

    if not pdf_files:
        print(f"[WARNING] No PDF files found in '{PDF_DIR}'. Exiting.")
        return

    print(f"\n📂  Found {len(pdf_files)} PDF file(s) in '{PDF_DIR}'")
    print(f"🔧  Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"🤖  Embedding model: {EMBEDDING_MODEL}")
    print(f"🗄️   Supabase table: {TABLE_NAME}\n")

    total_chunks_uploaded = 0
    total_chunks_attempted = 0
    failed_chunks = 0

    for pdf_path in pdf_files:
        document_name = pdf_path.name
        print(f"📄  Processing: {document_name}")

        # --- Extract ---
        try:
            raw_text = extract_text_from_pdf(pdf_path)
        except Exception as exc:
            print(f"    ❌  Failed to extract text: {exc}")
            continue

        if not raw_text.strip():
            print(f"    ⚠️   No text extracted from {document_name}. Skipping.")
            continue

        # --- Chunk ---
        chunks = chunk_text(raw_text)
        print(f"    ✂️   {len(chunks)} chunks generated.")
        total_chunks_attempted += len(chunks)

        # --- Embed (batched) + Upsert ---
        for batch_start in tqdm(range(0, len(chunks), EMBED_BATCH_SIZE), desc=f"    Embedding & uploading", unit="batch", leave=False):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
            try:
                embeddings = generate_embeddings_batch(batch)
            except Exception as exc:
                failed_chunks += len(batch)
                tqdm.write(f"    ❌  Embedding batch starting at {batch_start} failed: {exc}")
                continue

            for i, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                idx = batch_start + i
                try:
                    chunk_id = make_chunk_id(document_name, idx, chunk)
                    upsert_chunk(chunk_id, document_name, chunk, embedding, idx)
                    total_chunks_uploaded += 1
                except Exception as exc:
                    failed_chunks += 1
                    tqdm.write(f"    ❌  Chunk {idx} upsert failed: {exc}")

        print(f"    ✅  Done with {document_name}\n")

    # ---------------------------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------------------------
    print("=" * 55)
    print("  INGESTION SUMMARY")
    print("=" * 55)
    print(f"  PDFs processed       : {len(pdf_files)}")
    print(f"  Total chunks created : {total_chunks_attempted}")
    print(f"  Successfully uploaded: {total_chunks_uploaded}")
    print(f"  Failed chunks        : {failed_chunks}")
    print("=" * 55)

    if total_chunks_uploaded == total_chunks_attempted:
        print("\n🎉  All chunks embedded and uploaded successfully!")
    else:
        print(f"\n⚠️   {failed_chunks} chunk(s) failed. Check the errors above.")


if __name__ == "__main__":
    main()
