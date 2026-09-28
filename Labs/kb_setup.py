"""
ISDO - Knowledge Base ingestion into ChromaDB.

1. Reads every .md file in data/kb/
2. Splits each article into chunks at level-2 (##) headings
   (### sub-steps stay inside their parent ## section)
3. Stores all chunks in the ChromaDB collection 'isdo_kb'
4. Runs sample queries and prints the best-matching article + confidence

Dependencies: chromadb only (uses Chroma's built-in default embedding
model, all-MiniLM-L6-v2, which is downloaded once on first run).

Run from anywhere:
    python Labs/kb_chroma_ingest.py
"""

from pathlib import Path
import re

import chromadb

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # C:\ISDO Project
KB_DIR = PROJECT_ROOT / "data" / "kb"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"         # persistent store
COLLECTION_NAME = "isdo_kb"

# How many chunks to pull back per query before picking the best article
TOP_K_CHUNKS = 5

SAMPLE_QUERIES = [
    # (query, expected article file) - expected is only used for a PASS/CHECK label
    ("My VPN keeps saying authentication failed since I changed my password",
     "vpn_troubleshooting.md"),
    ("Outlook on my iPhone stopped syncing emails and keeps asking for my password",
     "email_troubleshooting.md"),
    ("The whole finance team gets DBCON_FAIL when logging into SAP",
     "erp_connectivity.md"),
    ("Everyone on the 3rd floor lost network and the switch is not responding",
     "network_outage.md"),
]


# --- 1 & 2: read and chunk -------------------------------------------------
H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
KB_ID_RE = re.compile(r"^(KB-[A-Z]+-\d+)")
CATEGORY_RE = re.compile(r"\*\*Category:\*\*\s*(.+)")


def split_article(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_name, section_text) at '## ' headings.

    Anything before the first '## ' (title + metadata) becomes an
    'Overview' chunk. '###' headings are NOT split points.
    """
    sections: list[tuple[str, str]] = []
    current_name = "Overview"
    current_lines: list[str] = []

    for line in text.splitlines():
        if line.startswith("## "):                      # exactly level 2
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_name, body))
            current_name = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    body = "\n".join(current_lines).strip()
    if body:
        sections.append((current_name, body))
    return sections


def load_chunks(kb_dir: Path):
    """Return parallel lists (ids, documents, metadatas) for every chunk."""
    ids, documents, metadatas = [], [], []
    md_files = sorted(kb_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {kb_dir}")

    for path in md_files:
        text = path.read_text(encoding="utf-8")

        title_match = H1_RE.search(text)
        title = title_match.group(1).strip() if title_match else path.stem
        kb_id_match = KB_ID_RE.match(title)
        kb_id = kb_id_match.group(1) if kb_id_match else path.stem
        cat_match = CATEGORY_RE.search(text)
        category = cat_match.group(1).strip() if cat_match else "Unknown"

        sections = split_article(text)
        for i, (section, body) in enumerate(sections):
            # Prefix the article title so every chunk carries its context
            # (e.g. a bare "## SLA" chunk still says which article it is from)
            documents.append(f"{title}\n\n{body}")
            ids.append(f"{path.stem}::{i:02d}")
            metadatas.append({
                "article": path.name,
                "kb_id": kb_id,
                "title": title,
                "category": category,
                "section": section,
                "chunk_index": i,
            })
        print(f"  {path.name:<28} -> {len(sections)} chunks")

    return ids, documents, metadatas


# --- 3: store in ChromaDB --------------------------------------------------
def build_collection(client, ids, documents, metadatas):
    # Start fresh each run so edited/removed articles don't leave stale chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    # Cosine distance, so confidence = 1 - distance is a 0..1 similarity
    try:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            configuration={"hnsw": {"space": "cosine"}},   # chromadb >= 1.0
        )
    except TypeError:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},             # older chromadb
        )

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


# --- 4: query --------------------------------------------------------------
def best_article(collection, query: str):
    """Return (article, confidence, section) for the best-matching article.

    Pulls the top chunks and keeps the highest-scoring chunk per article.
    """
    res = collection.query(
        query_texts=[query],
        n_results=min(TOP_K_CHUNKS, collection.count()),
        include=["metadatas", "distances"],
    )
    best: dict[str, tuple[float, str]] = {}
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        confidence = max(0.0, min(1.0, 1.0 - dist))
        article = meta["article"]
        if article not in best or confidence > best[article][0]:
            best[article] = (confidence, meta["section"])

    article, (confidence, section) = max(best.items(), key=lambda kv: kv[1][0])
    return article, confidence, section


def main():
    print(f"Reading KB articles from: {KB_DIR}")
    ids, documents, metadatas = load_chunks(KB_DIR)
    print(f"Total chunks: {len(ids)}\n")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    print("Embedding and storing chunks "
          "(first run downloads the embedding model, ~80 MB)...")
    collection = build_collection(client, ids, documents, metadatas)
    print(f"Collection '{COLLECTION_NAME}' now holds {collection.count()} chunks "
          f"(stored in {CHROMA_DIR})\n")

    print("=" * 78)
    print("Sample query results")
    print("=" * 78)
    passed = 0
    for n, (query, expected) in enumerate(SAMPLE_QUERIES, start=1):
        article, confidence, section = best_article(collection, query)
        ok = article == expected
        passed += ok
        print(f"\nQuery {n}: {query}")
        print(f"  Best match : {article}  (section: {section})")
        print(f"  Confidence : {confidence:.2%}")
        print(f"  Expected   : {expected}  [{'PASS' if ok else 'CHECK'}]")

    print(f"\n{passed}/{len(SAMPLE_QUERIES)} queries matched the expected article.")


if __name__ == "__main__":
    main()
