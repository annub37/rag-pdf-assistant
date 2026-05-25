import chromadb
from app.config import settings

client = chromadb.PersistentClient(path=settings.chroma_dir)
col = client.get_or_create_collection("documents")
count = col.count()
print(f"Total documents in ChromaDB: {count}")

if count > 0:
    results = col.peek(limit=5)
    print(f"\nFirst {min(5, count)} chunks:")
    for i in range(len(results["ids"])):
        meta = results["metadatas"][i]
        text_preview = results["documents"][i][:100]
        cid = results["ids"][i]
        fid = meta.get("file_id", "?")
        page = meta.get("page_number", "?")
        print(f"  [{i+1}] ID: {cid}")
        print(f"      File: {fid} | Page: {page}")
        print(f"      Text: {text_preview}...")
        print()

    all_data = col.get(include=["metadatas"])
    file_ids = set(m["file_id"] for m in all_data["metadatas"])
    print(f"Unique PDFs stored: {len(file_ids)}")
    for fid in file_ids:
        chunks_for_file = sum(1 for m in all_data["metadatas"] if m["file_id"] == fid)
        print(f"  - {fid} ({chunks_for_file} chunks)")
else:
    print("No documents found in ChromaDB.")
