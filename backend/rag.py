import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pypdf
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class RAGPipeline:
    def __init__(self, chroma_path: str = "./chroma_db"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 lives in-memory; rebuilt from ChromaDB on every startup
        self._bm25_texts: List[str] = []
        self._bm25_ids: List[str] = []
        self._bm25_sources: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load_bm25_from_chroma()

    # ── Text helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    @staticmethod
    def _extract_text(file_path: str, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            reader = pypdf.PdfReader(file_path)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if ext in {".txt", ".md"}:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        raise ValueError(f"Unsupported file type: {ext}")

    # ── BM25 management ──────────────────────────────────────────────────────

    def _load_bm25_from_chroma(self):
        """Rebuild BM25 index from persisted ChromaDB on startup."""
        result = self.collection.get(include=["documents", "metadatas"])
        if not result["documents"]:
            return
        self._bm25_texts = result["documents"]
        self._bm25_ids = result["ids"]
        self._bm25_sources = [m["source"] for m in result["metadatas"]]
        self._bm25 = BM25Okapi([self._tokenize(t) for t in self._bm25_texts])

    def _rebuild_bm25(self):
        if self._bm25_texts:
            self._bm25 = BM25Okapi([self._tokenize(t) for t in self._bm25_texts])
        else:
            self._bm25 = None

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        text = self._extract_text(file_path, filename)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Document appears to be empty or unreadable.")

        doc_id = str(uuid.uuid4())
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        embeddings = self.embedder.encode(chunks).tolist()
        metadatas = [
            {"source": filename, "doc_id": doc_id, "chunk_idx": i}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas
        )

        # Incrementally update BM25
        self._bm25_texts.extend(chunks)
        self._bm25_ids.extend(ids)
        self._bm25_sources.extend([filename] * len(chunks))
        self._rebuild_bm25()

        return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _semantic_search(self, query: str, n: int) -> List[Dict]:
        total = self.collection.count()
        if total == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self.embedder.encode(query).tolist()],
            n_results=min(n, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "id": id_,
                "content": doc,
                "source": meta["source"],
            }
            for id_, doc, meta, dist in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def _bm25_search(self, query: str, n: int) -> List[Dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            {
                "id": self._bm25_ids[i],
                "content": self._bm25_texts[i],
                "source": self._bm25_sources[i],
            }
            for i in top_idx
            if scores[i] > 0  # skip zero-score (no keyword overlap at all)
        ]

    @staticmethod
    def _rrf_fusion(*ranked_lists: List[Dict], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion across any number of ranked lists.
        Each hit gets score += 1 / (k + rank + 1) per list it appears in.
        """
        by_id: Dict[str, Dict] = {}
        rrf: Dict[str, float] = {}

        for ranked in ranked_lists:
            for rank, hit in enumerate(ranked):
                id_ = hit["id"]
                by_id.setdefault(id_, hit)
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (k + rank + 1)

        return [
            by_id[id_]
            for id_ in sorted(rrf, key=lambda x: rrf[x], reverse=True)
        ]

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Full pipeline:
          1. Semantic search  ──┐
                                ├─ RRF fusion → candidates
          2. BM25 keyword     ──┘
          3. Cross-encoder rerank → top n_results
        """
        pool = max(n_results * 2, 10)  # smaller pool = faster reranking on CPU

        candidates = self._rrf_fusion(
            self._semantic_search(query, pool),
            self._bm25_search(query, pool),
        )[:pool]

        if not candidates:
            return []

        # Cross-encoder scores each (query, chunk) pair jointly — much more
        # accurate than embedding cosine similarity alone
        rerank_scores = self.reranker.predict(
            [(query, c["content"]) for c in candidates]
        ).tolist()

        for candidate, score in zip(candidates, rerank_scores):
            candidate["rerank_score"] = score

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return [
            {
                "content": r["content"],
                "source": r["source"],
                "score": round(r["rerank_score"], 3),
            }
            for r in reranked[:n_results]
        ]

    # ── Listing / deletion ────────────────────────────────────────────────────

    def list_documents(self) -> List[str]:
        result = self.collection.get(include=["metadatas"])
        if not result["metadatas"]:
            return []
        return sorted({m["source"] for m in result["metadatas"]})

    def delete_document(self, filename: str) -> int:
        result = self.collection.get(include=["metadatas"])
        ids_to_delete = [
            id_
            for id_, meta in zip(result["ids"], result["metadatas"])
            if meta["source"] == filename
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)

        # Remove from BM25
        keep = [s != filename for s in self._bm25_sources]
        self._bm25_texts = [t for t, k in zip(self._bm25_texts, keep) if k]
        self._bm25_ids = [i for i, k in zip(self._bm25_ids, keep) if k]
        self._bm25_sources = [s for s, k in zip(self._bm25_sources, keep) if k]
        self._rebuild_bm25()

        return len(ids_to_delete)
