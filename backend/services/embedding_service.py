import os
import time
import hashlib
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
    SearchParams,
    Range,
    FilterSelector,
)

load_dotenv()


class EmbeddingService:
    """Service for generating embeddings and managing Qdrant storage"""

    def __init__(
        self,
        gemini_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        collection_name: str = "papers_collection",
    ):
        """Initialize embedding service"""
        self.gemini_api_key = gemini_api_key
        self.collection_name = collection_name

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )

        # Ensure collection & indexes exist
        self._ensure_collection()

    def _generate_point_id(self, paper_id: str, chunk_id: str = "") -> int:
        """
        Generate a unique integer ID from paper_id and chunk_id
        Qdrant requires integer or UUID, not arbitrary strings
        """
        unique_str = f"{paper_id}_{chunk_id}"
        hash_obj = hashlib.md5(unique_str.encode())
        return int.from_bytes(hash_obj.digest()[:8], byteorder="big")

    def _ensure_payload_indexes(self):
        """Make sure all required payload indexes exist (idempotent)."""
        index_specs = [
            ("paper_id", PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("section", PayloadSchemaType.KEYWORD),
            ("paper_source", PayloadSchemaType.KEYWORD),
            ("paper_year", PayloadSchemaType.INTEGER),
        ]

        for field_name, schema in index_specs:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema,
                )
                print(f"✅ Created index for {field_name}")
            except Exception as e:
                # Likely "index already exists" – safe to ignore
                print(f"ℹ️ Index for {field_name} may already exist: {e}")

    def _ensure_collection(self):
        """Create Qdrant collection if it doesn't exist and ensure indexes."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")

                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=768,  # Gemini text-embedding-004 dimension
                        distance=Distance.COSINE,
                    ),
                )

                print(f"✅ Collection created: {self.collection_name}")
            else:
                print(f"✅ Collection exists: {self.collection_name}")

            # Ensure all needed indexes exist (runs even if collection pre-exists)
            self._ensure_payload_indexes()

        except Exception as e:
            print(f"Error ensuring collection: {e}")
            raise

    def generate_embedding(self, text: str, retry_count: int = 3) -> List[float]:
        """
        Generate embedding for a single text using Gemini
        """
        for attempt in range(retry_count):
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                )
                # Your previous code used result['embedding'], so keep that
                return result["embedding"]

            except Exception as e:
                print(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        """
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1} ({len(batch)} texts)...")

            for text in batch:
                try:
                    embedding = self.generate_embedding(text)
                    embeddings.append(embedding)
                    time.sleep(0.1)  # Rate limiting
                except Exception as e:
                    print(f"Failed to embed text: {e}")
                    embeddings.append([0.0] * 768)

        return embeddings

    def embed_paper_abstract(
        self,
        paper_id: str,
        paper_title: str,
        abstract: str,
        paper_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Embed ONLY the abstract of a paper (for per-paper Q&A)
        """
        try:
            # Check if already embedded (abstract-level)
            if self.is_paper_embedded(paper_id, check_full=False):
                print(f"Paper {paper_id} already embedded (abstract)")
                return {
                    "success": True,
                    "cached": True,
                    "chunks_embedded": 1,
                }

            print(f"Embedding abstract for: {paper_title}")
            embedding = self.generate_embedding(abstract)

            point_id = self._generate_point_id(paper_id, "abstract")

            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "paper_id": paper_id,
                    "chunk_id": "abstract",
                    "chunk_text": abstract,
                    "section": "abstract",
                    "page": 1,
                    "paper_title": paper_title,
                    "paper_year": paper_metadata.get("year"),
                    "paper_authors": paper_metadata.get("authors", []),
                    "paper_source": paper_metadata.get("source", "unknown"),
                    "chunk_type": "abstract",  # <-- canonical value
                },
            )

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            print(f"✅ Abstract embedded for: {paper_id}")

            return {
                "success": True,
                "cached": False,
                "chunks_embedded": 1,
            }

        except Exception as e:
            print(f"Error embedding abstract: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def embed_paper_full(
        self,
        paper_id: str,
        paper_title: str,
        chunks: List[Dict[str, Any]],
        paper_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Embed ALL chunks of a paper (for global Q&A)
        """
        try:
            # Check if full paper already embedded
            if self.is_paper_embedded(paper_id, check_full=True):
                print(f"Paper {paper_id} already fully embedded")
                return {
                    "success": True,
                    "cached": True,
                    "chunks_embedded": len(chunks),
                }

            print(f"Embedding {len(chunks)} chunks for: {paper_title}")

            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_embeddings_batch(texts, batch_size=100)

            points: List[PointStruct] = []
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = self._generate_point_id(paper_id, f"chunk_{idx}")

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "paper_id": paper_id,
                        "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                        "chunk_text": chunk["text"],
                        "section": chunk.get("section", "unknown"),
                        "page": chunk.get("page", 0),
                        "paper_title": paper_title,
                        "paper_year": paper_metadata.get("year"),
                        "paper_authors": paper_metadata.get("authors", []),
                        "paper_source": paper_metadata.get("source", "unknown"),
                        "chunk_type": "full_paper",
                    },
                )
                points.append(point)

            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                print(
                    f"Uploaded batch {i // batch_size + 1}/"
                    f"{(len(points) + batch_size - 1) // batch_size}"
                )

            print(f"✅ Embedded {len(chunks)} chunks for: {paper_id}")

            return {
                "success": True,
                "cached": False,
                "chunks_embedded": len(chunks),
            }

        except Exception as e:
            print(f"Error embedding paper: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def is_paper_embedded(self, paper_id: str, check_full: bool = False) -> bool:
        """
        Check if a paper is already embedded
        - check_full = False → look for abstract chunk
        - check_full = True  → look for full_paper chunks
        """
        try:
            must_conditions = [
                FieldCondition(
                    key="paper_id",
                    match=MatchValue(value=paper_id),
                )
            ]

            if check_full:
                must_conditions.append(
                    FieldCondition(
                        key="chunk_type",
                        match=MatchValue(value="full_paper"),
                    )
                )
            else:
                must_conditions.append(
                    FieldCondition(
                        key="chunk_type",
                        match=MatchValue(value="abstract"),
                    )
                )

            search_filter = Filter(must=must_conditions)

            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1,
                scroll_filter=search_filter,  # new-style param name
                with_payload=False,
                with_vectors=False,
            )

            return len(points) > 0

        except Exception as e:
            print(f"Error checking if paper embedded: {e}")
            return False

    def search_within_paper(
        self,
        paper_id: str,
        query: str,
        top_k: int = 6,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search within a specific paper (per-paper Q&A)
        """
        try:
            query_embedding = self.generate_embedding(query)

            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            )

            resp = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=SearchParams(hnsw_ef=128),
            )

            formatted_results: List[Dict[str, Any]] = []
            for pt in resp.points:
                pl = pt.payload or {}
                formatted_results.append(
                    {
                        "chunk_id": pl.get("chunk_id"),
                        "text": pl.get("chunk_text"),
                        "section": pl.get("section"),
                        "page": pl.get("page"),
                        "score": pt.score,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error searching within paper: {e}")
            return []

    def search_global(
        self,
        query: str,
        top_k: int = 15,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across ALL papers (global Q&A)
        """
        try:
            query_embedding = self.generate_embedding(query)

            search_filter: Optional[Filter] = None
            if filters:
                conditions: List[FieldCondition] = []

                if "year_after" in filters:
                    try:
                        year_val = int(filters["year_after"])
                        conditions.append(
                            FieldCondition(
                                key="paper_year",
                                range={"gte": year_val},
                            )
                        )
                    except (ValueError, TypeError):
                        pass

                if "source" in filters:
                    conditions.append(
                        FieldCondition(
                            key="paper_source",
                            match=MatchValue(value=filters["source"]),
                        )
                    )

                if conditions:
                    search_filter = Filter(must=conditions)

            resp = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                search_params=SearchParams(hnsw_ef=128),
            )

            formatted_results: List[Dict[str, Any]] = []
            for pt in resp.points:
                pl = pt.payload or {}
                formatted_results.append(
                    {
                        "paper_id": pl.get("paper_id"),
                        "paper_title": pl.get("paper_title"),
                        "paper_year": pl.get("paper_year"),
                        "chunk_id": pl.get("chunk_id"),
                        "text": pl.get("chunk_text"),
                        "section": pl.get("section"),
                        "page": pl.get("page"),
                        "score": pt.score,
                    }
                )

            return formatted_results

        except Exception as e:
            print(f"Error in global search: {e}")
            return []

    def delete_paper(self, paper_id: str) -> bool:
        """
        Delete all embeddings for a paper
        """
        try:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="paper_id",
                        match=MatchValue(value=paper_id),
                    )
                ]
            )

            selector = FilterSelector(filter=search_filter)

            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=selector,
            )

            print(f"✅ Deleted embeddings for: {paper_id}")
            return True

        except Exception as e:
            print(f"Error deleting paper: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Qdrant collection"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            # Handle both single-vector and multi-vector configs
            vectors_cfg = collection_info.config.params.vectors
            if isinstance(vectors_cfg, dict):
                # named vectors config
                any_vec = next(iter(vectors_cfg.values()))
                vector_size = any_vec.size
            else:
                vector_size = vectors_cfg.size

            return {
                "total_vectors": collection_info.points_count,
                "collection_name": self.collection_name,
                "vector_size": vector_size,
                "status": "healthy",
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }