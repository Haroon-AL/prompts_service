import logging
from typing import List, Dict, Any, Tuple
import chromadb

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "attributes",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)

        # all-MiniLM-L6-v2 by default.
        self.collection = self.client.get_or_create_collection(name=collection_name)
        logger.info(
            f"Initialized VectorStore with collection '{collection_name}' at '{persist_directory}'"
        )

    def add_texts(
        self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]
    ):
        """Adds texts to the vector store."""
        if not texts:
            return

        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)
        logger.info(f"Upserted {len(texts)} documents into vector store")

    def search(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Searches for similar attributes."""
        results = self.collection.query(query_texts=[query], n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return [], []

        documents = results["documents"][0]
        metadatas = (
            results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
        )
        distances = (
            results["distances"][0] if results["distances"] else [0.0] * len(documents)
        )

        parsed_results = []
        for doc, meta in zip(documents, metadatas):
            parsed_results.append(
                {
                    "attribute_name": doc,  # the document is the attribute name
                    "prompt": meta.get("prompt", ""),
                    "system_role": meta.get("system_role", ""),
                }
            )

        return parsed_results, distances

    def count(self) -> int:
        return self.collection.count()
