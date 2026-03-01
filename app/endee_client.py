"""
Endee Vector Database Client
Handles all interactions with the Endee vector database.
"""

from endee import Endee
import pickle
import os
import uuid
import requests

from config import CACHE_DIR

class EndeeClient:
    """Wrapper for Endee vector database operations."""
    
    def __init__(self, base_url="http://localhost:8080", index_name=None, cache_file="chunk_cache.pkl"):
        self.endee = Endee()
        self.base_url = base_url
        self.index_name = index_name
        self.cache_file = os.path.join(CACHE_DIR, cache_file)

    def is_database_active(self):
        """
        Fast database health check using HTTP ping.
        Timeout set to 1 second.
        """
        
        try:
            requests.get(f"{self.base_url}/indexes", timeout=0.5)
            return True, None
        except:
            return False, "Database not reachable"
        
    def set_index_name(self, index_name):
        """Set or update the index name."""
        self.index_name = index_name
    
    def set_cache_file(self, cache_file):
        """Set or update the cache file path."""
        self.cache_file = os.path.join(CACHE_DIR, f"{cache_file}")
    
    def index_exists(self):
        """Check if the index already exists."""
        indexes = self.endee.list_indexes()["indexes"]
        return any(i["name"] == self.index_name for i in indexes)
    
    def create_index(self, dimension):
        """Create a new vector index."""
        self.endee.create_index(
            name=self.index_name,
            dimension=dimension,
            space_type="cosine",
            precision="float32"
        )
    
    def upsert_vectors(self, chunks, vectors):
        """Insert or update vectors in the index."""
        index = self.endee.get_index(self.index_name)
        id_map = {}
        
        payload = []
        for chunk, vector in zip(chunks, vectors):
            vector_id = str(uuid.uuid4())
            id_map[vector_id] = chunk
            payload.append({
                "id": vector_id,
                "vector": vector,
                "meta": {"text": chunk}
            })
        
        index.upsert(payload)
        
        # Save ID mapping to cache
        with open(self.cache_file, "wb") as f:
            pickle.dump(id_map, f)
        
        return len(payload)
    
    def load_cache(self):
        """Load the chunk cache from disk."""
        if not self.cache_file or not os.path.exists(self.cache_file):
            return {}
        with open(self.cache_file, "rb") as f:
            return pickle.load(f)
    
    def search_vectors(self, query_vector, top_k=6):
        """Search for similar vectors in the index."""
        index = self.endee.get_index(self.index_name)
        results = index.query(vector=query_vector, top_k=top_k)
        
        cache = self.load_cache()
        texts = []
        
        for result in results:
            vector_id = result["id"]
            if vector_id in cache:
                texts.append(cache[vector_id])
        
        return texts, len(results)
    
    def get_vector_count(self):
        """Get the total number of vectors in the index."""
        index = self.endee.get_index(self.index_name)
        return index.count
    
    def list_all_indexes(self):
        """List all available indexes."""
        return self.endee.list_indexes()