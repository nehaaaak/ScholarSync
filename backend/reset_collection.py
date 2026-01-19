"""
Reset Qdrant Collection
Deletes old collection and creates new one with proper indexes
Run this ONCE before testing
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "papers_collection"

print(f"Connecting to Qdrant: {QDRANT_URL}")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Delete old collection if exists
try:
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]
    
    if COLLECTION_NAME in collection_names:
        print(f"Deleting old collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        print(f"✅ Deleted")
    else:
        print(f"No existing collection found")
        
except Exception as e:
    print(f"Error checking collections: {e}")

# Create new collection
try:
    print(f"\nCreating new collection: {COLLECTION_NAME}")
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,  # Gemini embedding dimension
            distance=Distance.COSINE
        )
    )
    
    print(f"✅ Collection created")
    
    # Create payload index for paper_id
    print(f"Creating index for paper_id field...") 
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="paper_id",
        field_schema=PayloadSchemaType.KEYWORD
    )
    print(f"✅ Index for paper_id created")

    # NEW: index for chunk_type
    print("Creating index for chunk_type field...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_type",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("✅ Index for chunk_type created")

    # Optional but useful: index for section
    print("Creating index for section field...")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="section",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    print("✅ Index for section created")
    
    # Verify
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"\n✅ Collection ready!")
    print(f"   Vectors: {collection_info.points_count}")
    print(f"   Vector size: {collection_info.config.params.vectors.size}")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n🎉 Done! You can now run test_day7.py")