from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
points = client.scroll(collection_name="research_Alfa_Equity", limit=5)

for point in points[0]:
    text = point.payload.get("text", "")
    print(f"ID: {point.id}")
    print(f"Источник: {point.payload.get('source', 'N/A')}")
    print("Первые 700 символов:")
    print(repr(text[:700]))  # repr покажет спецсимволы!
    print("-" * 60)