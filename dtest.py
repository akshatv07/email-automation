from pymilvus import connections, utility, Collection

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List collections
collections = utility.list_collections()
print("✅ Collections in Milvus:\n")
for name in collections:
    print(f"📁 Collection: {name}")
    collection = Collection(name)
    print(f"   → Number of entities: {collection.num_entities}")
    print(f"   → Fields: {[f.name for f in collection.schema.fields]}")
    print()
