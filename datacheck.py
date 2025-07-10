from pymilvus import connections, utility, Collection
import pprint

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
print("✅ Connected to Milvus")

# List all collections
collections = utility.list_collections()
print(f"📁 Total collections found: {len(collections)}")

for name in collections:
    print(f"\n📂 Collection: {name}")
    try:
        # Load the collection
        collection = Collection(name)
        collection.load()

        # Print schema
        print("📑 Schema:")
        for field in collection.schema.fields:
            print(f"   - {field.name} ({field.dtype})")

        # Count number of entities
        count = collection.num_entities
        print(f"🔢 Total records: {count}")

        # Optionally: show a few sample entries
        if count > 0:
            print("🔍 Sample documents:")
            sample = collection.query(
                expr=None,
                offset=0,
                limit=min(3, count),  # Show first 3 entries
                output_fields=[field.name for field in collection.schema.fields]
            )
            pprint.pprint(sample)

        # Release memory
        collection.release()
    except Exception as e:
        print(f"❌ Error reading collection '{name}': {e}")

print("\n✅ Inspection complete.")
