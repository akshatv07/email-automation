from pymilvus import connections, utility

# ✅ Step 1: Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
print("✅ Connected to Milvus")

# ✅ Step 2: List all collections
collections = utility.list_collections()
print(f"📄 Found {len(collections)} collections.")

# ✅ Step 3: Drop each collection
for name in collections:
    try:
        utility.drop_collection(name)
        print(f"🗑️ Dropped collection: {name}")
    except Exception as e:
        print(f"❌ Failed to drop collection '{name}': {e}")

print("✅ All collections dropped.")
