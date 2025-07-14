from pymilvus import connections, utility

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# List all collections
collections = utility.list_collections()

if not collections:
    print("No collections found. Milvus DB is already empty.")
else:
    for coll in collections:
        print(f"Dropping collection: {coll}")
        utility.drop_collection(coll)
    print("All collections have been deleted. Milvus DB is now empty.")
