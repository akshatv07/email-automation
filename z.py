from pymilvus import connections, utility

connections.connect("default", host="localhost", port="19530")

collections = utility.list_collections()
print("\n📚 All collections in Milvus:")
for c in collections:
    print(f" - {c}")
